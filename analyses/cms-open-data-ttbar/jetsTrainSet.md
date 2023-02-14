---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# ttbar Analysis - Jet-Parton Assignment Training

This is the training notebook for the jet-parton assignment task. The goal is to associate the leading four jets in each event to their associated parent particles. We are trying to assign jets according to the labels in the diagram below:

<img src="utils/ttbar.png" alt="ttbar_labels" width="500"/>

top1 and top2 jets do not necessarily correspond to top/antitop, respectively. The top1 jet is defined as having a lepton/neutrino pair as cousins, where the top2 jet is defined as having two jets as cousins. The W jets are not distinguished from each other.

The strategy for solving this problem is to train a boosted decision tree to find the correct assignments for each jet. Since we consider four jets per event with three unique labels (W, top1, and top2), there are twelve possible combinations of assignments:

<img src="utils/jetcombinations.png" alt="jetcombinations" width="700"/>

The combination with the highest BDT score will be selected for each event.
____

The workflow for this training notebook is outlined as follows:
* Load data and calculate training features and labels using `coffea`/`dask`
* Optimize BDT (`xgboost` model) using `hyperopt` (TODO: Track using `mlflow`)
* Save best model (TODO: save to `onnx`)

```python
import asyncio
import time
import logging

import vector; vector.register_awkward()

from coffea.nanoevents import NanoAODSchema
from coffea import processor
import awkward as ak
import numpy as np
import hist
import json
import matplotlib.pyplot as plt
import uproot

import utils

from dask.distributed import Client
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, train_test_split
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier
import onnx
```

```python
### GLOBAL CONFIGURATION

# input files per process, set to e.g. 10 (smaller number = faster, want to use larger number for training)
N_FILES_MAX_PER_SAMPLE = 1
# set to "dask" for DaskExecutor, "futures" for FuturesExecutor
EXEC = "futures"

# number of cores if using FuturesExecutor
NUM_CORES = 4

# chunk size to use
CHUNKSIZE = 100_000

# analysis facility: set to "coffea_casa" for coffea-casa environments, "EAF" for FNAL, "local" for local setups
AF = "coffea_casa"

# optional mlflow logging
USE_MLFLOW = False
```

```python
permutations_dict, labels_dict = utils.get_permutations_dict(4, include_labels=True)
```

```python
# these matrices tell you the overlap between the predicted label (rows) and truth label (columns)
# the "score" in each matrix entry is the number of jets which are assigned correctly
evaluation_matrices = {} # overall event score

for n in range(4,4+1):
    print("n = ", n)
    evaluation_matrix = np.zeros((len(permutations_dict[n]),len(permutations_dict[n])))
    
    for i in range(len(permutations_dict[n])):
        for j in range(len(permutations_dict[n])):
            evaluation_matrix[i,j]=sum(np.equal(labels_dict[n][i], labels_dict[n][j]))
    
    evaluation_matrices[n] = evaluation_matrix
```

```python
## functions for calculating features and labels for the BDT
def training_filter(jets, electrons, muons, genparts, even):
    '''
    Filters events down to training set and calculates jet-level labels
    
    Args:
        jets: selected jets after region filter (and selecting leading four for each event)
        electrons: selected electrons after region filter
        muons: selected muons after region filter
        genparts: selected genpart after region filter
    
    Returns:
        jets, electrons, muons, labels
    '''
    #### filter genPart to valid matching candidates ####

    # get rid of particles without parents
    genpart_parent = genparts.distinctParent
    genpart_filter = np.invert(ak.is_none(genpart_parent, axis=1))
    genparts = genparts[genpart_filter]
    genpart_parent = genparts.distinctParent

    # ensure that parents are top quark or W
    genpart_filter2 = ((np.abs(genpart_parent.pdgId)==6) | (np.abs(genpart_parent.pdgId)==24))
    genparts = genparts[genpart_filter2]

    # ensure particle itself is a quark
    genpart_filter3 = ((np.abs(genparts.pdgId)<7) & (np.abs(genparts.pdgId)>0))
    genparts = genparts[genpart_filter3]

    # get rid of duplicates
    genpart_filter4 = genparts.hasFlags("isLastCopy")
    genparts = genparts[genpart_filter4]
            
        
    #### get jet-level labels and filter events to training set
        
    # match jets to nearest valid genPart candidate
    nearest_genpart = jets.nearest(genparts, threshold=0.4)
    nearest_parent = nearest_genpart.distinctParent # parent of matched particle

    parent_pdgid = nearest_parent.pdgId # pdgId of parent particle
    grandchild_pdgid = nearest_parent.distinctChildren.distinctChildren.pdgId # pdgId of particle's parent's grandchildren

    grandchildren_flat = np.abs(ak.flatten(grandchild_pdgid,axis=-1)) # flatten innermost axis for convenience

    # if particle has a cousin that is a lepton
    has_lepton_cousin = (ak.sum(((grandchildren_flat%2==0) & (grandchildren_flat>10) & (grandchildren_flat<19)),
                                axis=-1)>0)
    # if particle has a cousin that is a neutrino
    has_neutrino_cousin = (ak.sum(((grandchildren_flat%2==1) & (grandchildren_flat>10) & (grandchildren_flat<19)),
                                  axis=-1)>0)

    # if a particle has a lepton cousin and a neutrino cousin
    has_both_cousins = ak.fill_none((has_lepton_cousin & has_neutrino_cousin), False).to_numpy()

    # get labels from parent pdgId (fill none with 100 to filter out events with those jets)
    labels = np.abs(ak.fill_none(parent_pdgid,100).to_numpy())
    labels[has_both_cousins] = -6 # assign jets with both cousins as top1 (not necessarily antiparticle)

    training_event_filter = (np.sum(labels,axis=1)==48) # events with a label sum of 48 have the correct particles
            
    # filter events
    jets = jets[training_event_filter]
    electrons = electrons[training_event_filter]
    muons = muons[training_event_filter]
    labels = labels[training_event_filter]
    even = even[training_event_filter]
    
    return jets, electrons, muons, labels, even
    

def get_training_set(jets, electrons, muons, labels, permutations_dict, labels_dict):
    '''
    Calculate features for each of the 12 combinations per event and calculates combination-level labels
    
    Args:
        jets: selected jets after training filter
        electrons: selected electrons after training filter
        muons: selected muons after training filter
        labels: jet-level labels output by training_filter
    
    Returns:
        features, labels (flattened to remove event level)
    '''
    
    # calculate number of jets in each event
    njet = ak.num(jets).to_numpy()
    # don't consider every jet for events with high jet multiplicity
    njet[njet>max(permutations_dict.keys())] = max(permutations_dict.keys())
    # create awkward array of permutation indices
    perms = ak.Array([permutations_dict[n] for n in njet])
    perm_counts = ak.num(perms)
    
    
    #### calculate features ####
    
    #### calculate features ####
    features = np.zeros((sum(perm_counts),19))
    
    # grab lepton info
    leptons = ak.flatten(ak.concatenate((electrons, muons),axis=1),axis=-1)

    # delta R between top1 and lepton
    features[:,0] = ak.flatten(np.sqrt((leptons.eta - jets[perms[...,3]].eta)**2 + 
                                       (leptons.phi - jets[perms[...,3]].phi)**2)).to_numpy()

    # delta R between the two W
    features[:,1] = ak.flatten(np.sqrt((jets[perms[...,0]].eta - jets[perms[...,1]].eta)**2 + 
                                       (jets[perms[...,0]].phi - jets[perms[...,1]].phi)**2)).to_numpy()

    # delta R between W and top2
    features[:,2] = ak.flatten(np.sqrt((jets[perms[...,0]].eta - jets[perms[...,2]].eta)**2 + 
                                       (jets[perms[...,0]].phi - jets[perms[...,2]].phi)**2)).to_numpy()
    features[:,3] = ak.flatten(np.sqrt((jets[perms[...,1]].eta - jets[perms[...,2]].eta)**2 + 
                                       (jets[perms[...,1]].phi - jets[perms[...,2]].phi)**2)).to_numpy()

    # delta phi between top1 and lepton
    features[:,4] = ak.flatten(np.abs(leptons.phi - jets[perms[...,3]].phi)).to_numpy()

    # delta phi between the two W
    features[:,5] = ak.flatten(np.abs(jets[perms[...,0]].phi - jets[perms[...,1]].phi)).to_numpy()

    # delta phi between W and top2
    features[:,6] = ak.flatten(np.abs(jets[perms[...,0]].phi - jets[perms[...,2]].phi)).to_numpy()
    features[:,7] = ak.flatten(np.abs(jets[perms[...,1]].phi - jets[perms[...,2]].phi)).to_numpy()


    # combined mass of top1 and lepton
    features[:,8] = ak.flatten((leptons + jets[perms[...,3]]).mass).to_numpy()

    # combined mass of W
    features[:,9] = ak.flatten((jets[perms[...,0]] + jets[perms[...,1]]).mass).to_numpy()

    # combined mass of W and top2
    features[:,10] = ak.flatten((jets[perms[...,0]] + jets[perms[...,1]] + 
                                 jets[perms[...,2]]).mass).to_numpy()


    # pt of every jet
    features[:,11] = ak.flatten(jets[perms[...,0]].pt).to_numpy()
    features[:,12] = ak.flatten(jets[perms[...,1]].pt).to_numpy()
    features[:,13] = ak.flatten(jets[perms[...,2]].pt).to_numpy()
    features[:,14] = ak.flatten(jets[perms[...,3]].pt).to_numpy()


    # mass of every jet
    features[:,15] = ak.flatten(jets[perms[...,0]].mass).to_numpy()
    features[:,16] = ak.flatten(jets[perms[...,1]].mass).to_numpy()
    features[:,17] = ak.flatten(jets[perms[...,2]].mass).to_numpy()
    features[:,18] = ak.flatten(jets[perms[...,3]].mass).to_numpy()
    
    #### calculate combination-level labels ####
    permutation_labels = np.array(labels_dict[4])
    
    # which combination does the truth label correspond to?
    which_combination = np.zeros(len(jets), dtype=int)
    # no correct matches
    which_anti_combination = np.zeros(labels.shape[0], dtype=int)
    for i in range(12):
        which_combination[(labels==permutation_labels[i,:]).all(1)] = i
        which_anti_combination[np.invert((labels==permutation_labels[i,:]).any(1))] = i

    # convert to combination-level truth label (-1, 0 or 1)
    which_combination = list(zip(range(len(jets),), which_combination))
    which_anti_combination = list(zip(range(labels.shape[0],), which_anti_combination))
    
    truth_labels = -1*np.ones((len(jets),12))
    for i,tpl in enumerate(which_combination):
        truth_labels[tpl]=1
    for i,tpl in enumerate(which_anti_combination):
        truth_labels[tpl]=0
        
        
    #### flatten to combinations (easy to unflatten since each event always has 12 combinations) ####
    labels = truth_labels.reshape((truth_labels.shape[0]*truth_labels.shape[1],1))
    
    return features, labels, which_combination
```

### Defining a `coffea` Processor

The processor returns the training features and labels we will use in our BDT

```python
processor_base = processor.ProcessorABC
class JetClassifier(processor_base):
    def __init__(self, permutations_dict, labels_dict):
        super().__init__()
        self.permutations_dict = permutations_dict
        self.labels_dict = labels_dict
    
    def process(self, events):
        
        process = events.metadata["process"]  # "ttbar" etc.
        variation = events.metadata["variation"]  # "nominal" etc.
        
        # normalization for MC
        x_sec = events.metadata["xsec"]
        nevts_total = events.metadata["nevts"]
        lumi = 3378 # /pb
        xsec_weight = x_sec * lumi / nevts_total
            
        events["pt_nominal"] = 1.0
        pt_variations = ["pt_nominal"] if variation == "nominal" else ["pt_nominal"]
        for pt_var in pt_variations:
            
            # filter electrons, muons, and jets by pT
            selected_electrons = events.Electron[(events.Electron.pt > 30) & (np.abs(events.Electron.eta)<2.1) & 
                                                 (events.Electron.cutBased==4) & (events.Electron.sip3d < 4)]
            selected_muons = events.Muon[(events.Muon.pt > 30) & (np.abs(events.Muon.eta)<2.1) & (events.Muon.tightId) & 
                                         (events.Muon.sip3d < 4) & (events.Muon.pfRelIso04_all < 0.15)]
            jet_filter = (events.Jet.pt > 30) & (np.abs(events.Jet.eta) < 2.4)
            selected_jets = events.Jet[jet_filter]
            selected_genpart = events.GenPart
            even = (events.event%2==0)
            
            # single lepton requirement
            event_filters = ((ak.count(selected_electrons.pt, axis=1) + ak.count(selected_muons.pt, axis=1)) == 1)
            # require at least 4 jets
            event_filters = event_filters & (ak.count(selected_jets.pt, axis=1) >= 4)
            # require at least one jet above B_TAG_THRESHOLD
            B_TAG_THRESHOLD = 0.5
            event_filters = event_filters & (ak.sum(selected_jets.btagCSVV2 >= B_TAG_THRESHOLD, axis=1) >= 1)
            
            # apply event filters
            selected_events = events[event_filters]
            selected_electrons = selected_electrons[event_filters]
            selected_muons = selected_muons[event_filters]
            selected_jets = selected_jets[event_filters]
            selected_genpart = selected_genpart[event_filters]
            even = even[event_filters]
            
            ### only consider 4j2b region
            region_filter = ak.sum(selected_jets.btagCSVV2 > B_TAG_THRESHOLD, axis=1) >= 2 # at least two b-tagged jets
            selected_jets_region = selected_jets[region_filter][:,:4] # only keep top 4 jets
            selected_electrons_region = selected_electrons[region_filter]
            selected_muons_region = selected_muons[region_filter]
            selected_genpart_region = selected_genpart[region_filter]
            even = even[region_filter]
            
            # filter events and calculate labels
            jets, electrons, muons, labels, even = training_filter(selected_jets_region, 
                                                                   selected_electrons_region, 
                                                                   selected_muons_region, 
                                                                   selected_genpart_region,
                                                                   even)
            
            # calculate features and labels
            features, labels, which_combination = get_training_set(jets, electrons, muons, labels,
                                                                   self.permutations_dict, self.labels_dict)
    
            # calculate mbjj
            # reconstruct hadronic top as bjj system with largest pT
            # the jet energy scale / resolution effect is not propagated to this observable at the moment
            trijet = ak.combinations(jets, 3, fields=["j1", "j2", "j3"])  # trijet candidates
            trijet["p4"] = trijet.j1 + trijet.j2 + trijet.j3  # calculate four-momentum of tri-jet system
            trijet["max_btag"] = np.maximum(trijet.j1.btagCSVV2, np.maximum(trijet.j2.btagCSVV2, trijet.j3.btagCSVV2))
            trijet = trijet[trijet.max_btag > B_TAG_THRESHOLD]  # at least one-btag in trijet candidates
            # pick trijet candidate with largest pT and calculate mass of system
            trijet_mass = trijet["p4"][ak.argmax(trijet.p4.pt, axis=1, keepdims=True)].mass
            observable = ak.flatten(trijet_mass)
            
        output = {"nevents": {events.metadata["dataset"]: len(events)},
                  "features": {events.metadata["dataset"]: features.tolist()},
                  "labels": {events.metadata["dataset"]: labels.tolist()},
                  "observable": {events.metadata["dataset"]: observable.to_list()},
                  "even": {events.metadata["dataset"]: even.to_list()}}
            
        return output
        
    def postprocess(self, accumulator):
        return accumulator
```

### "Fileset" construction and metadata

Here, we gather all the required information about the files we want to process: paths to the files and asociated metadata.

```python
fileset = utils.construct_fileset(N_FILES_MAX_PER_SAMPLE, 
                                  use_xcache=False, 
                                  json_file = 'ntuples_nanoaod_agc.json')

# get rid of everything except ttbar__nominal for training purposes
fileset_keys = list(fileset.keys())
for key in fileset_keys:
    if key!="ttbar__nominal":
        fileset.pop(key)
```

```python tags=[]
fileset
```

### Execute the data delivery pipeline

```python tags=[]
NanoAODSchema.warn_missing_crossrefs = False

if EXEC == "futures":
    executor = processor.FuturesExecutor(workers=NUM_CORES)
elif EXEC == "dask":
    executor = processor.DaskExecutor(client=utils.get_client(AF))
    
run = processor.Runner(executor=executor, schema=NanoAODSchema, savemetrics=True, metadata_cache={}, 
                       chunksize=CHUNKSIZE)

# preprocess
filemeta = run.preprocess(fileset, treename="Events")

# process
output, metrics = run(fileset, 
                      "Events", 
                      processor_instance = JetClassifier(permutations_dict, labels_dict))
```

```python
import pickle
pickle.dump(output, open("output_temp.p", "wb"))
```

```python
import pickle
output = pickle.load(open("output_temp.p", "rb"))
```

```python
# grab features and labels and convert to np array
features = np.array(output['features']['ttbar__nominal'])
labels = np.array(output['labels']['ttbar__nominal'])
even = np.array(output['even']['ttbar__nominal'])

labels = labels.reshape((len(labels),))
even = np.repeat(even, 12)
```

```python
# investigate labels
print(len(labels))
print(len(labels)/12)
print(sum(labels==1))
```

The key for the labeling scheme is as follows

* 1: all jet assignments are correct
* 0: some jet assignments are correct (one or two are correct, others are incorrect)
* -1: all jet assignments are incorrect

There are twelve combinations for each event, so each event will have 1 correct combination, 2 completely incorrect combinations, and 9 partially correct combinations.

```python
# separate by label for plotting
all_correct = features[labels==1,:]
some_correct = features[labels==-1,:]
none_correct = features[labels==0,:]
```

# Histograms of Training Variables
To vizualize the separation power of the different variables, histograms are created for each of the three labels. Only `all_correct` and `none_correct` are used for training purposes.

```python
#### delta R histogram ####

# binning
deltar_low = 0.0
deltar_high = 8.0
deltar_numbins = 100
legend_list = ["All Matches Correct", "Some Matches Correct", "No Matches Correct"]

# define histogram
h = hist.Hist(
    hist.axis.Regular(deltar_numbins, deltar_low, deltar_high, name="deltar", label="$\Delta R$", flow=False),
    hist.axis.StrCategory(legend_list, name="truthlabel", label="Truth Label"),
    hist.axis.StrCategory(["top1_lepton","W_W","top2_W"], name="category", label="Category"),
)

# fill histogram
h.fill(deltar = all_correct[:,0], category="top1_lepton", truthlabel="All Matches Correct")
h.fill(deltar = some_correct[:,0], category="top1_lepton", truthlabel="Some Matches Correct")
h.fill(deltar = none_correct[:,0], category="top1_lepton", truthlabel="No Matches Correct")
h.fill(deltar = all_correct[:,1], category="W_W", truthlabel="All Matches Correct")
h.fill(deltar = some_correct[:,1], category="W_W", truthlabel="Some Matches Correct")
h.fill(deltar = none_correct[:,1], category="W_W", truthlabel="No Matches Correct")
h.fill(deltar = all_correct[:,2], category="top2_W", truthlabel="All Matches Correct")
h.fill(deltar = some_correct[:,2], category="top2_W", truthlabel="Some Matches Correct")
h.fill(deltar = none_correct[:,2], category="top2_W", truthlabel="No Matches Correct")
h.fill(deltar = all_correct[:,3], category="top2_W", truthlabel="All Matches Correct")
h.fill(deltar = some_correct[:,3], category="top2_W", truthlabel="Some Matches Correct")
h.fill(deltar = none_correct[:,3], category="top2_W", truthlabel="No Matches Correct")

# make plots
fig,ax = plt.subplots(1,1,figsize=(8,4))
h[0j::hist.rebin(2), :, "top1_lepton"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("$\Delta R$ between top1 jet and lepton")
fig.show()

fig,ax = plt.subplots(1,1,figsize=(8,4))
h[0j::hist.rebin(2), :, "W_W"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("$\Delta R$ between the two W jets")
fig.show()

fig,ax = plt.subplots(1,1,figsize=(8,4))
h[0j::hist.rebin(2), :, "top2_W"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("$\Delta R$ between W jet and top2 jet")
fig.show()
```

```python
#### delta phi histogram ####

# binning
deltaphi_low = 0.0
deltaphi_high = 2*np.pi
deltaphi_numbins = 100
legend_list = ["All Matches Correct", "Some Matches Correct", "No Matches Correct"]

# define histogram
h = hist.Hist(
    hist.axis.Regular(deltaphi_numbins, deltaphi_low, deltaphi_high, name="deltaphi", label="$\Delta \phi$", flow=False),
    hist.axis.StrCategory(legend_list, name="truthlabel", label="Truth Label"),
    hist.axis.StrCategory(["top1_lepton","W_W","top2_W"], name="category", label="Category"),
)

# fill histogram
h.fill(deltaphi = all_correct[:,4], category="top1_lepton", truthlabel="All Matches Correct")
h.fill(deltaphi = some_correct[:,4], category="top1_lepton", truthlabel="Some Matches Correct")
h.fill(deltaphi = none_correct[:,4], category="top1_lepton", truthlabel="No Matches Correct")
h.fill(deltaphi = all_correct[:,5], category="W_W", truthlabel="All Matches Correct")
h.fill(deltaphi = some_correct[:,5], category="W_W", truthlabel="Some Matches Correct")
h.fill(deltaphi = none_correct[:,5], category="W_W", truthlabel="No Matches Correct")
h.fill(deltaphi = all_correct[:,6], category="top2_W", truthlabel="All Matches Correct")
h.fill(deltaphi = some_correct[:,6], category="top2_W", truthlabel="Some Matches Correct")
h.fill(deltaphi = none_correct[:,6], category="top2_W", truthlabel="No Matches Correct")
h.fill(deltaphi = all_correct[:,7], category="top2_W", truthlabel="All Matches Correct")
h.fill(deltaphi = some_correct[:,7], category="top2_W", truthlabel="Some Matches Correct")
h.fill(deltaphi = none_correct[:,7], category="top2_W", truthlabel="No Matches Correct")

# make plots
fig,ax = plt.subplots(1,1,figsize=(8,4))
h[0j::hist.rebin(2), :, "top1_lepton"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("$\Delta \phi$ between top1 jet and lepton")
fig.show()

fig,ax = plt.subplots(1,1,figsize=(8,4))
h[0j::hist.rebin(2), :, "W_W"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("$\Delta \phi$ between the two W jets")
fig.show()

fig,ax = plt.subplots(1,1,figsize=(8,4))
h[0j::hist.rebin(2), :, "top2_W"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("$\Delta \phi$ between W jet and top2 jet")
fig.show()
```

```python
#### mass histogram ####

# binning
combinedmass_low = 0.0
combinedmass_high = 1500.0
combinedmass_numbins = 200
legend_list = ["All Matches Correct", "Some Matches Correct", "No Matches Correct", "Jet Triplet with Largest pT"]

# define histogram
h = hist.Hist(
    hist.axis.Regular(combinedmass_numbins, combinedmass_low, combinedmass_high, 
                      name="combinedmass", label="Combined Mass [GeV]", flow=False),
    hist.axis.StrCategory(legend_list, name="truthlabel", label="Truth Label"),
    hist.axis.StrCategory(["top1_lepton","W_W","top2_W_W"], name="category", label="Category"),
)

# fill histogram
h.fill(combinedmass = all_correct[:,8], category="top1_lepton", truthlabel="All Matches Correct")
h.fill(combinedmass = some_correct[:,8], category="top1_lepton", truthlabel="Some Matches Correct")
h.fill(combinedmass = none_correct[:,8], category="top1_lepton", truthlabel="No Matches Correct")
h.fill(combinedmass = all_correct[:,9], category="W_W", truthlabel="All Matches Correct")
h.fill(combinedmass = some_correct[:,9], category="W_W", truthlabel="Some Matches Correct")
h.fill(combinedmass = none_correct[:,9], category="W_W", truthlabel="No Matches Correct")
h.fill(combinedmass = all_correct[:,10], category="top2_W_W", truthlabel="All Matches Correct")
h.fill(combinedmass = some_correct[:,10], category="top2_W_W", truthlabel="Some Matches Correct")
h.fill(combinedmass = none_correct[:,10], category="top2_W_W", truthlabel="No Matches Correct")
h.fill(combinedmass = output["observable"]["ttbar__nominal"], category="top2_W_W", truthlabel="Jet Triplet with Largest pT")

# make plots
fig,ax = plt.subplots(1,1,figsize=(8,4))
h[:, :, "top1_lepton"].plot(density=True, ax=ax)
ax.legend(legend_list[:-1])
ax.set_title("Combined mass of top1 jet and lepton")
ax.set_xlim([0,400])
fig.show()

fig,ax = plt.subplots(1,1,figsize=(8,4))
h[:, :, "W_W"].plot(density=True, ax=ax)
ax.legend(legend_list[:-1])
ax.set_title("Combined mass of the two W jets")
ax.set_xlim([0,400])
fig.show()

fig,ax = plt.subplots(1,1,figsize=(8,4))
h[:, :, "top2_W_W"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("Combined mass of W jets and top2 jet")
ax.set_xlim([0,600])
fig.show()
```

```python
#### pT histogram ####

# binning
pt_low = 25.0
pt_high = 300.0
pt_numbins = 100
legend_list = ["All Matches Correct", "Some Matches Correct", "No Matches Correct"]

# define histogram
h = hist.Hist(
    hist.axis.Regular(pt_numbins, pt_low, pt_high, 
                      name="jetpt", label="Jet $p_T$ [GeV]", flow=False),
    hist.axis.StrCategory(legend_list, name="truthlabel", label="Truth Label"),
    hist.axis.StrCategory(["W","top1","top2"], name="category", label="Category"),
)

# fill histogram
h.fill(jetpt = all_correct[:,11], category="W", truthlabel="All Matches Correct")
h.fill(jetpt = some_correct[:,11], category="W", truthlabel="Some Matches Correct")
h.fill(jetpt = none_correct[:,11], category="W", truthlabel="No Matches Correct")
h.fill(jetpt = all_correct[:,12], category="W", truthlabel="All Matches Correct")
h.fill(jetpt = some_correct[:,12], category="W", truthlabel="Some Matches Correct")
h.fill(jetpt = none_correct[:,12], category="W", truthlabel="No Matches Correct")
h.fill(jetpt = all_correct[:,13], category="top2", truthlabel="All Matches Correct")
h.fill(jetpt = some_correct[:,13], category="top2", truthlabel="Some Matches Correct")
h.fill(jetpt = none_correct[:,13], category="top2", truthlabel="No Matches Correct")
h.fill(jetpt = all_correct[:,14], category="top1", truthlabel="All Matches Correct")
h.fill(jetpt = some_correct[:,14], category="top1", truthlabel="Some Matches Correct")
h.fill(jetpt = none_correct[:,14], category="top1", truthlabel="No Matches Correct")

# make plots
fig,ax = plt.subplots(1,1,figsize=(8,4))
h[:, :, "W"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("W Jet $p_T$")
ax.set_xlim([25,300])
fig.show()

fig,ax = plt.subplots(1,1,figsize=(8,4))
h[:, :, "top2"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("top2 Jet $p_T$")
ax.set_xlim([25,300])
fig.show()

fig,ax = plt.subplots(1,1,figsize=(8,4))
h[:, :, "top1"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("top1 Jet $p_T$")
ax.set_xlim([25,200])
fig.show()
```

```python
#### mass histogram ####

# binning
mass_low = 0.0
mass_high = 50.0
mass_numbins = 100
legend_list = ["All Matches Correct", "Some Matches Correct", "No Matches Correct"]

# define histogram
h = hist.Hist(
    hist.axis.Regular(mass_numbins, mass_low, mass_high, 
                      name="jetmass", label="Jet Mass [GeV]", flow=False),
    hist.axis.StrCategory(legend_list, name="truthlabel", label="Truth Label"),
    hist.axis.StrCategory(["W","top1","top2"], name="category", label="Category"),
)

# fill histogram
h.fill(jetmass = all_correct[:,15], category="W", truthlabel="All Matches Correct")
h.fill(jetmass = some_correct[:,15], category="W", truthlabel="Some Matches Correct")
h.fill(jetmass = none_correct[:,15], category="W", truthlabel="No Matches Correct")
h.fill(jetmass = all_correct[:,16], category="W", truthlabel="All Matches Correct")
h.fill(jetmass = some_correct[:,16], category="W", truthlabel="Some Matches Correct")
h.fill(jetmass = none_correct[:,16], category="W", truthlabel="No Matches Correct")
h.fill(jetmass = all_correct[:,17], category="top2", truthlabel="All Matches Correct")
h.fill(jetmass = some_correct[:,17], category="top2", truthlabel="Some Matches Correct")
h.fill(jetmass = none_correct[:,17], category="top2", truthlabel="No Matches Correct")
h.fill(jetmass = all_correct[:,18], category="top1", truthlabel="All Matches Correct")
h.fill(jetmass = some_correct[:,18], category="top1", truthlabel="Some Matches Correct")
h.fill(jetmass = none_correct[:,18], category="top1", truthlabel="No Matches Correct")

# make plots
fig,ax = plt.subplots(1,1,figsize=(8,4))
h[:, :, "W"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("W Jet Mass")
# fig.show()

fig,ax = plt.subplots(1,1,figsize=(8,4))
h[:, :, "top2"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("top2 Jet Mass")
fig.show()

fig,ax = plt.subplots(1,1,figsize=(8,4))
h[:, :, "top1"].plot(density=True, ax=ax)
ax.legend(legend_list)
ax.set_title("top1 Jet Mass")
fig.show()
```

# Model Optimization

The model used here is `xgboost`'s gradient-boosted decision tree (`XGBClassifier`). Hyperparameter optimization is performed using random selection from a sample space of hyperparameters then testing model fits in a parallelized manner using `dask`. Optional `mlflow` logging is included.

```python
# grab features and labels and convert to np array
features = np.array(output['features']['ttbar__nominal'])
labels = np.array(output['labels']['ttbar__nominal'])
even = np.array(output['even']['ttbar__nominal'])

labels = labels.reshape((len(labels),))
even = np.repeat(even, 12)

labels[labels==-1]=0 # consider all combination (partially correct is same as 0% correct for training)

features_even = features[even]
labels_even = labels[even]

features_odd = features[np.invert(even)]
labels_odd = labels[np.invert(even)]
```

```python
### separate data into train/val/testr ###

RANDOM_SEED = 5
TRAIN_RATIO = 0.9

# separate even into train/val
features_even_unflattened = features_even.reshape((int(features_even.shape[0]/12),12,19))
labels_even_unflattened = labels_even.reshape((int(features_even.shape[0]/12),12))

features_train_even, features_val_even, labels_train_even, labels_val_even = train_test_split(features_even_unflattened, 
                                                                                              labels_even_unflattened, 
                                                                                              train_size=TRAIN_RATIO, 
                                                                                              random_state=RANDOM_SEED)

print("features_train_even.shape = ", features_train_even.shape)
print("labels_train_even.shape = ", labels_train_even.shape)
print("features_val_even.shape = ", features_val_even.shape)
print("labels_val_even.shape = ", labels_val_even.shape)

which_combination_train_even = np.where(labels_train_even==1)[1]
features_train_even = features_train_even.reshape((12*features_train_even.shape[0],19))
labels_train_even = labels_train_even.reshape((12*labels_train_even.shape[0],))

which_combination_val_even = np.where(labels_val_even==1)[1]
features_val_even = features_val_even.reshape((12*features_val_even.shape[0],19))
labels_val_even = labels_val_even.reshape((12*labels_val_even.shape[0],))


# separate odd into train/val
features_odd_unflattened = features_odd.reshape((int(features_odd.shape[0]/12),12,19))
labels_odd_unflattened = labels_odd.reshape((int(features_odd.shape[0]/12),12))

features_train_odd, features_val_odd, labels_train_odd, labels_val_odd = train_test_split(features_odd_unflattened, 
                                                                                          labels_odd_unflattened, 
                                                                                          train_size=TRAIN_RATIO, 
                                                                                          random_state=RANDOM_SEED)

print("features_train_odd.shape = ", features_train_odd.shape)
print("labels_train_odd.shape = ", labels_train_odd.shape)
print("features_val_odd.shape = ", features_val_odd.shape)
print("labels_val_odd.shape = ", labels_val_odd.shape)

which_combination_train_odd = np.where(labels_train_odd==1)[1]
features_train_odd = features_train_odd.reshape((12*features_train_odd.shape[0],19))
labels_train_odd = labels_train_odd.reshape((12*labels_train_odd.shape[0],))

which_combination_val_odd = np.where(labels_val_odd==1)[1]
features_val_odd = features_val_odd.reshape((12*features_val_odd.shape[0],19))
labels_val_odd = labels_val_odd.reshape((12*labels_val_odd.shape[0],))
```

```python
# preprocess features so that they are more Gaussian-like
power = PowerTransformer(method='yeo-johnson', standardize=True)

features_train_even = power.fit_transform(features_train_even)
features_val_even = power.transform(features_val_even)
features_train_odd = power.transform(features_train_odd)
features_val_odd = power.transform(features_val_odd)
```

```python
sampler = ParameterSampler({'max_depth': np.arange(2,30,2,dtype=int), 
                            'n_estimators': np.arange(50,700,20,dtype=int), 
                            'learning_rate': np.logspace(-5, -1, 10),
                            'min_child_weight': np.logspace(-1, 2, 20), 
                            'reg_lambda': [0, 0.25, 0.5, 0.75, 1], 
                            'reg_alpha': [0, 0.25, 0.5, 0.75, 1],
                            'gamma': np.logspace(-4, 1, 20),}, 
                            n_iter = 50, 
                            random_state=34) 

samples = list(sampler)
```

```python
for i in range(len(samples)):
    samples[i]['trial_num'] = i
samples[0]
```

```python
def fit_model(params, 
              features_train, 
              labels_train, 
              which_combination_train,
              features_val, 
              labels_val, 
              which_combination_val,
              evaluation_matrix,
              USE_MLFLOW=False): 
    
    trial_num = params["trial_num"]
    
    if USE_MLFLOW:
        mlflowclient = MlflowClient()
        run = mlflowclient.create_run(experiment_id="9", run_name=f"run-{trial_num}") #run_name=f"run-{trial_num}", nested=True): 
        
        for param, value in params.items(): 
            mlflowclient.log_param(run.info.run_id, param, value) 

    params_copy = params.copy()
    params_copy.pop("trial_num") # remove trial_num as it is not a parameter for the BDT
            
    # initialize model with current parameters
    model = XGBClassifier(random_state=5, booster='gbtree', **params) 
        
    # fit model to training sample
    model.fit(features_train, labels_train)
    
    # predictions using trained model
    predict_train = model.predict(features_train)
    predict_proba_train = model.predict_proba(features_train)[:, 1]
    
    # calculated jet accuracy for training sample
    predict_proba_train_evt = predict_proba_train.reshape((int(len(predict_proba_train)/12),12))
    predicted_combination_train = np.argmax(predict_proba_train_evt,axis=1)

    scores = np.zeros(len(which_combination_train))
    zipped = list(zip(which_combination_train.tolist(), predicted_combination_train.tolist()))
    for i in range(len(which_combination_train)):
        scores[i] = evaluation_matrix[zipped[i]]
    jet_accuracy_train = -sum(scores)/len(scores)
        
    # log training metrics
    if USE_MLFLOW:
        mlflowclient.log_metric(run.info.run_id, 'train_accuracy', 
                                accuracy_score(labels_train, predict_train))
        mlflowclient.log_metric(run.info.run_id, 'train_precision', 
                                precision_score(labels_train, predict_train, zero_division=0))
        mlflowclient.log_metric(run.info.run_id, 'train_recall', 
                                recall_score(labels_train, predict_train))
        mlflowclient.log_metric(run.info.run_id, 'train_f1', 
                                f1_score(labels_train, predict_train))
        mlflowclient.log_metric(run.info.run_id, 'train_roc_auc', 
                                roc_auc_score(labels_train, predict_proba_train))
        mlflowclient.log_metric(run.info.run_id, 'train_jet_accuracy', jet_accuracy_train)
        
    # predictions using trained model
    predict_val= model.predict(features_val)
    predict_proba_val = model.predict_proba(features_val)[:, 1]
    
    # calculated jet accuracy for validation sample
    predict_proba_val_evt = predict_proba_val.reshape((int(len(predict_proba_val)/12),12))
    predicted_combination_val = np.argmax(predict_proba_val_evt,axis=1)

    scores = np.zeros(len(which_combination_val))
    zipped = list(zip(which_combination_val.tolist(), predicted_combination_val.tolist()))
    for i in range(len(which_combination_val)):
        scores[i] = evaluation_matrix[zipped[i]]
    jet_accuracy_val = -sum(scores)/len(scores)
    
    if USE_MLFLOW:
        mlflowclient.log_metric(run.info.run_id, 'val_accuracy', 
                                accuracy_score(labels_val, predict_val))
        mlflowclient.log_metric(run.info.run_id, 'val_precision', 
                                precision_score(labels_val, predict_val, zero_division=0))
        mlflowclient.log_metric(run.info.run_id, 'val_recall', 
                                recall_score(labels_val, predict_val))
        mlflowclient.log_metric(run.info.run_id, 'val_f1', 
                                f1_score(labels_val, predict_val))
        mlflowclient.log_metric(run.info.run_id, 'val_roc_auc', 
                                roc_auc_score(labels_val, predict_proba_val))
        mlflowclient.log_metric(run.info.run_id, 'val_jet_accuracy', jet_accuracy_val)
    
        # logging model
        signature = infer_signature(features_train, predict_train)
        # mlflow.xgboost.log_model(model, f'sigbkg_bdt_{trial_num}', signature=signature)

        # explicitly close client
        mlflowclient.set_terminated(run.info.run_id)
        
    return jet_accuracy_val
```

```python
# to transfer env. variables to workers
def initialize_mlflow(): 

    os.environ['MLFLOW_TRACKING_URI'] = "https://mlflow.software-dev.ncsa.cloud"
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "https://mlflow-minio-api.software-dev.ncsa.cloud"
    os.environ['AWS_ACCESS_KEY_ID'] = ""
    os.environ['AWS_SECRET_ACCESS_KEY'] = ""
    
    mlflow.set_tracking_uri('https://mlflow.software-dev.ncsa.cloud') 
    mlflow.set_experiment("agc-demo-example")
```

```python

```

```python

```

```python

```

```python

```

```python
permutation_labels = np.array(labels_dict[4])
```

```python
evaluation_matrix = np.zeros((12,12))
for i in range(len(permutation_labels)):
    for j in range(len(permutation_labels)):
        evaluation_matrix[i,j]=sum(np.equal(permutation_labels[i,:],permutation_labels[j,:]))/4
print(evaluation_matrix)
```

```python
mlflow.set_tracking_uri("https://mlflow.software-dev.ncsa.cloud")
EXPERIMENT_ID = mlflow.set_experiment('optimize-reconstruction-bdt-00')
```

```python
%env MLFLOW_TRACKING_URI=https://mlflow.software-dev.ncsa.cloud
%env MLFLOW_S3_ENDPOINT_URL=https://mlflow-minio-api.software-dev.ncsa.cloud
%env AWS_ACCESS_KEY_ID=
%env AWS_SECRET_ACCESS_KEY=leftfoot1
```

```python
current_experiment=dict(mlflow.get_experiment_by_name('optimize-reconstruction-bdt-00'))
EXP_ID=current_experiment['experiment_id']
```

```python
# training method for hyperopt
def train_and_evaluate(params):
    
    # mlflow.xgboost.autolog()
    
    with mlflow.start_run(experiment_id=EXP_ID, nested=True):
    
        model = xgb.XGBClassifier(**params) # define model with current parameters
        model = model.fit(features_train, labels_train) # train model
        
        mlflow.log_params(params)

        # predicting train set and validation set
        train_predicted = model.predict(features_train)
        train_predicted_prob = model.predict_proba(features_train)[:, 1]
        val_predicted = model.predict(features_val)
        val_predicted_prob = model.predict_proba(features_val)[:, 1]

        # model metrics to track
        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'aucroc']

        # Training evaluation metrics
        train_metrics = {
            'Accuracy': accuracy_score(labels_train, train_predicted), 
            'Precision': precision_score(labels_train, train_predicted, zero_division=0), 
            'Recall': recall_score(labels_train, train_predicted), 
            'F1': f1_score(labels_train, train_predicted), 
            'AUCROC': roc_auc_score(labels_train, train_predicted_prob),
        }

        # Validation evaluation metrics
        val_metrics = {
            'Accuracy': accuracy_score(labels_val, val_predicted), 
            'Precision': precision_score(labels_val, val_predicted, zero_division=0), 
            'Recall': recall_score(labels_val, val_predicted), 
            'F1': f1_score(labels_val, val_predicted), 
            'AUCROC': roc_auc_score(labels_val, val_predicted_prob),
        }

        train_predicted_prob = train_predicted_prob.reshape((int(len(train_predicted_prob)/12),12))
        train_predicted_combination = np.argmax(train_predicted_prob,axis=1)

        scores = np.zeros(len(which_combination_train))
        zipped = list(zip(which_combination_train.tolist(), train_predicted_combination.tolist()))
        for i in range(len(which_combination_train)):
            scores[i] = evaluation_matrix[zipped[i]]
        score_train = -sum(scores)/len(scores)
        
        val_predicted_prob = val_predicted_prob.reshape((int(len(val_predicted_prob)/12),12))
        val_predicted_combination = np.argmax(val_predicted_prob,axis=1)

        scores = np.zeros(len(which_combination_val))
        zipped = list(zip(which_combination_val.tolist(), val_predicted_combination.tolist()))
        for i in range(len(which_combination_val)):
            scores[i] = evaluation_matrix[zipped[i]]
        score_val = -sum(scores)/len(scores)
        
        train_metrics["Jet-Accuracy"] = -score_train
        train_metrics_values = list(train_metrics.values())
        
        val_metrics["Jet-Accuracy"] = -score_val
        val_metrics_values = list(val_metrics.values())
        
        # Logging model signature, class, and name
        signature = infer_signature(features_train, val_predicted)
        mlflow.xgboost.log_model(model, 'model', signature=signature)
        mlflow.set_tag('estimator_name', model.__class__.__name__)
        mlflow.set_tag('estimator_class', model.__class__)

        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'aucroc', "jet-accuracy"]
        # Logging each metric
        for name, metric in list(zip(metric_names, train_metrics_values)):
            mlflow.log_metric(f'training_{name}', metric)
        for name, metric in list(zip(metric_names, val_metrics_values)):
            mlflow.log_metric(f'validation_{name}', metric)

    return {'status': STATUS_OK, 'loss': score_val}
```

```python tags=[]
trials = Trials()

# optimize model
with mlflow.start_run(experiment_id=EXP_ID, run_name='xgboost_bdt_models'):
    best_parameters = fmin(
        fn=train_and_evaluate, 
        space=trial_params,
        algo=tpe.suggest,
        trials=trials,
        max_evals=20 # how many trials to run
                          )
```

```python
# convert max_depth and n_estimators to integers
best_parameters['max_depth'] = int(best_parameters['max_depth'])
best_parameters['n_estimators'] = int(best_parameters['n_estimators'])
```

```python
print("Best Model Parameters = ", best_parameters)
```

# Training/Evaluation with Optimized Model

```python
# define and fit model to data
model = xgb.XGBClassifier(**best_parameters)
model = model.fit(features_train, labels_train)
```

```python
# make predictions
train_predicted = model.predict(features_train)
train_predicted_prob = model.predict_proba(features_train)[:, 1]
val_predicted = model.predict(features_val)
val_predicted_prob = model.predict_proba(features_val)[:, 1]
```

```python
train_accuracy = accuracy_score(labels_train, train_predicted).round(3)
train_precision = precision_score(labels_train, train_predicted).round(3)
train_recall = recall_score(labels_train, train_predicted).round(3)
train_f1 = f1_score(labels_train, train_predicted).round(3)
train_aucroc = roc_auc_score(labels_train, train_predicted_prob).round(3)
print("Training Accuracy = ", train_accuracy)
print("Training Precision = ", train_precision)
print("Training Recall = ", train_recall)
print("Training f1 = ", train_f1)
print("Training AUC = ", train_aucroc)
print()

val_accuracy = accuracy_score(labels_val, val_predicted).round(3)
val_precision = precision_score(labels_val, val_predicted).round(3)
val_recall = recall_score(labels_val, val_predicted).round(3)
val_f1 = f1_score(labels_val, val_predicted).round(3)
val_aucroc = roc_auc_score(labels_val, val_predicted_prob).round(3)
print("Validation Accuracy = ", val_accuracy)
print("Validation Precision = ", val_precision)
print("Validation Recall = ", val_recall)
print("Validation f1 = ", val_f1)
print("Validation AUC = ", val_aucroc)
```

```python
val_predicted_prob = val_predicted_prob.reshape((int(len(val_predicted_prob)/12),12))
val_predicted_combination = np.argmax(val_predicted_prob,axis=1)
    
scores = np.zeros(len(which_combination_val))
zipped = list(zip(which_combination_val.tolist(), val_predicted_combination.tolist()))
for i in range(len(which_combination_val)):
    scores[i] = evaluation_matrix[zipped[i]]
        
score = -sum(scores)/len(scores)
print("Validation Jet Score = ", score)

train_predicted_prob = train_predicted_prob.reshape((int(len(train_predicted_prob)/12),12))
train_predicted_combination = np.argmax(train_predicted_prob,axis=1)
    
scores = np.zeros(len(which_combination_train))
zipped = list(zip(which_combination_train.tolist(), train_predicted_combination.tolist()))
for i in range(len(which_combination_train)):
    scores[i] = evaluation_matrix[zipped[i]]
        
score = -sum(scores)/len(scores)
print("Training Jet Score = ", score)
```

```python
# make predictions
test_predicted = model.predict(features_test)
test_predicted_prob = model.predict_proba(features_test)[:, 1]
```

```python
test_accuracy = accuracy_score(labels_test, test_predicted).round(3)
test_precision = precision_score(labels_test, test_predicted).round(3)
test_recall = recall_score(labels_test, test_predicted).round(3)
test_f1 = f1_score(labels_test, test_predicted).round(3)
test_aucroc = roc_auc_score(labels_test, test_predicted_prob).round(3)
print("Test Accuracy = ", test_accuracy)
print("Test Precision = ", test_precision)
print("Test Recall = ", test_recall)
print("Test f1 = ", test_f1)
print("Test AUC = ", test_aucroc)
```

```python
test_predicted_prob = test_predicted_prob.reshape((int(len(test_predicted_prob)/12),12))
test_predicted_combination = np.argmax(test_predicted_prob,axis=1)
    
scores = np.zeros(len(which_combination_test))
zipped = list(zip(which_combination_test.tolist(), test_predicted_combination.tolist()))
for i in range(len(which_combination_test)):
    scores[i] = evaluation_matrix[zipped[i]]
        
score = sum(scores)/len(scores)
print("Test Jet Score = ", score)
print("Random Assignment Jet Score = ", 0.375)
```

```python
print("How many events are 100% correct: ", sum(scores==1)/len(scores), ", Random = ",sum(evaluation_matrix[0,:]==1)/12)
print("How many events are 50% correct: ", sum(scores==0.5)/len(scores), ", Random = ",sum(evaluation_matrix[0,:]==0.5)/12)
print("How many events are 25% correct: ", sum(scores==0.25)/len(scores), ", Random = ",sum(evaluation_matrix[0,:]==0.25)/12)
print("How many events are 0% correct: ", sum(scores==0)/len(scores), ", Random = ",sum(evaluation_matrix[0,:]==0)/12)
```

```python
# save model to json. this file can be used with the FIL backend in nvidia-triton!
model.save_model("models/model_xgb_230206.json")
```
