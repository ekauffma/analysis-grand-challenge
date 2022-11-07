---
jupyter:
  jupytext:
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

```python
import asyncio
import time
import logging

import vector; vector.register_awkward()

import awkward as ak
import cabinetry
from coffea import processor
from coffea.processor import servicex
from coffea.nanoevents import transforms
from coffea.nanoevents.methods import base, vector
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms
from func_adl import ObjectStream
import hist
import json
import matplotlib.pyplot as plt
import numpy as np
import uproot

import utils  # contains code for bookkeeping and cosmetics, as well as some boilerplate

logging.getLogger("cabinetry").setLevel(logging.INFO)
```

```python
### GLOBAL CONFIGURATION

# input files per process, set to e.g. 10 (smaller number = faster)
N_FILES_MAX_PER_SAMPLE = 10

# pipeline to use:
# - "coffea" for pure coffea setup
# - "servicex_processor" for coffea with ServiceX processor
# - "servicex_databinder" for downloading query output and subsequent standalone coffea
PIPELINE = "coffea"

# enable Dask (may not work yet in combination with ServiceX outside of coffea-casa)
USE_DASK = True

# ServiceX behavior: ignore cache with repeated queries
SERVICEX_IGNORE_CACHE = True

# analysis facility: set to "coffea_casa" for coffea-casa environments, "EAF" for FNAL, "local" for local setups
AF = "coffea_casa"
```

```python
fileset = utils.construct_fileset(N_FILES_MAX_PER_SAMPLE, use_xcache=False)
```

```python
fileset.keys()
```

```python
fileset['ttbar__nominal'].keys()
```

```python
fileset['ttbar__nominal']['files']
```

```python
fileset['wjets__nominal']['files']
```

```python
file_sig = uproot.open(fileset['ttbar__nominal']['files'][0])
events_tree_sig = file_sig['events']
events_sig = events_tree_sig.arrays(["electron_pt",
                                    "muon_pt",
                                    "jet_pt",
                                    "jet_eta",
                                    "jet_phi",
                                    "jet_mass",
                                    "jet_btag"],entry_stop=1000)

file_bkg = uproot.open(fileset['wjets__nominal']['files'][0])
events_tree_bkg = file_bkg['events']
events_bkg = events_tree_bkg.arrays(["electron_pt",
                                     "muon_pt",
                                     "jet_pt",
                                     "jet_eta",
                                     "jet_phi",
                                     "jet_mass",
                                     "jet_btag"])
```

```python

```

```python
len(events_bkg)
```

```python
selected_electrons_pt_sig = events_sig.electron_pt[events_sig.electron_pt > 25]
selected_muons_pt_sig = events_sig.muon_pt[events_sig.muon_pt > 25]
selected_jet_pt_sig = events_sig.jet_pt[events_sig.jet_pt > 25]
selected_jet_btag_sig = events_sig.jet_btag[events_sig.jet_pt > 25]

selected_electrons_pt_bkg = events_bkg.electron_pt[events_bkg.electron_pt > 25]
selected_muons_pt_bkg = events_bkg.muon_pt[events_bkg.muon_pt > 25]
selected_jet_pt_bkg = events_bkg.jet_pt[events_bkg.jet_pt > 25]
selected_jet_btag_bkg = events_bkg.jet_btag[events_bkg.jet_pt > 25]
```

```python
B_TAG_THRESHOLD = 0.5

event_filters = ((ak.count(selected_electrons_pt_sig, axis=1) + ak.count(selected_muons_pt_sig, axis=1)) == 1)
event_filters = event_filters & (ak.count(selected_jet_pt_sig, axis=1) >= 4)
event_filters = event_filters & (ak.sum(selected_jet_btag_sig >= B_TAG_THRESHOLD, axis=1) >= 1) # one b-tagged jet

selected_events_sig = events_sig[event_filters]
selected_electrons_pt_sig = selected_electrons_pt_sig[event_filters]
selected_muons_pt_sig = selected_muons_pt_sig[event_filters]
selected_jet_pt_sig = selected_jet_pt_sig[event_filters]
selected_jet_btag_sig = selected_jet_btag_sig[event_filters]


event_filters = ((ak.count(selected_electrons_pt_bkg, axis=1) + ak.count(selected_muons_pt_bkg, axis=1)) == 1)
event_filters = event_filters & (ak.count(selected_jet_pt_bkg, axis=1) >= 4)
event_filters = event_filters & (ak.sum(selected_jet_btag_bkg >= B_TAG_THRESHOLD, axis=1) >= 1)

selected_events_bkg = events_bkg[event_filters]
selected_electrons_pt_bkg = selected_electrons_pt_bkg[event_filters]
selected_muons_pt_bkg = selected_muons_pt_bkg[event_filters]
selected_jet_pt_bkg = selected_jet_pt_bkg[event_filters]
selected_jet_btag_bkg = selected_jet_btag_bkg[event_filters]
```

```python
selected_lepton_pt_sig = ak.sum(selected_electrons_pt_sig,axis=-1) + ak.sum(selected_muons_pt_sig,axis=-1)
selected_lepton_pt_bkg = ak.sum(selected_electrons_pt_bkg,axis=-1) + ak.sum(selected_muons_pt_bkg,axis=-1)
```

```python
selected_jet_ht_sig = ak.sum(selected_jet_pt_sig, axis=-1)
selected_jet_ht_bkg = ak.sum(selected_jet_pt_bkg, axis=-1)
```

```python
selected_jet_pt0_sig = ak.max(selected_jet_pt_sig, axis=-1)
selected_jet_pt0_bkg = ak.max(selected_jet_pt_bkg, axis=-1)
```

```python
sigbkghist = hist.Hist(
    hist.axis.Regular(50, 0, 2000, name="HT", label="$H_T$"),
    hist.axis.Regular(50, 0, 300, name="leptonpt", label="$p_T^{\mu,e}$"),
    hist.axis.Regular(50, 0, 300, name="ljpt", label="Leading Jet $p_T$"),
    hist.axis.StrCategory(["sig", "bkg"], name="category", label="Category"),
)
```

```python
sigbkghist.fill(HT = selected_jet_ht_sig, 
                leptonpt = selected_lepton_pt_sig, 
                ljpt = selected_jet_pt0_sig, 
                category = "sig")
sigbkghist.fill(HT = selected_jet_ht_bkg, 
                leptonpt = selected_lepton_pt_bkg, 
                ljpt = selected_jet_pt0_bkg, 
                category = "bkg")
```

```python
print("Num Signal = ", len(selected_jet_ht_sig))
print("Num Background = ", len(selected_jet_ht_bkg))
```

```python
sigbkghist.axes
```

```python
s = sigbkghist.stack("category")
```

```python
s.project("HT").plot(density=True)
```

```python
s.project("leptonpt").plot(density=True)
```

```python
s.project("ljpt").plot(density=True)
```

```python
selected_jet_ht = ak.concatenate([selected_jet_ht_sig,selected_jet_ht_bkg],axis = -1)
selected_lepton_pt = ak.concatenate([selected_lepton_pt_sig,selected_lepton_pt_bkg],axis = -1)
selected_lj_pt = ak.concatenate([selected_jet_pt0_sig,selected_jet_pt0_bkg],axis = -1)
event_cat = ak.concatenate([ak.ones_like(selected_jet_ht_sig), ak.zeros_like(selected_jet_ht_bkg)], axis = -1)
```

```python
from sklearn.neighbors import KNeighborsClassifier

features=ak.zip((selected_jet_ht, selected_lepton_pt, selected_lj_pt))

model = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training sets
model.fit(features.to_list(),event_cat.to_list())
```

```python
predicted = model.predict(features.to_list())
```

```python
predicted = predicted.astype(bool)
labels = np.array(event_cat.to_list(), dtype=bool)
sum(predicted*labels)/len(predicted)
```

```python
print(sum(labels)/len(labels))
print(sum(predicted)/len(predicted))
```

```python
var = 'hello'
if var:
    print(var)
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

```

```python
class AGCSchema(BaseSchema):
    # credit to Mat Adamec for implementing this schema
    def __init__(self, base_form):
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        names = set([k.split('_')[0] for k in branch_forms.keys() if not (k.startswith('number'))])
        # Remove n(names) from consideration. It's safe to just remove names that start with n,
        # as nothing else begins with n in our fields.
        # Also remove GenPart, PV and MET because they deviate from the pattern of having a 'number' field.
        names = [k for k in names if not (
                    k.startswith("n") | k.startswith("met") | k.startswith("GenPart") | k.startswith("PV")
        )]
        output = {}
        for name in names:
            offsets = transforms.counts2offsets_form(branch_forms['number' + name])
            content = {
                k[len(name) + 1 :]: branch_forms[k]
                for k in branch_forms
                if (k.startswith(name + "_") & (k[len(name) + 1 :] != "e"))
            }
            # Add energy separately so its treated correctly by the p4 vector.
            content['energy'] = branch_forms[name+'_e']
            # Check for LorentzVector
            output[name] = zip_forms(content, name, 'PtEtaPhiELorentzVector', offsets=offsets)

        return output
    
    @property
    def behavior(self):
        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        return behavior
```

```python
from coffea.nanoevents import NanoEventsFactory
```

```python
events_sig = NanoEventsFactory.from_root(fileset['ttbar__nominal']['files'][0], 
                                         schemaclass=AGCSchema, 
                                         treepath="events").events()
events_bkg = NanoEventsFactory.from_root(fileset['wjets__nominal']['files'][0], 
                                         schemaclass=AGCSchema, 
                                         treepath="events").events()

```

```python
len(events_bkg)
```

```python
selected_electrons_sig = events_sig.electron[events_sig.electron.pt > 25]
selected_muons_sig = events_sig.muon[events_sig.muon.pt > 25]
selected_jets_sig = events_sig.jet[events_sig.jet.pt > 25]

event_filters = ((ak.count(selected_electrons_sig.pt, axis=1) + ak.count(selected_muons_sig.pt, axis=1)) == 1)
event_filters = event_filters & (ak.count(selected_jets_sig.pt, axis=1) >= 4)
B_TAG_THRESHOLD = 0.5
event_filters = event_filters & (ak.sum(selected_jets_sig.btag > B_TAG_THRESHOLD, axis=1) >= 2)

selected_jets_sig = selected_jets_sig[event_filters]

selected_electrons_bkg = events_bkg.electron[events_bkg.electron.pt > 25]
selected_muons_bkg = events_bkg.muon[events_bkg.muon.pt > 25]
selected_jets_bkg = events_bkg.jet[events_bkg.jet.pt > 25]

event_filters = ((ak.count(selected_electrons_bkg.pt, axis=1) + ak.count(selected_muons_bkg.pt, axis=1)) == 1)
event_filters = event_filters & (ak.count(selected_jets_bkg.pt, axis=1) >= 4)
B_TAG_THRESHOLD = 0.5
event_filters = event_filters & (ak.sum(selected_jets_bkg.btag > B_TAG_THRESHOLD, axis=1) >= 2)

selected_jets_bkg = selected_jets_bkg[event_filters]
```

```python
print(len(selected_jets_sig))
print(len(selected_jets_bkg))
```

```python tags=[]
trijet_sig = ak.combinations(selected_jets_sig, 3, fields=["j1", "j2", "j3"])  # trijet candidate
trijet_sig["p4"] = trijet_sig.j1 + trijet_sig.j2 + trijet_sig.j3  # calculate four-momentum of tri-jet system
deltar_0 = np.sqrt((trijet_sig.j1.eta - trijet_sig.j2.eta)**2 + 
                   (trijet_sig.j1.phi - trijet_sig.j2.phi)**2)
deltar_1 = np.sqrt((trijet_sig.j3.eta - trijet_sig.j2.eta)**2 + 
                   (trijet_sig.j3.phi - trijet_sig.j2.phi)**2)
deltar_2 = np.sqrt((trijet_sig.j3.eta - trijet_sig.j1.eta)**2 + 
                   (trijet_sig.j3.phi - trijet_sig.j1.phi)**2)


trijet_sig["avg_deltar"] = ak.mean(ak.concatenate([deltar_0[..., np.newaxis], 
                                                   deltar_1[..., np.newaxis], 
                                                   deltar_2[..., np.newaxis]], 
                                                  axis=2), axis=-1)


trijet_sig["max_btag"] = np.maximum(trijet_sig.j1.btag, np.maximum(trijet_sig.j2.btag, trijet_sig.j3.btag))
trijet_sig = trijet_sig[trijet_sig.max_btag > B_TAG_THRESHOLD]  # require at least one-btag in trijet candidates
trijet_mass_sig = ak.flatten(trijet_sig["p4"][ak.argmax(trijet_sig.p4.pt, axis=1, keepdims=True)].mass)
trijet_deltar_sig = ak.flatten(trijet_sig["avg_deltar"][ak.argmax(trijet_sig.p4.pt, axis=1, keepdims=True)])

trijet_bkg = ak.combinations(selected_jets_bkg, 3, fields=["j1", "j2", "j3"])  # trijet candidate
trijet_bkg["p4"] = trijet_bkg.j1 + trijet_bkg.j2 + trijet_bkg.j3  # calculate four-momentum of tri-jet system
deltar_0 = np.sqrt((trijet_bkg.j1.eta - trijet_bkg.j2.eta)**2 + 
                   (trijet_bkg.j1.phi - trijet_bkg.j2.phi)**2)
deltar_1 = np.sqrt((trijet_bkg.j3.eta - trijet_bkg.j2.eta)**2 + 
                   (trijet_bkg.j3.phi - trijet_bkg.j2.phi)**2)
deltar_2 = np.sqrt((trijet_bkg.j3.eta - trijet_bkg.j1.eta)**2 + 
                   (trijet_bkg.j3.phi - trijet_bkg.j1.phi)**2)
trijet_bkg["avg_deltar"] = ak.mean(ak.concatenate([deltar_0[..., np.newaxis], 
                                                   deltar_1[..., np.newaxis], 
                                                   deltar_2[..., np.newaxis]], 
                                                  axis=2), axis=-1)


trijet_bkg["max_btag"] = np.maximum(trijet_bkg.j1.btag, np.maximum(trijet_bkg.j2.btag, trijet_bkg.j3.btag))
trijet_bkg = trijet_bkg[trijet_bkg.max_btag > B_TAG_THRESHOLD]  # require at least one-btag in trijet candidates
trijet_mass_bkg = ak.flatten(trijet_bkg["p4"][ak.argmax(trijet_bkg.p4.pt, axis=1, keepdims=True)].mass)
trijet_deltar_bkg = ak.flatten(trijet_bkg["avg_deltar"][ak.argmax(trijet_bkg.p4.pt, axis=1, keepdims=True)])
```

```python
print(len(trijet_deltar_bkg))
print(len(trijet_mass_bkg))
```

```python
plt.hist(trijet_deltar_bkg,bins=50,histtype='step',density=True)
plt.hist(trijet_deltar_sig,bins=50,histtype='step',density=True)
plt.show()
```

```python
predicted = model.predict(features_test).astype(bool)
labels_test = np.array(labels_test, dtype=bool)

print("Efficiency = ", 100*sum(predicted*labels_test)/len(predicted), "%")
```

```python
events = NanoEventsFactory.from_root(local_file_name, schemaclass=AGCSchema, treepath="events").events()
```

```python

```
