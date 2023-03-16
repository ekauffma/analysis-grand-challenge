# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import uproot
from coffea.nanoevents import NanoEventsFactory
import numpy as np
import matplotlib.pyplot as plt
import torch
import awkward as ak
import dataclasses


# %%
def filterEvents(jets, electrons, muons, genpart):

    
    selected_electrons = electrons[electrons.pt > 25]
    selected_muons = muons[muons.pt > 25]
    jet_filter = (jets.pt > 25)
    selected_jets = jets[jet_filter]

    # single lepton requirement
    event_filters = ((ak.count(selected_electrons.pt, axis=1) + ak.count(selected_muons.pt, axis=1)) == 1)
    # at least four jets
    event_filters = event_filters & (ak.count(selected_jets.pt, axis=1) >= 4)
    # at least one b-tagged jet ("tag" means score above threshold)
    B_TAG_THRESHOLD = 0.5
    event_filters = event_filters & (ak.sum(selected_jets.btagCSVV2 >= B_TAG_THRESHOLD, axis=1) >= 1)

    selected_events = events[event_filters]
    selected_electrons = selected_electrons[event_filters]
    selected_muons = selected_muons[event_filters]
    selected_jets = selected_jets[event_filters]
    selected_genpart = genpart[event_filters]

    ### only consider 4j2b region
    region_filter = ak.sum(selected_jets.btagCSVV2 > B_TAG_THRESHOLD, axis=1) >= 2

    selected_jets_region = selected_jets[region_filter]
    selected_electrons_region = selected_electrons[region_filter]
    selected_muons_region = selected_muons[region_filter]
    selected_genpart_region = selected_genpart[region_filter]
    
    
    #### filter genPart to valid matching candidates ####

    # get rid of particles without parents
    genpart_parent = selected_genpart_region.distinctParent
    genpart_filter = np.invert(ak.is_none(genpart_parent, axis=1))
    selected_genpart_region_reduced = selected_genpart_region[genpart_filter]
    genpart_parent_reduced = selected_genpart_region_reduced.distinctParent

    # ensure that parents are top quark or W
    genpart_filter2 = ((np.abs(genpart_parent_reduced.pdgId)==6) | 
                       (np.abs(genpart_parent_reduced.pdgId)==24))
    selected_genpart_region_reduced = selected_genpart_region_reduced[genpart_filter2]

    # ensure particle itself is a quark
    genpart_filter3 = ((np.abs(selected_genpart_region_reduced.pdgId)<7) & 
                       (np.abs(selected_genpart_region_reduced.pdgId)>0))
    selected_genpart_region_reduced = selected_genpart_region_reduced[genpart_filter3]

    # get rid of duplicates
    genpart_filter4 = selected_genpart_region_reduced.hasFlags("isLastCopy")
    selected_genpart_region_reduced = selected_genpart_region_reduced[genpart_filter4]
    
    # match jets to nearest valid genPart candidate
    nearest_genpart = selected_jets_region.nearest(selected_genpart_region_reduced, 
                                                   threshold=1.0)
    nearest_parent = nearest_genpart.distinctParent # parent of matched particle

    parent_pdgid = nearest_parent.pdgId # pdgId of parent particle
    grandchild_pdgid = nearest_parent.distinctChildren.distinctChildren.pdgId # pdgId of particle's parent's grandchildren

    jet_counts = ak.num(selected_jets_region)
    grandchildren_flat = np.abs(ak.flatten(grandchild_pdgid,axis=-1)) # flatten innermost axis for convenience
    
    # if particle has a cousin that is a lepton
    has_lepton_cousin = (ak.sum(((grandchildren_flat%2==0) & (grandchildren_flat>10) & (grandchildren_flat<19)),
                                axis=-1)>0)
    # if particle has a cousin that is a neutrino
    has_neutrino_cousin = (ak.sum(((grandchildren_flat%2==1) & (grandchildren_flat>10) & (grandchildren_flat<19)),
                                  axis=-1)>0)
    # if a particle has a lepton cousin and a neutrino cousin
    has_both_cousins = ak.fill_none((has_lepton_cousin & has_neutrino_cousin), False)
    
    has_both_cousins_flat = ak.flatten(has_both_cousins)
    # get labels from parent pdgId (fill none with 100 to filter them)
    labels_flat = np.abs(ak.fill_none(ak.flatten(parent_pdgid),100).to_numpy())
    labels_flat[has_both_cousins_flat] = -6 # assign jets with both cousins as top1
    
    # W jet labels
    labels_W_flat = np.copy(labels_flat)
    labels_W_flat[labels_W_flat!=24]=0
    labels_W_flat[labels_W_flat==24]=1
    labels_W = ak.unflatten(labels_W_flat, jet_counts)
    
    
    # top1 jet labels
    labels_top1_flat = np.copy(labels_flat)
    labels_top1_flat[labels_top1_flat!=-6]=0
    labels_top1_flat[labels_top1_flat==-6]=1
    labels_top1 = ak.unflatten(labels_top1_flat, jet_counts)
    
    # top2 jet labels
    labels_top2_flat = np.copy(labels_flat)
    labels_top2_flat[labels_top2_flat!=6]=0
    labels_top2_flat[labels_top2_flat==6]=1
    labels_top2 = ak.unflatten(labels_top2_flat, jet_counts)
    
    # top2 jet labels
    labels_other_flat = np.zeros(labels_flat.shape)
    labels_other_flat[(labels_flat!=6) & (labels_flat!=-6) & (labels_flat!=24)]=1
    labels_other = ak.unflatten(labels_other_flat, jet_counts)
    
    # labels = ak.concatenate([x[..., np.newaxis] for x in ak.unzip(labels)], axis=1)
    labels = ak.concatenate([labels_W[..., np.newaxis],
                             labels_top1[..., np.newaxis],
                             labels_top2[..., np.newaxis],
                             labels_other[..., np.newaxis]],axis=2)
    print(labels_W[0])
    print(labels_top1[0])
    print(labels_top2[0])
    print(labels_other[0])
    print(labels[0])
    
    labels_id = ak.unflatten(labels_flat, jet_counts)

    has_W = ak.sum(labels_id==24,axis=-1) == 2
    has_top2 = ak.sum(labels_id==6,axis=-1) == 1
    has_top1 = ak.sum(labels_id==-6,axis=-1) == 1
    training_event_filter = has_W & has_top2 & has_top1

    selected_jets_region = selected_jets_region[training_event_filter]
    selected_electrons_region = selected_electrons_region[training_event_filter]
    selected_muons_region = selected_muons_region[training_event_filter]
    labels = labels[training_event_filter]
    
    return selected_jets_region, selected_electrons_region, selected_muons_region, labels


# %%
def get_data_mask(data, length):
    mask_shape = data.shape[:-1]
    data_mask = torch.full(size=mask_shape, fill_value=False, dtype=torch.bool)
    for m, l in zip(data_mask, length):
        m[: l].fill_(True)
    return data_mask


# %%
events = NanoEventsFactory.from_root("https://xrootd-local.unl.edu:1094//store/user/AGC/nanoAOD/TT_TuneCUETP8M1_13TeV-powheg-pythia8/cmsopendata2015_ttbar_19980_PU25nsData2015v1_76X_mcRun2_asymptotic_v12_ext3-v1_00000_0004.root", 
                                     treepath="Events", entry_stop=5000).events()

# %%
jets, electrons, muons, labels = filterEvents(events.Jet, events.Electron, 
                                              events.Muon, events.GenPart)

# %%
for i in range(20):
    print(jets.pt[i])
    print(labels[i])
    print()

# %%
print(len(jets.pt))
print(len(labels))


# %%
@dataclasses.dataclass
class Batch:
    data: torch.Tensor
    target: torch.Tensor
    length: torch.Tensor
    mask: torch.Tensor

    def to(self, device):
        return Batch(*[each.to(device) for each in dataclasses.astuple(self)])


# %%
features = [jets.pt, jets.eta, jets.phi, jets.mass]
features_zipped = ak.zip(features)
features_awkward = ak.concatenate([x[..., np.newaxis] for x in ak.unzip(features_zipped)], axis=2)
njets = ak.num(features_awkward)

features_padded = torch.nn.utils.rnn.pad_sequence([torch.Tensor(features_awkward[i]) 
                                                   for i in range(len(features_awkward))], batch_first=True) 
mask = get_data_mask(features_padded,njets)
labels_torch = torch.nn.utils.rnn.pad_sequence([torch.Tensor(labels[i])
                                                for i in range(len(labels))], batch_first=True) 

dataset = Batch(features_padded, labels_torch, torch.LongTensor(njets), mask)

# %%
ievt = 32
print(dataset.data[ievt])
print(dataset.length[ievt])
print(dataset.mask[ievt])
print(dataset.target[ievt])

# %%
