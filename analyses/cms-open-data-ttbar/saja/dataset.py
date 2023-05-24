import dataclasses
from typing import Union
from typing import List
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import tqdm
import uproot
import awkward as ak
from coffea.nanoevents.methods.nanoaod import JetArray, MuonArray, ElectronArray


def get_data_mask(data, length):
    mask_shape = data.shape[:-1]
    data_mask = torch.full(size=mask_shape, fill_value=False, dtype=torch.bool)
    for m, l in zip(data_mask, length):
        m[: l].fill_(True)
    return data_mask

def get_isjet(target, length):
    shape = target.shape
    isjet = torch.full(size=shape, fill_value=False, dtype=torch.bool)
    for i, (m, l) in enumerate(zip(isjet, length)):
        m[: l-1].fill_(True)
    return isjet


@dataclasses.dataclass
class Batch:
    data: Tensor
    target: Tensor
    length: Tensor
    mask: Tensor
    isjet: Tensor

    def to(self, device):
        return Batch(*[each.to(device) for each in dataclasses.astuple(self)])


# TODO typing
class JetPartonAssignmentDataset(torch.utils.data.Dataset):

    def __init__(self,
                 jets: JetArray,
                 target: str,
                 num_workers: int = 1,
                 electrons: ElectronArray = None,
                 muons: MuonArray = None
    ) -> None:
        """
        Args:
        """
        self.jets = jets
        self.target = target
        self.electrons = electrons
        self.muons = muons
        
        self._examples = self._process(self.jets, self.target, 
                                       chunk_electrons=self.electrons, chunk_muons=self.muons)

    def _process(self, chunk_jets, target, chunk_electrons=None, chunk_muons=None):
        if (not chunk_electrons is None) and (not chunk_muons is None):
            combined_pt = ak.concatenate((chunk_jets.pt, chunk_electrons.pt),axis=-1)
            combined_pt = ak.concatenate((combined_pt, chunk_muons.pt),axis=-1)
            combined_eta = ak.concatenate((chunk_jets.eta, chunk_electrons.eta),axis=-1)
            combined_eta = ak.concatenate((combined_eta, chunk_muons.eta),axis=-1)
            combined_phi = ak.concatenate((chunk_jets.phi, chunk_electrons.phi),axis=-1)
            combined_phi = ak.concatenate((combined_phi, chunk_muons.phi),axis=-1)
            combined_mass = ak.concatenate((chunk_jets.mass, chunk_electrons.mass),axis=-1)
            combined_mass = ak.concatenate((combined_mass, chunk_muons.mass),axis=-1)
        else:
            combined_pt = chunk_jets.pt
            combined_eta = chunk_jets.eta
            combined_phi = chunk_jets.phi
            combined_mass = chunk_jets.mass
            
        data_chunk = [combined_pt, combined_eta, combined_phi, combined_mass]
        data_chunk = zip(*data_chunk)
        data_chunk = [np.stack(each, axis=1) for each in data_chunk]
        data_chunk = [np.array(each).astype(np.float32) for each in data_chunk]
        data_chunk = [torch.from_numpy(each) for each in data_chunk]
        
        target_chunk = [np.concatenate((np.array(each),[0.0])).astype(np.int64) for each in target]
        target_chunk = [torch.from_numpy(each) for each in target_chunk]

        example_chunk = list(zip(data_chunk, target_chunk))
        return example_chunk

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]
 
    @classmethod
    def collate(cls, batch):
        data, target = list(zip(*batch))
        length = torch.LongTensor([each.size(0) for each in data])
        length_target = torch.LongTensor([each.size(0) for each in target])
        data = pad_sequence(data, batch_first=True)
        mask = get_data_mask(data, length)
        target = pad_sequence(target, batch_first=True)
        isjet = get_isjet(target, length_target)
        return Batch(data, target, length, mask, isjet)
