#!/usr/bin/env python3

import sys
import torch
from pathlib import Path
from basic import amino_acids

class Tensor:
    def __init__(self, tensor_path):

        self.tensor_name=Path(tensor_path).stem
        self.tensor_path = tensor_path
        self.tensor_file = self.load_tensor()

    def load_tensor(self):
        try:
            self.file = torch.load(self.tensor_path)
            if not isinstance(self.file, torch.Tensor):
                raise TypeError("Loaded object is not a tensor")
            return self.file
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Tensor file '{self.tensor_path}' not found.")
        except Exception as e:
            raise Exception(f"Failed to load tensor: {str(e)}")
        
    def get_shape(self):
        """ It returns the tensor shape."""
        if not hasattr(self,"file"):
            self.tensor_file()
            
        return self.file.shape
    
    def remove_batch_size(self):
        if not hasattr(self,"file"):
            self.tensor_file()

        dims = self.file.dim()
        if self.file.shape[0] == 1 and dims > 3:
            self.file = self.file.squeeze(dim=0)

        return self.file
    
    def normalize_tensor(self):
        """Normalize last dimension of a tensor."""

        if not hasattr(self,"file"):
            self.tensor_file()

        normalized_tensor = torch.nn.functional.normalize(self.file, p=2, dim=-1)
        self.file = normalized_tensor

        return self
    

    def tensor_reshaping(self, aa_list=amino_acids, mutated_sequence=str):
        if not hasattr(self, "file"):
            self.load_tensor()

        seq_length = self.file.shape[-2]
        n_aa = len(aa_list)
        mut_seq = len(mutated_sequence)
        emb_dim = self.file.shape[-1]
        reshaped_tensor = self.file.reshape(mut_seq, n_aa, seq_length, emb_dim)
        self.file = reshaped_tensor

        return self

    def preprocessing(self):
        # Remove the batch_size dimension
        self.remove_batch_size()
        # Normalize the tensor
        self.normalize_tensor()

        return self 