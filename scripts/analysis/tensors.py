#!/usr/bin/env python3

import torch
from pathlib import Path
from scripts.analysis.basic import amino_acids

class Tensor:
    def __init__(self, tensor_path):
        self.tensor_name = Path(tensor_path).stem
        self.tensor_path = tensor_path
        self.tensor_file = self.load_tensor()
        
    def load_tensor(self):
        try:
            tensor = torch.load(self.tensor_path, weights_only=True)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Loaded object is not a tensor")
            
            return tensor
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Tensor file '{self.tensor_path}' not found.")
        except Exception as e:
            raise Exception(f"Failed to load tensor: {str(e)}")
        
    def get_shape(self):
        """ It returns the tensor shape."""     
        return self.tensor_file.shape
    
    def remove_batch_size(self):
        """Remove the batch_size dimension of the tensor."""
        dims = self.tensor_file.dim()
        if self.tensor_file.shape[0] == 1 and dims > 3:
            self.tensor_file = self.tensor_file.squeeze(dim=0)

        return self.tensor_file
    
    def normalize_tensor(self):
        """Normalize last dimension of a tensor."""
        normalized_tensor = torch.nn.functional.normalize(self.tensor_file, p=2, dim=-1)
        self.tensor_file = normalized_tensor

        return self
    

    def tensor_reshaping(self, aa_list=amino_acids, mutated_sequence=str):
        """Reshape the tensor to the desired shape."""
        seq_length = self.tensor_file.shape[-2]
        n_aa = len(aa_list)
        mut_seq = len(mutated_sequence)
        emb_dim = self.tensor_file.shape[-1]
        reshaped_tensor = self.tensor_file.reshape(mut_seq, n_aa, seq_length, emb_dim)
        self.tensor_file = reshaped_tensor

        return self

    def preprocessing(self):
        """Preprocess the tensor by removing batch size and normalizing it."""
        self.remove_batch_size()
        self.normalize_tensor()

        return self 