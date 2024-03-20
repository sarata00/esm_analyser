#!/usr/bin/env python3

from basic import amino_acids
import torch

class Distances:

    def __init__(self, tensor, mutated_sequence, aa_list=amino_acids):
        self.tensor = tensor
        self.wt_tensor = self.find_wt(mutated_sequence, aa_list)
        self.tensor_differences = self.calculate_diferences()
        
    def find_wt(self, mutated_sequence, aa_list=amino_acids):
        """ Find the tensor which contains the wildtype results. 
        The wildtype tensor is the one that contains the wildtype sequence
        this means that where the residues of the sequence match the residue
        in aa_list is the wildtype embedding. We just need to find one of them.
        -----------------------------------
        Parameters:
                - mutated_sequence: sequence of the protein with the mutations.
                - aa_list: list of amino acids to be used.
        -----------------------------------
        """

        first_residue = mutated_sequence[0]
        wt_index = aa_list.index(first_residue)
        wt_tensor = self.tensor[0,wt_index,:,:]

        return wt_tensor

    def calculate_diferences(self):
        """ Calculate the differences between the variant and the wildtype tensors."""
        tensor_differences = self.tensor - self.wt_tensor

        return tensor_differences


    def get_euclidean_distance(self):
        """ Calculate the euclidean distance between the variant and the wildtype tensors.
        This will return a 2D matrix: [mutated_protein_residues, mutant_aa].
        """
        euclidean_distance = torch.linalg.norm(self.tensor_differences, dim=(-2,-1))

        return euclidean_distance

    def get_cosine_distance(self):
        """ Calculate the cosine distance between the variant and the wildtype tensors."""

        # Reshape wild type tensor to dimensions [1,1,Z*W]
        reshaped_wt = self.wt_tensor.view(-1)
        reshaped_wt = reshaped_wt.unsqueeze(0).unsqueeze(0)

        # Reshape variants tensor to dimensions [X,Y,Z*W]
        reshaped_tensor = self.tensor.view(*self.tensor.shape[:-2], -1)

        cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        cosine_similarity = cos(reshaped_tensor, reshaped_wt)
        cosine_distance = 1 - cosine_similarity

        return cosine_distance


    def global_analysis(self, distance="euclidean"):
        """ Perform the global analysis.
        -----------------------------------
        Parameters:
                - distance: distance metric to be used. It can be "euclidean" or "cosine".
        -----------------------------------
        """

        if distance=="euclidean":
            global_distance = self.get_euclidean_distance()

        elif distance=="cosine":
            global_distance = self.get_cosine_distance()
        
        return global_distance


    def positional_analysis(self, distance="euclidean"):
        """ Perform the positional analysis.
        -----------------------------------
        Parameters:
                - distance: distance metric to be used. It can be "euclidean" or "cosine".
        -----------------------------------
        """        
        
        positional_distance = torch.zeros((self.tensor_differences.shape[0], self.tensor_differences.shape[1]))

        if distance == "euclidean":
            for i in range((self.tensor_differences.shape[0])):
                for j in range(self.tensor_differences.shape[1]):
                    slice_norm = torch.norm(self.tensor_differences[i,j,i,:], dim=-1)
                    
                    positional_distance[i,j] = slice_norm

    
        elif distance =="cosine":
            cos = torch.nn.CosineSimilarity(dim=-1)
            # Create a tensor full of zeros and with the desired dimensionality
            positional_distance = torch.zeros((self.tensor_differences.shape[0], self.tensor_differences.shape[1]))

            for i in range((self.tensor_differences.shape[0])):
                for j in range(self.tensor_differences.shape[1]):
                    # Calculate cosine similarity
                    cosine_similarity = cos(self.tensor[i,j,i,:], self.wt_tensor[i,:])
                    # Calculate cosine distance
                    cosine_distance = 1 - cosine_similarity
                    
                    positional_distance[i,j] = cosine_distance

        return positional_distance
