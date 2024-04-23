#!/usr/bin/env python3

import sys
import os
from basic import amino_acids, mutant_dictionary
import warnings
import scanpy as sc
import matplotlib.pyplot as plt



def create_annData(mean_tensor, mutated_sequence):
    """ Create an AnnData object from a tensor. 
    --------------------------------------------
    Parameters:
            - mean_tensor: 2D tensor with the mean value of sequence_length dimension.
            - mutated_sequence: protein sequence which has been mutated in the analysis.
    ------------------------------------------
    """
    
    # 1. Generate a mutant dictionary in which keys are each amino acid of 
    # the mutated sequence and values each of the amino acids by which they are mutated:
    mutant_dict = mutant_dictionary(mutated_sequence=mutated_sequence, aa_list=amino_acids)

    if len(mean_tensor.shape) == 2:
        adata=sc.AnnData(X=mean_tensor.numpy())
        wt_values = []
        mut_values = []

        for aa, mut_list in mutant_dict.items():
            for mut in mut_list:
                wt_values.append(aa)
                mut_values.append(mut)

        adata.obs["WT"] = wt_values
        adata.obs["Mut"] = mut_values
        adata.obs["variant"] = [f"{i}" for i in range(mean_tensor.shape[0])]

    else:
        sys.stderr.write(f"Error, the loaded tensor is not 2D. Its shape is: {mean_tensor.shape}")
        sys.exit(1)
    
    return adata


class DimensionalityReduction():

    def __init__(self, adata, n_components = 2):
        self.adata = adata
        self.n_components = n_components

    def perform_UMAP(self):
        """ Perform UMAP analysis."""

        sc.pp.neighbors(self.adata, use_rep="X", metric="cosine")
        sc.tl.umap(self.adata)
        return self.adata

    def plot_UMAP(self, outpath, file_name, color):
        """ Plot UMAP analysis results."""

        fig, ax = plt.subplots(figsize=(10,10))

        figures_dir = os.path.join(outpath, "figures_DR")
        if not os.path.isdir(figures_dir):
            os.makedirs(figures_dir)

        f = f"{figures_dir}/{file_name}_umap.svg"

        warnings.filterwarnings("ignore", category=UserWarning)
        sc.pl.umap(self.adata, color=color, edges=False, ax=ax, show=False, return_fig=True)
        plt.gcf().set_size_inches(15, 10)
        plt.tight_layout()


        fig.savefig(f, format="svg")
        plt.close(fig)
        

    def perform_PCA(self):
        """ Perform PCA analysis."""

        sc.pp.pca(self.adata, self.n_components, random_state=42)
        return self.adata

    def plot_PCA(self, outpath, file_name, color):
        """ Plot PCA analysis results."""

        fig, ax = plt.subplots(figsize=(10,10))

        figures_dir = os.path.join(outpath, "figures_DR")
        if not os.path.isdir(figures_dir):
            os.makedirs(figures_dir)

        f = f"{figures_dir}/{file_name}_pca.svg"

        warnings.filterwarnings("ignore", category=UserWarning)
        sc.pl.pca(self.adata, color=color, edges=False, ax=ax, show=False, return_fig=True)
        plt.gcf().set_size_inches(15, 10)
        plt.tight_layout()

        fig.savefig(f, format="svg")
        plt.close(fig)