#!/usr/bin/env python3
# Author: Sara Tolosa Alarcón

# List of amino acids
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

# Amino acid dictionary by default
def mutant_dictionary(mutated_sequence=str, aa_list=amino_acids):
    """ This function generates a dictionary in which the keys are the residues in order of the mutated sequence
    and their values are the amino acids for which they have been mutated (given as a list). 
    """
    # Create a dictionary of mutation per position
    mutant_dict = {}
    
    for pos,aa in enumerate(mutated_sequence):
        id_aa=aa+"_"+ str(pos)
        mutant_dict[id_aa] = aa_list
    
    return mutant_dict

def index_aa_dictionary(amino_acids = list):
    """ This function generates a dictionary in which the keys are the elements of a given list and
    the values are their position in the list (like an index)."""

    index_dictionary = {}

    for position, aa in enumerate(amino_acids):
        index_dictionary[aa] = position

    return index_dictionary

