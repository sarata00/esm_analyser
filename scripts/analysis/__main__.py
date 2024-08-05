#!/usr/bin/env python3
# Author: Sara Tolosa Alarcón

import argparse
import json
import scripts.analysis.basic as bs

from scripts.analysis.tensors import Tensor
from scripts.analysis.correlation import CorrelationAnalysis
from scripts.analysis.dimensionality import DimensionalityReduction, create_annData

def load_json(file_path):
    """ Load a config json file.  """

    with open(file_path, "r") as f:
        config = json.load(f)
    return config

def run(tensor_path,
        exp_data_path = None, 
        mutated_sequence = None, 
        process = None, 
        should_preprocess = True, 
        analysis = None, 
        distance = None,
        method = None,
        output_path = None,
        interface = bool,
        interface_residue_list = list):

    """ Run the process analysis. """

    # 1. Ensure analysis and distance are lists
    if not isinstance(analysis, list):
        analysis = [analysis]

    if not isinstance(distance, list):
        distance = [distance]
       
    # 2. Create an instance of the Tensor class
    # and pre-process the tensor if required.
    tensor_object = Tensor(tensor_path)
    model_name = tensor_object.tensor_name
    
    if should_preprocess:
        tensor_object = tensor_object.preprocessing()

    # 3. Select the analysis process
        #################################
        #   A. Correlation analysis     #
        #################################

    if process == "Correlation":
        print("> Starting Correlation analysis... ")

        correlation = CorrelationAnalysis()

        # Reshape model tensor to a tensor with the following dimensions:
        # [length of mutated sequence, nº of aa, full sequence length, embedding dimensions].
        # Take into account that mutated sequence length and sequence length may not be the same.
        reshaped_tensor = tensor_object.tensor_reshaping(aa_list=bs.amino_acids, mutated_sequence=mutated_sequence)

        # Run correlation analysis
        correlation.run_correlation_analysis(exp_data_path=exp_data_path,
                                                tensor=reshaped_tensor.tensor_file,
                                                analyses=analysis,
                                                distance_list=distance,
                                                mutated_seq=mutated_sequence,
                                                outpath=output_path,
                                                file_name=model_name,
                                                interface=interface,
                                                interface_residue_list=interface_residue_list)
    #################################
    #   B. Dimensionality reduction #
    #################################
                        
    elif process == "Dimensionality_reduction":
        print("> Starting Dimensionality reduction analysis... ")

        # 1. Reshape the tensor to transform it into 2D tensor
        mean_tensor = tensor_object.tensor_file.mean(dim=-2)

        if method == "UMAP":
            # We need to transform our tensor data into an Annotation Matrix:
            adata = create_annData(mean_tensor=mean_tensor, mutated_sequence=mutated_sequence)
            dr = DimensionalityReduction(adata, n_components=2)
            dr.perform_UMAP()
            dr.plot_UMAP(outpath=output_path, file_name=model_name, color="WT")

        if method == "PCA":
            adata = create_annData(mean_tensor=mean_tensor, mutated_sequence=mutated_sequence)
            dr = DimensionalityReduction(adata, n_components=2)
            dr.perform_PCA()
            dr.plot_PCA(outpath=output_path, file_name=model_name, color="WT")



##################################################################################################################################     

def main():
    # 1. Define the command line arguments
    parser = argparse.ArgumentParser(description="Run the process analysis.")
    parser.add_argument("-c", "--config", help="Path to the config file.")
    parser.add_argument("-v", "--verbose", type=bool, default=True, help="Verbose mode.")
    args = parser.parse_args()

    # 2. Validate the arguments
    if args.config is None:
        raise ValueError("The config file is required.")
    
    config_file = load_json(args.config)

    # 3. Extract all the input data from the config file
    tensor_path = config_file["inputs"]["path_to_tensor"]
    mutated_sequence = config_file["inputs"]["mutated_sequence"]
    should_preprocess = config_file["inputs"]["should_preprocess"]

    output_path = config_file["output_path"]
   
   # 4. Determine the process type
    for process_type, process_config in config_file["process"].items():
        if process_type == "Correlation":
            exp_data = process_config["inputs"]["experimental_data"]
            analysis = process_config["inputs"]["analysis"]
            distance = process_config["inputs"]["distance"]
            interface = process_config["inputs"]["interface_analysis"]
            if interface:
                interface_residues = process_config["inputs"]["list_interface_residues"]
            else:
                interface_residues = None

            run(tensor_path=tensor_path, 
                exp_data_path=exp_data, 
                mutated_sequence=mutated_sequence, 
                process=process_type, 
                should_preprocess=should_preprocess, 
                analysis=analysis, 
                distance=distance,
                output_path=output_path,
                interface=interface,
                interface_residue_list=interface_residues)

        elif process_type == "Dimensionality_reduction":
            method = process_config["inputs"]["method"]

            run(tensor_path=tensor_path,
                mutated_sequence=mutated_sequence,
                process=process_type,
                method=method,
                should_preprocess=should_preprocess,
                output_path=output_path)
   

if __name__ == "__main__":
    main()
    print("""\nWork completed!\n""")