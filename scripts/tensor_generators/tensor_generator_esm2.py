# Script to extract the last_hidden_state from ESM-2 models.
# Author: Sara Tolosa Alarcón

import torch
import esm
import argparse
import pickle
import os
import time

def print_v(text):
        """Print verbose"""
        if verbose:
            print(text)

def load_esm_model(model_name):
    """Load ESM-2 model"""
    models = {
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
        "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
    }
    
    model, alphabet = models[model_name]()
    n_layers = model.num_layers

    return model, alphabet, n_layers

def read_fasta(file_path):
    """ Read a FASTA file and return a dictionary of IDs and sequences."""
    dict_sequences = {}

    with open(file_path, "r") as file:
        current_sequence_name = None
        current_sequence = ""

        for line in file:
            line = line.strip()
            if line.startswith(">"):
                # New sequence header
                if current_sequence_name is not None:
                    dict_sequences[current_sequence_name] = current_sequence
                current_sequence_name = line[1:]
                current_sequence = ""
            else:
                # Append sequence data
                current_sequence += line

        # Add the last sequence
        if current_sequence_name is not None:
            dict_sequences[current_sequence_name] = current_sequence

    return dict_sequences

def extract_features(model, n_layers, batch_converter, device, fasta_file=dict, emb_normalized=False):

    # Create empty lists to store the model outputs:
    all_last_hidden_states = []
    
    # Iterate over the 
    for id, sequence in fasta_file.items():
        
        # Prepare data
        data = [(str(id), sequence)]
    
        # Convert a batch of sequences into Torch tensors using batch_converter
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # Run the model:        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[n_layers])
        # Get the results:
        representations = results["representations"][n_layers].detach().cpu()
  
        # Append the hidden states and attentions to the lists
        all_last_hidden_states.append(representations) 

    # Once we obtain all the results, let's stack all the results in a single tensor:
    last_hidden_state = torch.stack(all_last_hidden_states, dim=1)

    # Remove the first dimension (batch_size) if it is equal to 1
    if last_hidden_state.shape[0] == 1:
        last_hidden_state = last_hidden_state.squeeze(dim=0)
    # Normalize the last dimension of the resulting tensor 
    if emb_normalized:
        last_hidden_state = torch.nn.functional.normalize(last_hidden_state, dim=-1)

    return last_hidden_state


def main(): 
    # 1. Define input and output arguments
    parser = argparse.ArgumentParser(description="This script extracts the embeddings, attention weights and contacts generated \
                                        by using a ESM-2 model and a given list of protein sequences. This information is stored as \
                                        a tensor object.",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_path", type=str, help="mutant library input path.")
    parser.add_argument("-m", "--model_name", type=str, help="ESM-2 model name")
    parser.add_argument("-o", "--output_path",type=str, help="output path.",default="./")
    parser.add_argument("-norm", "--normalized",type=bool, default=False)
    parser.add_argument("-v", "--verbose", type=bool, default=False)
    args = vars(parser.parse_args())

    # Validate the arguments
    if args.input_path is None:
        raise ValueError("The input library is required.")
    if args.model_name is None:
        raise ValueError("Model is required.")
    if args.output_path is None:
        raise ValueError("Output path is required.")

    # Set them as variables
    input_path = args["input_path"]           # First argument is the name of the input library (FASTA file)
    model_name = args["model_name"]           # Fecond argument is the name of the ESM-2 model
    output_path = args["output_path"]         # Last argument is the name of the output as a .pth file


    # 2. Load ESM-2 model
    model, alphabet, n_layers = load_esm_model(model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use CUDA if available
    model.to(device)

    # Load the mutant library
    data = read_fasta(input_path)

    # Run now the model and store the time it takes
    start = time.time()
    last_hidden_state = extract_features(model, n_layers, batch_converter, device, data)
    end = time.time()
    
    time = end - start
    print_v(f"It takes {time:.2f} seconds")

    # 3. Store the results in a PKL file
    output_dir = os.path.abspath(output_path)                # Absolute path of the output directory
    if not os.path.exists(f"{output_dir}/{model_name}"):     # Ensure the parent directory exists
        os.makedirs(f"{output_dir}/{model_name}")
    new_path = f"{output_dir}/{model_name}"

    # Generate the output file paths
    last_hidden_states_path = os.path.join(new_path, f"last_hidden_{model_name}.pth")
    # Now create the tensors for each result:
    torch.save(last_hidden_state, last_hidden_states_path)

if __name__ == "__main__":
    main()
    print("""\nWork completed!\n""")