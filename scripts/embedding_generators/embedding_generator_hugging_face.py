# Tensor generator

# 1. Libraries and functions
import argparse
import os
import time
import torch
from transformers import EsmModel, AutoTokenizer


# Function to create a dictionary from
# the sequences of the fasta file
def read_fasta(file_path):
    """
    This function creates a dictionary using as input a fasta file in which there is a list of protein sequences.
    As a result, it returns a dictionary whose keys are the ID of the sequences and as values the associated
    sequences with that ID. 
    """
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


def extract_embeddings(fasta_dict, tokenizer, model, model_name, output_path):
    # Create empty lists to store the model outputs,
    last_hidden_states = []

    # Process each sequence of the dictionary
    start = time.time()    
    for seq_name, seq_data in fasta_dict.items():

        # Tokenize the sequence
        inputs = tokenizer(seq_data, return_tensors="pt")

        # Pass the tokenized sequence through the model
        with torch.no_grad():
            model_outputs = model(**inputs)

        # Append the hidden states and attentions to the lists
        last_hidden_states.append(model_outputs.last_hidden_state) 
    end = time.time()          

    # Print the time it takes
    time = end - start
    print(f"It takes {time:.2f} seconds")

    # 5. Concatenate the results
    ## 5.1. Last hidden state results
    concatenated_last_hidden_states = torch.stack(last_hidden_states, dim=1)
    # We have to normalize the embeddings and remove the first dimension (batch_size)
    concatenated_last_hidden_states_normalized = torch.nn.functional.normalize(concatenated_last_hidden_states, dim=3)
    concatenated_last_hidden_states_normalized = torch.squeeze(concatenated_last_hidden_states_normalized, dim=0)


    print("Creating the tensors...")

    # 6. Save the final tensors in the given output path
    checkpoint_modified = model_name.split("/", 1)[1]    # Remove the "facebook/" part
    output_dir = os.path.abspath(output_path)            # Get the absolute path of the output directory

    if not os.path.exists(f"{output_dir}/{checkpoint_modified}"):
        os.makedirs(f"{output_dir}/{checkpoint_modified}")

    # Generate the output file paths
    normalized_last_hidden_state_file_path = os.path.join(f"{output_dir}/{checkpoint_modified}", f"normalized_last_hidden_state_data_variants_{checkpoint_modified}.pth")

    # Check if the file is being created
    print("Normalized Last Hidden State File:", normalized_last_hidden_state_file_path)

    # Save the final tensors in the given output path
    torch.save(concatenated_last_hidden_states_normalized, normalized_last_hidden_state_file_path)

    print("The process ends") 
    


def main():
    # 2. Define input and output arguments
    parser= argparse.ArgumentParser(description="This script extracts the embeddings and attention weights generated \
                                        by using a ESM-2 model and a given list of protein sequences. This information is stored as \
                                        a tensor object.  ",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_library",type=str, help="input list of protein sequences. In case of complexes, merge both sequence in a single line and separate both monomers using ':'. ")
    parser.add_argument("-m", "--model_name", type=str ,help=" expecify the ESM-2 model you want to use. ")
    parser.add_argument("-o", "--output_path",type=str, help="output path.",default="./")
    parser.add_argument("-v", "--verbose", type=bool, default=False)
    args = vars(parser.parse_args())


    # Validate the arguments
    if args.input_path is None:
        raise ValueError("The input library is required.")
    if args.model_name is None:
        raise ValueError("Model is required.")
    if args.output_path is None:
        raise ValueError("Output path is required.")

    # 3. Create the dictionary
    fasta_dict = read_fasta(args.input_library)

    # 4. Load ESM-2 model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = EsmModel.from_pretrained(args.model_name,
                                    output_hidden_states=True,
                                    output_attentions=True)
    
    # 5. Run the process
    extract_embeddings(fasta_dict, tokenizer, model, args.model_name, args.output_path)


if __name__ == "__main__":
    main()
    print("""\nWork completed!\n""")