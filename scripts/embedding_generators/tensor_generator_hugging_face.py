# Tensor generator

# 1. Libraries and functions
import argparse
import os
import time
import torch
from transformers import EsmModel, AutoTokenizer


# Function to print verbose 
def print_v(text):
        if verbose:
            print(text)

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


# 2. Define input and output arguments
parser= argparse.ArgumentParser(description="This script extracts the embeddings and attention weights generated \
                                    by using a ESM-2 model and a given list of protein sequences. This information is stored as \
                                    a tensor object.  ",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", "--input_library",type=str, help="input list of protein sequences. In case of complexes, merge both sequence in a single line and separate both monomers using ':'. ")
parser.add_argument("-m", "--model", type=str ,help=" expecify the ESM-2 model you want to use. ")
parser.add_argument("-o", "--output_path",type=str, help="output path.",default="./")
parser.add_argument("-v", "--verbose", type=bool, default=False)
args = vars(parser.parse_args())


# Set them as variables
sequence_list = args["input_library"]     # first argument is the protein sequence
checkpoint = args["model"]                # second argument is the ESM-2 model we want
output_path = args["output_path"]         # last argument is the name of the output PKL file
verbose = args["verbose"]


# 3. Create the dictionary
fasta_dict = read_fasta(sequence_list)


# 4. Run ESM-2 model

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Load the model
model = EsmModel.from_pretrained(checkpoint,
                                 output_hidden_states=True,
                                 output_attentions=True)

# Create empty lists to store the model outputs,
all_last_hidden_states = []
all_attentions = []
all_embeddings = []

# Process each sequence of the dictionary
start = time.time()    # Store starting time

for seq_name, seq_data in fasta_dict.items():

    # Tokenize the sequence
    inputs = tokenizer(seq_data, return_tensors="pt")

    # Pass the tokenized sequence through the model
    with torch.no_grad():
        model_outputs = model(**inputs)

    # Append the hidden states and attentions to the lists
    all_last_hidden_states.append(model_outputs.last_hidden_state) 
    all_attentions.append(model_outputs.attentions)
    all_embeddings.append(model_outputs.hidden_states)

end = time.time()           # Store ending time

# Print the time it takes
time = end - start
print_v(f"It takes {time:.2f} seconds")

# 5. Concatenate the results
## 5.1. Last hidden state results
concatenated_last_hidden_states = torch.stack(all_last_hidden_states, dim=1)
# We have to normalize the embeddings and remove the first dimension (batch_size)
concatenated_last_hidden_states_normalized = torch.nn.functional.normalize(concatenated_last_hidden_states, dim=3)
concatenated_last_hidden_states_normalized = torch.squeeze(concatenated_last_hidden_states_normalized, dim=0)


## 5.2. All embedding results
all_embeddings =  [tupla[0] for tupla in all_embeddings]
concatenated_embeddings = torch.stack(all_embeddings, dim=1)

## 5.3. Attention weight results
all_attentions_tensor = [tupla[0] for tupla in all_attentions]
concatenated_attentions = torch.stack(all_attentions_tensor, dim=1)


print_v("Creating the tensors...")

# 6. Save the final tensors in the given output path
# First I will modify the checkpoint variable and remove the "facebook/" part
checkpoint_modified = checkpoint.split("/", 1)[1]

# Get the absolute path of the output directory
output_dir = os.path.abspath(output_path)

# Ensure the parent directory exists
if not os.path.exists(f"{output_dir}/{checkpoint_modified}"):
    os.makedirs(f"{output_dir}/{checkpoint_modified}")


# Generate the output file paths
normalized_last_hidden_state_file_path = os.path.join(f"{output_dir}/{checkpoint_modified}", f"normalized_last_hidden_state_data_variants_{checkpoint_modified}.pth")
attentions_file_path = os.path.join(f"{output_dir}/{checkpoint_modified}", f"attentions_variants_{checkpoint_modified}.pth")
embeddings_file_path = os.path.join(f"{output_dir}/{checkpoint_modified}", f"embeddings_variants_{checkpoint_modified}.pth")

# Check if the file is being created
print("Normalized Last Hidden State File:", normalized_last_hidden_state_file_path)
print("Attentions File:", attentions_file_path)
print("Embeddings File:", embeddings_file_path)

# Save the final tensors in the given output path
torch.save(concatenated_last_hidden_states_normalized, normalized_last_hidden_state_file_path)
torch.save(concatenated_last_hidden_states_normalized, attentions_file_path )
torch.save(concatenated_last_hidden_states_normalized, embeddings_file_path)


print_v("The process ends") 