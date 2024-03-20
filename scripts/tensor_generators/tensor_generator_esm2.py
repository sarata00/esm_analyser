# Script to extract all the results from ESM-2 model:
# attentions, contacts and last hidden state

import torch
import esm
import argparse
import pickle
import os
import time

# Function to print verbose 
def print_v(text):
        if verbose:
            print(text)


def extract_features(model, n_layers, batch_converter, device, data_list=list, emb_normalized=False):

    # Create empty lists to store the model outputs:
    all_last_hidden_states = []
    all_attentions = []
    all_contacts = []
    
    # Iterate over the DataFrame
    for id, sequence in data_list:
        
        # Prepare data
        data = [(str(id), sequence)]
    
        # Convert a batch of sequences into Torch tensors using batch_converter
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # Extract logits
        # Disable contacts and repr_layers to only get mutation predictions
        
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[n_layers], return_contacts=True)

        # Get the results:
        representations = results["representations"][n_layers].detach().cpu()
        attentions = results["attentions"].detach().cpu()
        contacts = results["contacts"].detach().cpu()

        # Append the hidden states and attentions to the lists
        all_last_hidden_states.append(representations) 
        all_attentions.append(attentions)
        all_contacts.append(contacts)

        # Clean up
        del batch_tokens, results
        torch.cuda.empty_cache()
    
    # Once we obtain all the results, let's stack all the results in a single tensor:
    last_hidden_state = torch.stack(all_last_hidden_states, dim=1)
    attention_weigths = torch.stack(all_attentions,dim=1)
    contacts = torch.stack(all_contacts, dim=1)

    # Remove the first dimension (batch_size) if it is equal to 1
    if last_hidden_state.shape[0] == 1:
        last_hidden_state = last_hidden_state.squeeze(dim=0)
    if attention_weigths.shape[0] == 1:
        attention_weigths = attention_weigths.squeeze(dim=0)
    if contacts.shape[0] == 1:
        contacts = contacts.squeeze(dim=0)

    if emb_normalized:
        last_hidden_state = torch.nn.functional.normalize(last_hidden_state, dim=-1)

        
    return last_hidden_state, attention_weigths, contacts


# 1. Define input and output arguments
parser= argparse.ArgumentParser(description="This script extracts the embeddings, attention weights and contacts generated \
                                    by using a ESM-2 model and a given list of protein sequences. This information is stored as \
                                    a tensor object.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-i", "--input_path", type=str, help="mutant library input path.")
parser.add_argument("-m", "--model_name", type=str, help="ESM-2 model name")
parser.add_argument("-o", "--output_path",type=str, help="output path.",default="./")
parser.add_argument("-norm", "--normalized",type=bool, default=False)
parser.add_argument("-v", "--verbose", type=bool, default=False)
args = vars(parser.parse_args())


# Set them as variables
input_path = args["input_path"]           # first argument is the name of the input library PKL file
model_name = args["model_name"]           # second argument is the name of the ESM-2 model
output_path = args["output_path"]         # last argument is the name of the output PKL file
verbose = args["verbose"]


# 2. Load ESM-2 model
if model_name == "esm2_t36_3B_UR50D":
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    n_layers = model.num_layers

elif model_name == "esm2_t12_35M_UR50D":
     model,alphabet = esm.pretrained.esm2_t12_35M_UR50D()
     n_layers = model.num_layers

batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the mutant library
data = pickle.load(open(input_path, "rb"))

# Run now the model
start = time.time()

last_hidden_state, attentions_weigths, contacts = extract_features(model, n_layers, batch_converter, device, data)

end = time.time()

    # Print the time it takes
time = end - start
print_v(f"It takes {time:.2f} seconds")

# 3. Store the results in a PKL file

# Get the absolute path of the output directory
output_dir = os.path.abspath(output_path)

# Ensure the parent directory exists
if not os.path.exists(f"{output_dir}/{model_name}"):
    os.makedirs(f"{output_dir}/{model_name}")

new_path = f"{output_dir}/{model_name}"

# Generate the output file paths
last_hidden_states_path = os.path.join(new_path, f"last_hidden_{model_name}.pth")
attentions_path = os.path.join(new_path, f"attentions_{model_name}.pth")
contacts_path = os.path.join(new_path, f"contacs_{model_name}.pth")

# Now create the tensors for each result:
torch.save(last_hidden_state, last_hidden_states_path)
torch.save(attentions_weigths, attentions_path)
torch.save(contacts, contacts_path)

print_v("The process ends")