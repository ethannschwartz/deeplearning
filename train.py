import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nltk.tokenize import word_tokenize
from dataset import NomenclatureDataset
from model import NomenclatureModel

# Define your model, optimizer, and other parameters

# Define the loss function
criterion = nn.MSELoss()

import nltk

nltk.download('punkt')

# Prepare your data

chemical_names = [
    "methane",
    "ethane",
    "propane",
    "butane",
    "pentane",
    "hexane",
    "heptane",
    "octane",
    "nonane",
    "decane",
]  # List of initial chemical names

# Tokenize chemical names into individual characters
tokenized_names = [list(name) for name in chemical_names]

# Create a mapping of unique tokens to numerical IDs
token_to_id = {}
for name in tokenized_names:
    for token in name:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)

# Convert chemical names to numerical sequences
names = []
max_length = max(len(name) for name in tokenized_names)  # Find the maximum length of the names
for name in tokenized_names:
    numerical_sequence = [token_to_id[token] for token in name]
    # Pad the sequence with padding token (0) to match the maximum length
    padded_sequence = numerical_sequence + [0] * (max_length - len(numerical_sequence))
    names.append(torch.tensor(padded_sequence))

# Convert the list of tensor sequences to a tensor
names = torch.stack(names)

# Create the dataset using the converted names
dataset = NomenclatureDataset(names)

# Create your model instance
input_size = 10  # Size of the vocabulary
hidden_size = 10  # Size of the hidden layer
output_size = 10  # Size of the output vocabulary
model = NomenclatureModel(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Custom collate function
def collate_fn(batch):
    # Extract inputs and targets from the batch
    batch_inputs = [item[0] for item in batch]
    batch_targets = [item[1] for item in batch]

    return batch_inputs, batch_targets


# Create an instance of DataLoader with the custom collate function
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # ...
    # Code for data loading, batching, and preprocessing

    # Iterate over your batches
    for batch_inputs, batch_targets in dataloader:
        print('batch_inputs', batch_inputs)
        print('batch_targets', batch_targets)

torch.save(model.state_dict(), 'model.py')
