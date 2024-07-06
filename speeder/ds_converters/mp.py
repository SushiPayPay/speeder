import torch
from torchvision import datasets, transforms
import multiprocessing
import time

import numpy as np
import os
from PIL import Image


# Define a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-100 dataset
dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# Function to process each chunk of data
def process_chunk(dataset, chunk_indices):
    for idx in chunk_indices:
        # Fetch the data item
        img, label = dataset[idx]
        class_name = dataset.classes[label]
        img = Image.fromarray(np.array(img))
        img.save(os.path.join('./clean_data', class_name, f"{idx}.png"))

if __name__ == "__main__":
    # Number of workers
    N = multiprocessing.cpu_count()
    
    # Calculate chunk size
    chunk_size = len(dataset) // N
    
    # Create chunks of indices
    chunks = [list(range(i * chunk_size, (i + 1) * chunk_size)) for i in range(N)]
    
    # Handle the last chunk which might have more elements
    chunks[-1].extend(range(N * chunk_size, len(dataset)))
    
    args = [(chunk, dataset) for chunk in chunks]


    # Create a pool of workers
    with multiprocessing.Pool(N) as pool:
        # Distribute the computation to the worker processes
        results = pool.starmap(process_chunk, args)
    
    # Flatten the list of results
    flat_results = [item for sublist in results for item in sublist]
    
    # Print some of the results
    print(flat_results[:10])
