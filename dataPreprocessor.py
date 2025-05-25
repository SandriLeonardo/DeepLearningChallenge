import os
import sys
import torch
import json
from tqdm import tqdm
import signal

#Set up path to the src directory for import of pre-made teacher modules
src_path = os.path.join(os.path.abspath('.'), 'hackaton', 'src')
sys.path.insert(0, src_path)
from loadData import dictToGraphObject

def signal_handler(sig, frame):
    print('\nProcessing interrupted!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def preprocess_dataset(json_path, output_dir):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        graphs = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, graph_dict in enumerate(tqdm(graphs, desc=f"Processing {os.path.basename(json_path)}")):
        if i % 1000 == 0:  # Log every 1000 graphs
            print(f"Progress: {i}/{len(graphs)} ({i/len(graphs)*100:.1f}%)")
        
        graph_obj = dictToGraphObject(graph_dict)
        torch.save(graph_obj, f"{output_dir}/graph_{i}.pt")
    
    torch.save(len(graphs), f"{output_dir}/num_graphs.pt")
    print(f"✓ Completed: {len(graphs)} graphs saved to {output_dir}")

# Process all datasets
data_folders = ['A', 'B', 'C', 'D']
for folder in data_folders:
    for split in ['train', 'test']:
        json_path = f"./data/{folder}/{split}.json"
        output_dir = f"./data/data_processed/{folder}/{split}"
        
        if os.path.exists(json_path):
            preprocess_dataset(json_path, output_dir)