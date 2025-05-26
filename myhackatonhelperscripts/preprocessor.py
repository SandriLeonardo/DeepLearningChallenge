import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, degree
from sklearn.model_selection import train_test_split
import os
import numpy as np
from tqdm import tqdm
import argparse
import gc


# RWSE Configuration
RWSE_MAX_K = 16


class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None, use_processed=False):
        """
        Enhanced GraphDataset with support for processed data and multiple loading modes.
        
        Args:
            filename: Path to data file or processed directory
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform to apply to data
            use_processed: If True, load from processed directory structure
        """
        if use_processed:
            self.processed_dir = filename
            self.num_graphs = torch.load(f"{filename}/num_graphs.pt")
        else:
            self.raw = filename
            self.num_graphs, self.graphs_dicts = self._count_and_load_graphs() 
        self.use_processed = use_processed
        super().__init__(None, transform, pre_transform)

    def len(self):
        return self.num_graphs  
    
    def get(self, idx):
        if self.use_processed:
            return torch.load(f"{self.processed_dir}/graph_{idx}.pt")
        else:
            return GraphDataset.dictToGraphObject(self.graphs_dicts[idx])

    def _count_and_load_graphs(self):
        """Load and count graphs from JSON file with support for compressed files."""
        print(f"Loading graphs from {self.raw}...")
        print("This may take a few minutes, please wait...")
        
        if self.raw.endswith(".gz"):
            with gzip.open(self.raw, "rt", encoding="utf-8") as f:
                graphs_dicts = json.load(f)
        else:
            with open(self.raw, "r", encoding="utf-8") as f:
                graphs_dicts = json.load(f)
        
        return len(graphs_dicts), graphs_dicts

    @staticmethod
    def dictToGraphObject(graph_dict, is_test_set=False, graph_idx_info=""):
        """
        Convert graph dictionary to PyTorch Geometric Data object with enhanced processing.
        
        Args:
            graph_dict: Dictionary containing graph data
            is_test_set: Whether this is from test set (affects label handling)
            graph_idx_info: Info string for debugging/logging
        """
        num_nodes = graph_dict.get('num_nodes', 0)
        if not isinstance(num_nodes, int) or num_nodes < 0:
            num_nodes = 0

        # Create node features - using zeros with shape (num_nodes, 1)
        x = torch.zeros(num_nodes, 1, dtype=torch.long)

        # Handle edge data
        raw_edge_index = graph_dict.get('edge_index', [])
        raw_edge_attr = graph_dict.get('edge_attr', [])
        edge_attr_dim = graph_dict.get('edge_attr_dim', 7)
        
        if not isinstance(edge_attr_dim, int) or edge_attr_dim <= 0:
            edge_attr_dim = 7

        if num_nodes == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
        else:
            edge_index = torch.tensor(raw_edge_index, dtype=torch.long) if raw_edge_index else torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(raw_edge_attr, dtype=torch.float) if raw_edge_attr else torch.empty((0, edge_attr_dim), dtype=torch.float)
            
            # Validate edge_index shape
            if edge_index.numel() > 0 and edge_index.shape[0] != 2:
                print(f"Warning: Invalid edge_index shape for graph {graph_idx_info}. Clearing edges.")
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)

            # Validate edge_attr dimensions
            if edge_attr.numel() > 0:
                if edge_attr.shape[1] != edge_attr_dim:
                    print(f"Warning: Mismatch edge_attr_dim (expected {edge_attr_dim}, got {edge_attr.shape[1]}) for graph {graph_idx_info}.")
                    if 'edge_attr_dim' in graph_dict:
                        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)
                        if edge_index.shape[1] > 0:
                            print(f"  Cleared edge_attr for {graph_idx_info} due to dim mismatch.")

                # Validate edge count consistency
                if edge_index.shape[1] != edge_attr.shape[0]:
                    print(f"Warning: Mismatch edge_index/edge_attr count for graph {graph_idx_info}.")
                    if edge_index.shape[1] == 0:
                        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float)

        # Handle labels
        y_val_raw = graph_dict.get('y')
        y_val = -1
        
        if is_test_set:
            y_val = -1  # Test sets have no labels
        elif y_val_raw is not None:
            temp_y = y_val_raw
            # Handle nested list structures like [[label]] or [label]
            while isinstance(temp_y, list):
                if len(temp_y) == 1:
                    temp_y = temp_y[0]
                else:
                    temp_y = -1
                    break
            if isinstance(temp_y, int):
                y_val = temp_y
            else:
                y_val = -1

        if y_val == -1 and not is_test_set:
            print(f"Warning: 'y' missing or malformed for TRAIN/VAL graph {graph_idx_info}. Using -1.")
        
        y = torch.tensor([y_val], dtype=torch.long)

        # Create Data object
        data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=num_nodes)

        # Add RWSE positional encoding
        if data_obj.num_nodes > 0 and data_obj.num_edges > 0:
            data_obj.rwse_pe = GraphDataset.get_rw_landing_probs(data_obj.edge_index, data_obj.num_nodes, k_max=RWSE_MAX_K)
        else:
            data_obj.rwse_pe = torch.zeros((data_obj.num_nodes, RWSE_MAX_K))
            
        return data_obj

    @staticmethod
    def get_rw_landing_probs(edge_index, num_nodes, k_max):
        """
        Compute Random Walk Structural Encoding (RWSE) landing probabilities.
        
        Args:
            edge_index: Edge connectivity tensor
            num_nodes: Number of nodes in the graph
            k_max: Maximum number of steps for random walk
            
        Returns:
            Tensor of shape (num_nodes, k_max) with landing probabilities
        """
        if num_nodes == 0: 
            return torch.zeros((0, k_max), device=edge_index.device)
        if edge_index.numel() == 0: 
            return torch.zeros((num_nodes, k_max), device=edge_index.device)

        if num_nodes > 1000:
            print(f"Info: RWSE for graph with {num_nodes} nodes. This may be slow.")

        source, _ = edge_index[0], edge_index[1]
        deg = degree(source, num_nodes=num_nodes, dtype=torch.float)
        deg_inv = deg.pow(-1.)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)

        try:
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        except RuntimeError as e:
            max_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
            print(f"Error in to_dense_adj: {e}. Max_idx: {max_idx}, Num_nodes: {num_nodes}. Returning zeros for RWSE.")
            return torch.zeros((num_nodes, k_max), device=edge_index.device)

        P_dense = deg_inv.view(-1, 1) * adj
        rws_list = []
        Pk = torch.eye(num_nodes, device=edge_index.device)

        for _ in range(1, k_max + 1):
            if Pk.numel() == 0 or P_dense.numel() == 0:
                return torch.zeros((num_nodes, k_max), device=edge_index.device)
            try:
                Pk = Pk @ P_dense
            except RuntimeError as e:
                print(f"RuntimeError during Pk @ P_dense: {e}. Returning zeros.")
                return torch.zeros((num_nodes, k_max), device=edge_index.device)
            rws_list.append(torch.diag(Pk))

        return torch.stack(rws_list, dim=1) if rws_list else torch.zeros((num_nodes, k_max), device=edge_index.device)


class MultiDatasetLoader:
    """
    Enhanced data loader supporting multiple datasets with separate processing.
    Handles datasets A, B, C, D with train/validation/test splits.
    """
    
    def __init__(self, base_path=None, val_split_ratio=0.1):
        """
        Initialize the multi-dataset loader.
        
        Args:
            base_path: Base path to dataset directory (defaults to script directory)
            val_split_ratio: Ratio for validation split from training data
        """
        if base_path is None:
            script_base_path = os.path.dirname(os.path.abspath(__file__))
            self.original_dataset_base_path = os.path.join(script_base_path, 'original_dataset')
        else:
            self.original_dataset_base_path = base_path
            
        self.val_split_ratio = val_split_ratio
        self.dataset_names = ['A', 'B', 'C', 'D']
        
        # Setup processed data directory
        script_base_path = os.path.dirname(os.path.abspath(__file__))
        self.processed_dir = os.path.join(script_base_path, 'processed_data_separate')
        os.makedirs(self.processed_dir, exist_ok=True)

    def find_single_json_file(self, directory_path):
        """
        Find a single JSON file in the given directory.
        
        Args:
            directory_path: Path to search for JSON files
            
        Returns:
            Path to JSON file or None if not found/multiple found
        """
        if not os.path.isdir(directory_path):
            print(f"Warning: Directory not found: {directory_path}.")
            return None
        try:
            json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
            if len(json_files) == 1:
                return os.path.join(directory_path, json_files[0])
            elif len(json_files) == 0:
                print(f"Warning: No JSON file found in {directory_path}.")
                return None
            else:
                print(f"Warning: Multiple JSON files found in {directory_path}. Using first: {sorted(json_files)[0]}.")
                return os.path.join(directory_path, sorted(json_files)[0])
        except Exception as e:
            print(f"Error accessing directory {directory_path}: {e}")
            return None

    def find_json_file(self, dataset_name, file_type):
        """
        Find JSON file with flexible path resolution and debugging.
        """
        print(f"  Looking for {file_type}.json for dataset {dataset_name}")
        
        # Try direct file path first
        direct_path = os.path.join(self.original_dataset_base_path, dataset_name, f'{file_type}.json')
        print(f"    Trying direct path: {direct_path}")
        if os.path.exists(direct_path):
            file_size = os.path.getsize(direct_path) / (1024*1024)  # MB
            print(f"    Found! Size: {file_size:.2f} MB")
            return direct_path
            
        # Try looking in subdirectory
        subdir_path = os.path.join(self.original_dataset_base_path, dataset_name, file_type)
        print(f"    Trying subdir path: {subdir_path}")
        if os.path.isdir(subdir_path):
            json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
            if len(json_files) >= 1:
                found_file = os.path.join(subdir_path, sorted(json_files)[0])
                file_size = os.path.getsize(found_file) / (1024*1024)  # MB
                print(f"    Found in subdir! Size: {file_size:.2f} MB")
                return found_file
        
        # List what actually exists
        dataset_dir = os.path.join(self.original_dataset_base_path, dataset_name)
        if os.path.exists(dataset_dir):
            contents = os.listdir(dataset_dir)
            print(f"    Dataset {dataset_name} contains: {contents}")
        else:
            print(f"    Dataset directory {dataset_dir} doesn't exist")
            
        print(f"    No {file_type}.json found for dataset {dataset_name}")
        return None

    def process_single_dataset(self, dataset_name):
        """
        Process a single dataset and create train/val/test splits.
        
        Args:
            dataset_name: Name of the dataset (A, B, C, or D)
            
        Returns:
            dict: Contains 'train', 'val', 'test' splits for this dataset
        """
        print(f"-- Processing dataset: {dataset_name} --")
        dataset_splits = {'train': [], 'val': [], 'test': []}

        # Load Training Data
        actual_train_json_file = self.find_json_file(dataset_name, 'train')

        raw_train_graphs = []
        if actual_train_json_file:
            print(f"  Loading training data from: {actual_train_json_file}")
            try:
                with open(actual_train_json_file, 'r') as f:
                    train_json_list = json.load(f)
                
                print(f"    JSON loaded successfully. Type: {type(train_json_list)}")
                if isinstance(train_json_list, list):
                    print(f"    Found {len(train_json_list)} training samples")
                elif isinstance(train_json_list, dict):
                    print(f"    JSON is dict with keys: {list(train_json_list.keys())}")
                    # Handle case where data might be nested in dict
                    if 'graphs' in train_json_list:
                        train_json_list = train_json_list['graphs']
                        print(f"    Using 'graphs' key, now {len(train_json_list)} samples")
                    elif 'data' in train_json_list:
                        train_json_list = train_json_list['data']
                        print(f"    Using 'data' key, now {len(train_json_list)} samples")
                
                processed_count = 0
                for i, g_data in tqdm(enumerate(train_json_list), total=len(train_json_list),
                                      desc=f"  {dataset_name} Train Data"):
                    if not isinstance(g_data, dict):
                        print(f"    Warning: Item {i} is not dict, type: {type(g_data)}")
                        continue
                    g_data['_source_dataset'] = dataset_name
                    processed_graph = GraphDataset.dictToGraphObject(
                        g_data, is_test_set=False, graph_idx_info=f"{dataset_name}train{i}"
                    )
                    raw_train_graphs.append(processed_graph)
                    processed_count += 1
                    
                print(f"    Successfully processed {processed_count} training graphs")
                    
            except json.JSONDecodeError as e:
                print(f"  ERROR: JSON decode failed for {actual_train_json_file}: {e}")
            except Exception as e:
                print(f"  ERROR loading {actual_train_json_file}: {e}. Skipping.")
        else:
            print(f"  No training JSON file found for dataset {dataset_name}")

        # Split Training Data into Train/Val
        if len(raw_train_graphs) == 0:
            print(f"  WARNING: No training graphs loaded for dataset {dataset_name}")
            dataset_splits['train'] = []
            dataset_splits['val'] = []
        elif len(raw_train_graphs) == 1:
            print(f"  WARNING: Only 1 training sample for dataset {dataset_name}.")
            dataset_splits['train'] = raw_train_graphs
            dataset_splits['val'] = []
        else:
            # Try stratified split
            train_labels = [g.y.item() for g in raw_train_graphs if g.y.item() != -1]

            can_stratify = False
            if train_labels:
                unique_labels, counts = np.unique(train_labels, return_counts=True)
                if len(unique_labels) > 1 and all(c >= 2 for c in counts):
                    can_stratify = True

            try:
                if can_stratify:
                    dataset_splits['train'], dataset_splits['val'] = train_test_split(
                        raw_train_graphs, test_size=self.val_split_ratio, 
                        random_state=42, stratify=train_labels
                    )
                    print(f"  Used stratified split for dataset {dataset_name}")
                else:
                    dataset_splits['train'], dataset_splits['val'] = train_test_split(
                        raw_train_graphs, test_size=self.val_split_ratio, random_state=42
                    )
                    print(f"  Used random split for dataset {dataset_name}")
            except ValueError as e:
                print(f"  Warning: Split failed for dataset {dataset_name} ({e}). Using all for training.")
                dataset_splits['train'] = raw_train_graphs
                dataset_splits['val'] = []

        # Load Test Data
        actual_test_json_file = self.find_json_file(dataset_name, 'test')

        if actual_test_json_file:
            print(f"  Loading test data from: {actual_test_json_file}")
            try:
                with open(actual_test_json_file, 'r') as f:
                    test_json_list = json.load(f)
                
                print(f"    JSON loaded successfully. Type: {type(test_json_list)}")
                if isinstance(test_json_list, list):
                    print(f"    Found {len(test_json_list)} test samples")
                elif isinstance(test_json_list, dict):
                    print(f"    JSON is dict with keys: {list(test_json_list.keys())}")
                    if 'graphs' in test_json_list:
                        test_json_list = test_json_list['graphs']
                        print(f"    Using 'graphs' key, now {len(test_json_list)} samples")
                    elif 'data' in test_json_list:
                        test_json_list = test_json_list['data']
                        print(f"    Using 'data' key, now {len(test_json_list)} samples")
                
                processed_count = 0
                for i, g_data in tqdm(enumerate(test_json_list), total=len(test_json_list),
                                      desc=f"  {dataset_name} Test Data"):
                    if not isinstance(g_data, dict):
                        print(f"    Warning: Item {i} is not dict, type: {type(g_data)}")
                        continue
                    g_data['_source_dataset'] = dataset_name
                    processed_graph = GraphDataset.dictToGraphObject(
                        g_data, is_test_set=True, graph_idx_info=f"{dataset_name}test{i}"
                    )
                    dataset_splits['test'].append(processed_graph)
                    processed_count += 1
                    
                print(f"    Successfully processed {processed_count} test graphs")
                    
            except json.JSONDecodeError as e:
                print(f"  ERROR: JSON decode failed for {actual_test_json_file}: {e}")
            except Exception as e:
                print(f"  ERROR loading {actual_test_json_file}: {e}. Skipping.")
        else:
            print(f"  No test JSON file found for dataset {dataset_name}")

        print(f"  Dataset {dataset_name} - Train: {len(dataset_splits['train'])}, "
              f"Val: {len(dataset_splits['val'])}, Test: {len(dataset_splits['test'])}")

        return dataset_splits

    def get_data_splits(self, force_reprocess=False):
        """
        Load and process graph datasets, creating separate train/val/test splits.
        
        Args:
            force_reprocess: If True, reprocess data even if cached files exist
            
        Returns:
            dict: Contains separate splits for each dataset
        """
        # Define paths for processed files
        processed_paths = {}
        for ds_name in self.dataset_names:
            processed_paths[ds_name] = {
                'train': os.path.join(self.processed_dir, f'{ds_name}_train_graphs.pt'),
                'val': os.path.join(self.processed_dir, f'{ds_name}_val_graphs.pt'),
                'test': os.path.join(self.processed_dir, f'{ds_name}_test_graphs.pt')
            }

        # Check if all pre-processed files exist
        all_files_exist = all(
            os.path.exists(processed_paths[ds_name]['train']) and
            os.path.exists(processed_paths[ds_name]['val']) and
            os.path.exists(processed_paths[ds_name]['test'])
            for ds_name in self.dataset_names
        )

        if not force_reprocess and all_files_exist:
            print("Loading pre-processed data...")
            loaded_data = {}
            for ds_name in self.dataset_names:
                loaded_data[ds_name] = {
                    'train': torch.load(processed_paths[ds_name]['train'], weights_only=False),
                    'val': torch.load(processed_paths[ds_name]['val'], weights_only=False),
                    'test': torch.load(processed_paths[ds_name]['test'], weights_only=False)
                }
            self.print_split_summary(loaded_data)
            return loaded_data

        print("Processing data from JSONs in original_dataset/...")

        # Process each dataset separately to minimize memory usage
        all_splits = {}
        for ds_name in self.dataset_names:
            print(f"Processing dataset {ds_name}...")

            # Process single dataset
            dataset_splits = self.process_single_dataset(ds_name)

            # Save immediately to disk (including empty lists)
            print(f"Saving dataset {ds_name} to disk...")
            torch.save(dataset_splits['train'], processed_paths[ds_name]['train'])
            torch.save(dataset_splits['val'], processed_paths[ds_name]['val'])
            torch.save(dataset_splits['test'], processed_paths[ds_name]['test'])

            # Store for summary
            all_splits[ds_name] = {
                'train': len(dataset_splits['train']),
                'val': len(dataset_splits['val']),
                'test': len(dataset_splits['test'])
            }

            # Clear from memory
            del dataset_splits
            gc.collect()
            print(f"Dataset {ds_name} processed and saved. Memory cleared.")

        # Load all datasets back for return
        print("Loading all processed data for return...")
        final_data = {}
        for ds_name in self.dataset_names:
            final_data[ds_name] = {
                'train': torch.load(processed_paths[ds_name]['train'], weights_only=False),
                'val': torch.load(processed_paths[ds_name]['val'], weights_only=False),
                'test': torch.load(processed_paths[ds_name]['test'], weights_only=False)
            }

        self.print_split_summary(final_data)
        return final_data

    def print_split_summary(self, splits_dict):
        """Print comprehensive summary of dataset splits."""
        print("\n--- Data Split Summary (Separate Datasets) ---")
        total_train, total_val, total_test = 0, 0, 0

        for ds_name in self.dataset_names:
            if ds_name in splits_dict:
                train_count = len(splits_dict[ds_name]['train'])
                val_count = len(splits_dict[ds_name]['val'])
                test_count = len(splits_dict[ds_name]['test'])

                print(f"Dataset {ds_name}:")
                print(f"  Train: {train_count}")
                print(f"  Val:   {val_count}")
                print(f"  Test:  {test_count}")
                print(f"  Total: {train_count + val_count + test_count}")

                total_train += train_count
                total_val += val_count
                total_test += test_count
            else:
                print(f"Dataset {ds_name}: No data found")

        print(f"\nOverall Totals:")
        print(f"  Total Train: {total_train}")
        print(f"  Total Val:   {total_val}")
        print(f"  Total Test:  {total_test}")
        print(f"  Grand Total: {total_train + total_val + total_test}")
        print("--------------------------------------------\n")


# Legacy function for backward compatibility
def dictToGraphObject(graph_dict):
    """Legacy function - use GraphDataset.dictToGraphObject instead."""
    return GraphDataset.dictToGraphObject(graph_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enhanced Graph Data Loader with Multi-Dataset Support")
    parser.add_argument('--force_reprocess_data', action='store_true', 
                       help="Force re-processing of all data")
    parser.add_argument('--val_split_ratio', type=float, default=0.1, 
                       help="Validation split ratio (default: 0.1)")
    parser.add_argument('--dataset_path', type=str, default=None,
                       help="Path to dataset directory (optional)")
    args = parser.parse_args()

    # Initialize multi-dataset loader
    loader = MultiDatasetLoader(base_path=args.dataset_path, val_split_ratio=args.val_split_ratio)
    
    # Load data splits
    print("Loading separate data splits...")
    all_splits = loader.get_data_splits(force_reprocess=args.force_reprocess_data)

    # Show examples from each dataset
    for ds_name in loader.dataset_names:
        if ds_name in all_splits:
            print(f"\n=== Dataset {ds_name} Examples ===")

            # Show train example
            train_graphs = all_splits[ds_name]['train']
            if train_graphs:
                print(f"First training graph from dataset {ds_name}:")
                sg = train_graphs[0]
                print(sg)
                print(f"  Source Dataset: {getattr(sg, '_source_dataset', 'N/A')}")
                print(f"  x: {sg.x.shape}, {sg.x.dtype}")
                if hasattr(sg, 'rwse_pe') and sg.rwse_pe is not None:
                    print(f"  rwse_pe: {sg.rwse_pe.shape}")
                print(f"  edge_index: {sg.edge_index.shape}")
                if sg.edge_attr is not None:
                    print(f"  edge_attr: {sg.edge_attr.shape}")
                print(f"  y: {sg.y} | num_nodes: {sg.num_nodes}")

            # Show val example
            val_graphs = all_splits[ds_name]['val']
            if val_graphs:
                print(f"First validation graph from dataset {ds_name}:")
                sg = val_graphs[0]
                print(f"  y: {sg.y} | num_nodes: {sg.num_nodes}")

            # Show test example
            test_graphs = all_splits[ds_name]['test']
            if test_graphs:
                print(f"First test graph from dataset {ds_name}:")
                sg = test_graphs[0]
                print(f"  y (should be -1): {sg.y} | num_nodes: {sg.num_nodes}")
        else:
            print(f"\nDataset {ds_name}: No data available")