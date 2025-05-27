import torch
import glob
import os
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_datasets_from_folder(folder_path):
    """Load all datasets from folder structure"""
    datasets = {}

    # find all datasets letters (A, B, C, D)
    dataset_letters = set()
    for file in os.listdir(folder_path):
        if file.endswith('_train_graphs.pt'):
            letter = file.split('_')[0]
            dataset_letters.add(letter)

    for letter in sorted(dataset_letters):
        try:
            train_data = torch.load(f'{folder_path}/{letter}_train_graphs.pt', weights_only=False)
            val_data = torch.load(f'{folder_path}/{letter}_val_graphs.pt', weights_only=False)
            test_data = torch.load(f'{folder_path}/{letter}_test_graphs.pt', weights_only=False)
            
            # Keep each dataset separate
            datasets[letter] = {
                'train': train_data,
                'val': val_data, 
                'test': test_data
            }
            print(f"✅ Loaded dataset {letter}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        except Exception as e:
            print(f"❌ Failed to load dataset {letter}: {e}")
    return datasets

class BasicGraphClassifier(nn.Module):
    """Simplified version for initial testing"""
    
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, batch=None):
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        
        if batch is not None:
            graph_repr = global_mean_pool(h2, batch)
        else:
            graph_repr = h2.mean(dim=0, keepdim=True)
            
        return self.classifier(graph_repr)
    
def analyze_all_datasets(folder_path, device):
    """Analyze each dataset separately"""
    print("🚀 ANALYZING ALL DATASETS")
    
    datasets = load_datasets_from_folder(folder_path)
    all_results = {}
    
    for letter, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"ANALYZING DATASET {letter}")
        print(f"{'='*50}")
        
        # Analyze train split (or combine train+val if needed)
        dataset_to_analyze = data['train']  # Keep as list
        
        # Add required attributes
        dataset_to_analyze.num_classes = 6
        dataset_to_analyze.num_node_features = 1
        
        results = run_phase1_analysis(dataset_to_analyze, device)
        all_results[letter] = results
        
        print(f"📊 SPLIT SIZES:")
        print(f"   Train: {len(data['train'])}")
        print(f"   Val: {len(data['val'])}")
        print(f"   Test: {len(data['test'])}")
    
    return all_results

class DatasetAnalyzer:
    """Tool for understanding your dataset characteristics"""
    
    def __init__(self):
        self.label_distribution = {}
        self.graph_sizes = []
        self.node_feature_stats = {}
        
    def analyze_dataset(self, dataset):
        """Comprehensive dataset analysis"""
        print("=== DATASET ANALYSIS ===")
        
        # Basic statistics
        print(f"Total graphs: {len(dataset)}")
        print(f"Number of classes: {dataset.num_classes}")
        print(f"Node feature dimension: {dataset.num_node_features}")
        
        # Label distribution analysis
        labels = [data.y.item() for data in dataset]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        print("\n=== LABEL DISTRIBUTION ===")
        for label, count in zip(unique_labels, counts):
            percentage = count / len(labels) * 100
            print(f"Class {label}: {count} samples ({percentage:.1f}%)")
            
        # Check for severe class imbalance
        imbalance_ratio = max(counts) / min(counts)
        if imbalance_ratio > 10:
            print(f"⚠️  WARNING: Severe class imbalance detected (ratio: {imbalance_ratio:.1f})")
            print("   Consider using weighted loss or resampling techniques")
        
        # Graph size analysis
        graph_sizes = []
        node_features_samples = []
        
        for i, data in enumerate(dataset[:min(100, len(dataset))]):  # Sample first 100
            graph_sizes.append(data.x.size(0))  # Number of nodes
            if i < 10:  # Collect features from first 10 graphs
                node_features_samples.append(data.x)
        
        print(f"\n=== GRAPH STRUCTURE ===")
        print(f"Average nodes per graph: {np.mean(graph_sizes):.1f}")
        print(f"Min nodes: {min(graph_sizes)}, Max nodes: {max(graph_sizes)}")
        print(f"Std deviation: {np.std(graph_sizes):.1f}")
        
        # Node feature analysis
        if node_features_samples:
            all_features = torch.cat(node_features_samples, dim=0)
            print(f"\n=== NODE FEATURES ===")
            print(f"Feature range: [{all_features.min():.3f}, {all_features.max():.3f}]")
            print(f"Feature mean: {all_features.mean():.3f}")
            print(f"Feature std: {all_features.std():.3f}")
            
            # Check for potential normalization needs
            if all_features.std() > 10 or all_features.mean().abs() > 10:
                print("⚠️  WARNING: Features might benefit from normalization")
        
        return {
            'label_distribution': dict(zip(unique_labels, counts)),
            'graph_sizes': graph_sizes,
            'imbalance_ratio': imbalance_ratio,
            'feature_stats': {
                'min': all_features.min().item() if node_features_samples else None,
                'max': all_features.max().item() if node_features_samples else None,
                'mean': all_features.mean().item() if node_features_samples else None,
                'std': all_features.std().item() if node_features_samples else None
            }
        }

def test_basic_functionality(dataset, device):
    """Test 1: Ensure basic model can train without errors"""
    print("\n=== TESTING BASIC FUNCTIONALITY ===")
    
    # Create train/test split
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = BasicGraphClassifier(
        input_dim=dataset.num_node_features,
        hidden_dim=64,
        num_classes=dataset.num_classes
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Test forward pass
    try:
        for batch in train_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y)
            print(f"✅ Forward pass successful. Output shape: {output.shape}")
            print(f"✅ Loss computation successful. Loss: {loss.item():.4f}")
            break
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test backward pass
    try:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("✅ Backward pass successful")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False
    
    # Quick training test (5 epochs)
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/5, Avg Loss: {avg_loss:.4f}")
    
    # Test evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            pred = output.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    accuracy = correct / total
    print(f"✅ Basic model test accuracy: {accuracy:.4f}")
    
    if accuracy > 0.1:  # Better than random for most datasets
        print("✅ Basic functionality test PASSED")
        return True
    else:
        print("⚠️  Low accuracy - check dataset or model implementation")
        return False

def detect_potential_noise_patterns(dataset, model, device):
    """Test 2: Analyze potential noise patterns in the dataset"""
    print("\n=== DETECTING POTENTIAL NOISE PATTERNS ===")
    
    # Train a simple model to get predictions
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in train_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            probs = F.softmax(output, dim=1)
            
            pred = output.argmax(dim=1)
            confidence = probs.max(dim=1)[0]
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    
    # Analyze prediction confidence distribution
    print(f"Average prediction confidence: {np.mean(all_confidences):.3f}")
    print(f"Confidence std deviation: {np.std(all_confidences):.3f}")
    
    # Identify low-confidence samples (potential noise)
    low_conf_threshold = np.percentile(all_confidences, 20)  # Bottom 20%
    low_conf_mask = all_confidences < low_conf_threshold
    
    print(f"Low confidence samples (< {low_conf_threshold:.3f}): {low_conf_mask.sum()}")
    
    # Check if low-confidence samples have different label distribution
    low_conf_labels = all_labels[low_conf_mask]
    high_conf_labels = all_labels[~low_conf_mask]
    
    print("\nLabel distribution in low-confidence samples:")
    unique_low, counts_low = np.unique(low_conf_labels, return_counts=True)
    for label, count in zip(unique_low, counts_low):
        percentage = count / len(low_conf_labels) * 100
        print(f"  Class {label}: {percentage:.1f}%")
    
    # Look for disagreement patterns
    disagreement_mask = all_predictions != all_labels
    disagreement_rate = disagreement_mask.mean()
    
    print(f"\nOverall disagreement rate: {disagreement_rate:.3f}")
    
    if disagreement_rate > 0.3:
        print("⚠️  HIGH disagreement rate - possible noisy labels or difficult dataset")
    elif disagreement_rate > 0.15:
        print("⚠️  MODERATE disagreement rate - some noise likely present")
    else:
        print("✅ LOW disagreement rate - dataset appears relatively clean")
    
    return {
        'avg_confidence': np.mean(all_confidences),
        'disagreement_rate': disagreement_rate,
        'low_confidence_samples': low_conf_mask.sum(),
        'potential_noise_indicators': disagreement_rate > 0.2
    }

# Usage example for Phase 1
def run_phase1_analysis(dataset, device):
    """Complete Phase 1 analysis pipeline"""
    print("🚀 STARTING PHASE 1: FOUNDATION TESTING")
    
    # Step 1: Dataset analysis
    analyzer = DatasetAnalyzer()
    dataset_stats = analyzer.analyze_dataset(dataset)
    
    # Step 2: Basic functionality test
    basic_test_passed = test_basic_functionality(dataset, device)
    
    if not basic_test_passed:
        print("❌ Phase 1 FAILED - fix basic issues before proceeding")
        return None
    
    # Step 3: Noise pattern detection (reuse the trained model)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = BasicGraphClassifier(
        input_dim=dataset.num_node_features,
        hidden_dim=64,
        num_classes=dataset.num_classes
    ).to(device)
    
    # Quick training for noise analysis
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):  # Quick training
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
    
    noise_analysis = detect_potential_noise_patterns(dataset, model, device)
    
    print("\n✅ PHASE 1 COMPLETED SUCCESSFULLY")
    print("📋 SUMMARY:")
    print(f"   - Dataset size: {len(dataset)}")
    print(f"   - Classes: {dataset.num_classes}")
    print(f"   - Potential noise detected: {'Yes' if noise_analysis['potential_noise_indicators'] else 'No'}")
    print(f"   - Disagreement rate: {noise_analysis['disagreement_rate']:.3f}")
    
    return {
        'dataset_stats': dataset_stats,
        'noise_analysis': noise_analysis,
        'ready_for_phase2': True
    }

# At the end of datasetTesting.py
folder_path = 'C:/Users/leosa/Documents/università/Sapienza/Deep Learning/progetto/DeepLearningChallenge/processed_data_separate'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = analyze_all_datasets(folder_path, device)