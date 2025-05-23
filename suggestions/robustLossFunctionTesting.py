import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader

class RobustLossCollection:
    """Collection of robust loss functions with detailed explanations"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.loss_history = {'ce': [], 'gce': [], 'sce': [], 'mae': []}
        
    def cross_entropy_loss(self, logits, labels):
        """Standard cross-entropy for comparison"""
        return F.cross_entropy(logits, labels, reduction='none')
    
    def generalized_cross_entropy(self, logits, labels, q=0.7):
        """
        Generalized Cross Entropy Loss
        
        Key insight: Standard CE becomes infinite when p_true → 0
        GCE flattens this penalty, making it more forgiving of uncertain predictions
        
        Mathematical form: L_q = (1 - p_true^q) / q
        - When q → 0: reduces to standard CE
        - When q → 1: reduces to MAE (completely noise-robust)
        - Sweet spot q=0.7: good balance between fitting and robustness
        """
        probs = F.softmax(logits, dim=1)
        
        # Gather probabilities for true labels
        # This is crucial: we're looking at how confident the model is about the TRUE label
        true_probs = probs.gather(1, labels.view(-1, 1)).squeeze()
        
        # Add small epsilon to prevent numerical issues
        true_probs = torch.clamp(true_probs, min=1e-8, max=1.0)
        
        if q == 0:
            # Equivalent to standard cross-entropy
            return -torch.log(true_probs)
        else:
            # GCE formula
            loss = (1 - torch.pow(true_probs, q)) / q
            return loss
    
    def symmetric_cross_entropy(self, logits, labels, alpha=0.1, beta=1.0):
        """
        Symmetric Cross Entropy Loss
        
        Key insight: If labels are clean, both directions should be consistent:
        - Forward: How well do model predictions match true labels?
        - Reverse: How well do true labels predict model outputs?
        
        If labels are noisy, the reverse direction will have high loss
        """
        # Forward direction: standard cross-entropy
        ce_forward = F.cross_entropy(logits, labels, reduction='none')
        
        # Reverse direction: how well do labels predict model outputs?
        probs = F.softmax(logits, dim=1)
        
        # Create one-hot encoding of labels
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Reverse cross-entropy: -sum(p_model * log(p_label))
        # We treat the one-hot labels as a "prediction" and probs as "ground truth"
        probs_clamped = torch.clamp(probs, min=1e-8, max=1.0)
        ce_reverse = -torch.sum(probs_clamped * torch.log(one_hot + 1e-8), dim=1)
        
        # Combine both directions
        return alpha * ce_forward + beta * ce_reverse
    
    def mean_absolute_error(self, logits, labels):
        """
        MAE Loss for classification
        
        Key insight: MAE is completely robust to symmetric label noise
        But it's less expressive than cross-entropy for clean data
        """
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # L1 distance between predicted and true distributions
        return torch.sum(torch.abs(probs - one_hot), dim=1)

class LossComparator:
    """Tool for comparing different loss functions on your dataset"""
    
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.loss_functions = RobustLossCollection(dataset.num_classes)
        
    def compare_loss_behaviors(self, num_samples=100):
        """
        Compare how different loss functions behave on your dataset
        This helps you understand which loss is most appropriate
        """
        print("\n=== COMPARING LOSS FUNCTION BEHAVIORS ===")
        
        # Get a sample of data
        loader = DataLoader(self.dataset, batch_size=min(num_samples, len(self.dataset)), shuffle=True)
        
        self.model.eval()
        
        for batch in loader:
            batch = batch.to(self.device)
            
            with torch.no_grad():
                logits = self.model(batch.x, batch.edge_index, batch.batch)
                
                # Compute all losses
                ce_losses = self.loss_functions.cross_entropy_loss(logits, batch.y)
                gce_losses = self.loss_functions.generalized_cross_entropy(logits, batch.y)
                sce_losses = self.loss_functions.symmetric_cross_entropy(logits, batch.y)
                mae_losses = self.loss_functions.mean_absolute_error(logits, batch.y)
                
                # Convert to numpy for analysis
                ce_losses = ce_losses.cpu().numpy()
                gce_losses = gce_losses.cpu().numpy()
                sce_losses = sce_losses.cpu().numpy()
                mae_losses = mae_losses.cpu().numpy()
                
                # Calculate statistics
                print(f"Cross Entropy    - Mean: {ce_losses.mean():.4f}, Std: {ce_losses.std():.4f}")
                print(f"GCE (q=0.7)      - Mean: {gce_losses.mean():.4f}, Std: {gce_losses.std():.4f}")
                print(f"Symmetric CE     - Mean: {sce_losses.mean():.4f}, Std: {sce_losses.std():.4f}")
                print(f"MAE              - Mean: {mae_losses.mean():.4f}, Std: {mae_losses.std():.4f}")
                
                # Identify outliers (potential noisy samples)
                ce_outliers = ce_losses > np.percentile(ce_losses, 90)
                gce_outliers = gce_losses > np.percentile(gce_losses, 90)
                
                outlier_difference = np.sum(ce_outliers) - np.sum(gce_outliers)
                
                print(f"\nOutlier Analysis:")
                print(f"CE outliers: {np.sum(ce_outliers)}")
                print(f"GCE outliers: {np.sum(gce_outliers)}")
                print(f"GCE reduces outliers by: {outlier_difference}")
                
                if outlier_difference > 0:
                    print("✅ GCE shows robustness benefit - good sign for noisy data")
                else:
                    print("⚠️  GCE doesn't show clear benefit - data might be clean")
                
                break  # Only analyze first batch
        
        return {
            'ce_mean': ce_losses.mean(),
            'gce_mean': gce_losses.mean(),
            'robustness_benefit': outlier_difference > 0
        }

class RobustModel(nn.Module):
    """Enhanced model with robust loss integration"""
    
    def __init__(self, input_dim, hidden_dim, num_classes, loss_type='gce'):
        super().__init__()
        
        # Same architecture as before, but now with loss awareness
        self.conv1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Crucial for stability
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.gin1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.gin2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Loss function setup
        self.loss_type = loss_type
        self.robust_losses = RobustLossCollection(num_classes)
        
        # For tracking training dynamics
        self.training_stats = {
            'epoch_losses': [],
            'confidence_evolution': [],
            'prediction_changes': []
        }
        
    def forward(self, x, edge_index, batch=None):
        """Forward pass with intermediate representations for analysis"""
        # Initial transformation
        h0 = self.conv1(x)
        
        # GIN layers with skip connections
        h1 = self.gin1(h0) + h0  # Skip connection for stability
        h2 = self.gin2(h1) + h1  # Another skip connection
        
        # Global pooling
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            graph_repr = global_mean_pool(h2, batch)
        else:
            graph_repr = h2.mean(dim=0, keepdim=True)
        
        # Classification
        logits = self.classifier(graph_repr)
        
        return logits, graph_repr  # Return both for analysis
    
    def compute_loss(self, logits, labels, epoch=0):
        """Compute loss based on selected robust loss function"""
        if self.loss_type == 'ce':
            loss = self.robust_losses.cross_entropy_loss(logits, labels).mean()
        elif self.loss_type == 'gce':
            loss = self.robust_losses.generalized_cross_entropy(logits, labels).mean()
        elif self.loss_type == 'sce':
            loss = self.robust_losses.symmetric_cross_entropy(logits, labels).mean()
        elif self.loss_type == 'mae':
            loss = self.robust_losses.mean_absolute_error(logits, labels).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Track training dynamics
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            confidence = probs.max(dim=1)[0].mean().item()
            self.training_stats['confidence_evolution'].append(confidence)
        
        return loss

def test_robust_loss_effectiveness(dataset, device, epochs=50):
    """
    Test different robust loss functions to see which works best
    This is your experimental validation phase
    """
    print("\n=== TESTING ROBUST LOSS EFFECTIVENESS ===")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Test different loss functions
    loss_types = ['ce', 'gce', 'sce', 'mae']
    results = {}
    
    for loss_type in loss_types:
        print(f"\n--- Testing {loss_type.upper()} ---")
        
        # Initialize model
        model = RobustModel(
            input_dim=dataset.num_node_features,
            hidden_dim=64,
            num_classes=dataset.num_classes,
            loss_type=loss_type
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training loop
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                logits, _ = model(batch.x, batch.edge_index, batch.batch)
                loss = model.compute_loss(logits, batch.y, epoch)
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            scheduler.step()
            train_losses.append(epoch_loss / len(train_loader))
            
            # Evaluation every 10 epochs
            if epoch % 10 == 0:
                model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        logits, _ = model(batch.x, batch.edge_index, batch.batch)
                        pred = logits.argmax(dim=1)
                        correct += (pred == batch.y).sum().item()
                        total += batch.y.size(0)
                
                accuracy = correct / total
                test_accuracies.append(accuracy)
                
                print(f"Epoch {epoch}: Loss {train_losses[-1]:.4f}, Accuracy {accuracy:.4f}")
        
        # Final evaluation
        model.eval()
        final_correct = 0
        final_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits, _ = model(batch.x, batch.edge_index, batch.batch)
                pred = logits.argmax(dim=1)
                final_correct += (pred == batch.y).sum().item()
                final_total += batch.y.size(0)
        
        final_accuracy = final_correct / final_total
        
        results[loss_type] = {
            'final_accuracy': final_accuracy,
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'confidence_evolution': model.training_stats['confidence_evolution']
        }
        
        print(f"Final {loss_type.upper()} accuracy: {final_accuracy:.4f}")
    
    # Compare results
    print("\n=== COMPARISON SUMMARY ===")
    best_loss = max(results.keys(), key=lambda k: results[k]['final_accuracy'])
    
    for loss_type in loss_types:
        acc = results[loss_type]['final_accuracy']
        improvement = acc - results['ce']['final_accuracy']
        print(f"{loss_type.upper()}: {acc:.4f} (improvement: {improvement:+.4f})")
    
    print(f"\n🏆 Best performing loss: {best_loss.upper()}")
    
    # Check for concerning patterns
    ce_acc = results['ce']['final_accuracy']
    if results['gce']['final_accuracy'] > ce_acc + 0.02:
        print("✅ GCE shows clear benefit - likely noisy labels present")
    elif results['sce']['final_accuracy'] > ce_acc + 0.02:
        print("✅ SCE shows clear benefit - asymmetric noise likely")
    else:
        print("⚠️  No clear robust loss benefit - dataset might be clean")
    
    return results, best_loss

# Critical Implementation Warnings and Best Practices
def implementation_warnings():
    """
    CRITICAL WARNINGS - READ BEFORE IMPLEMENTING
    These are the most common mistakes that can silently corrupt your results
    """
    warnings = """
    🚨 CRITICAL IMPLEMENTATION WARNINGS 🚨
    
    1. NUMERICAL STABILITY:
       - Always add epsilon (1e-8) to log operations
       - Clamp probabilities to [1e-8, 1.0] range
       - Use torch.clamp() instead of manual if-statements
       
    2. GRADIENT FLOW:
       - Use .detach() when computing statistics for sample selection
       - Don't accidentally break gradients in loss computation
       - Monitor gradient norms - they should be reasonable (< 10)
       
    3. BATCH NORMALIZATION:
       - BatchNorm behaves differently in train vs eval mode
       - Always call model.train() and model.eval() appropriately
       - BatchNorm can interact poorly with very small batch sizes
       
    4. LOSS REDUCTION:
       - Use reduction='none' for sample-wise losses, then manually reduce
       - This allows you to analyze individual sample losses
       - Don't use reduction='mean' when you need sample selection
       
    5. DEVICE CONSISTENCY:
       - Always move data to device before operations
       - Check that all tensors are on the same device
       - Use .to(device) consistently throughout
       
    6. LABEL ENCODING:
       - Ensure labels are 0-indexed integers
       - Check that num_classes matches actual number of unique labels
       - Watch for off-by-one errors in label ranges
       
    7. EVALUATION MODE:
       - Always use model.eval() and torch.no_grad() during evaluation
       - Dropout and BatchNorm behave differently in eval mode
       - This affects your noise detection accuracy
    """
    print(warnings)

# Phase 2 main execution function
def run_phase2_robust_loss(dataset, device, phase1_results):
    """Complete Phase 2: Robust Loss Implementation"""
    print("🚀 STARTING PHASE 2: ROBUST LOSS IMPLEMENTATION")
    
    # Show critical warnings
    implementation_warnings()
    
    # Test loss function effectiveness
    results, best_loss = test_robust_loss_effectiveness(dataset, device, epochs=50)
    
    # Create final recommendation
    print(f"\n📋 PHASE 2 SUMMARY:")
    print(f"   - Best performing loss function: {best_loss.upper()}")
    print(f"   - Performance improvement: {results[best_loss]['final_accuracy'] - results['ce']['final_accuracy']:+.4f}")
    
    # Prepare for Phase 3
    recommendation = {
        'best_loss_function': best_loss,
        'performance_improvement': results[best_loss]['final_accuracy'] - results['ce']['final_accuracy'],
        'all_results': results,
        'ready_for_phase3': True
    }
    
    return recommendation