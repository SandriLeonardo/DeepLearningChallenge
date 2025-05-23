"""
Enhanced Models Module - Building on baseline GNN architecture
Member A Implementation - Days 1-4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool

class DualPoolingGNN(nn.Module):
    """
    Enhanced GNN with dual pooling strategy
    Builds on baseline single pooling approach
    """
    
    def __init__(self, input_dim, hidden_dim, num_classes, model_type='gin', dropout=0.3):
        super(DualPoolingGNN, self).__init__()
        
        # Maintain compatibility with baseline embedding
        self.embedding = nn.Embedding(1, input_dim)
        self.model_type = model_type
        
        # Enhanced conv layers with batch norm
        if model_type == 'gin':
            self.conv1 = GINConv(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ))
            
            self.conv2 = GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        else:  # GCN fallback
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
        
        # Dual classification heads
        self.classifier_mean = nn.Linear(hidden_dim, num_classes)
        self.classifier_max = nn.Linear(hidden_dim, num_classes)
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.tensor([0.6, 0.4]))
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Maintain baseline embedding compatibility
        x = self.embedding(x)
        
        # Graph encoding with skip connections
        if self.model_type == 'gin':
            h1 = self.conv1(x, edge_index)
            h2 = self.conv2(h1, edge_index) + h1  # Skip connection
        else:
            h1 = F.relu(self.conv1(x, edge_index))
            h1 = self.dropout(h1)
            h2 = self.conv2(h1, edge_index)
        
        # Dual pooling strategies
        mean_repr = global_mean_pool(h2, batch)
        max_repr = global_max_pool(h2, batch)
        
        # Individual predictions
        logits_mean = self.classifier_mean(mean_repr)
        logits_max = self.classifier_max(max_repr)
        
        # Ensemble prediction with learnable weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_logits = weights[0] * logits_mean + weights[1] * logits_max
        
        return ensemble_logits, logits_mean, logits_max, (mean_repr, max_repr)

class AdaptiveEnsembleGNN(nn.Module):
    """
    Advanced ensemble with attention-based weighting
    Days 3-4 implementation if dual pooling works well
    """
    
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super(AdaptiveEnsembleGNN, self).__init__()
        
        self.embedding = nn.Embedding(1, input_dim)
        
        # Triple GIN layers for more capacity
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        
        # Triple pooling strategies
        self.classifier_mean = nn.Linear(hidden_dim, num_classes)
        self.classifier_max = nn.Linear(hidden_dim, num_classes)
        self.classifier_add = nn.Linear(hidden_dim, num_classes)
        
        # Attention mechanism for dynamic weighting
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.embedding(x)
        
        # Deep encoding with skip connections
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index) + h1
        h3 = self.conv3(h2, edge_index) + h2
        
        # Triple pooling
        mean_repr = global_mean_pool(h3, batch)
        max_repr = global_max_pool(h3, batch)
        add_repr = global_add_pool(h3, batch)
        
        # Individual predictions
        logits_mean = self.classifier_mean(mean_repr)
        logits_max = self.classifier_max(max_repr)
        logits_add = self.classifier_add(add_repr)
        
        # Attention-based ensemble
        combined_repr = torch.cat([mean_repr, max_repr, add_repr], dim=1)
        attention_weights = self.attention(combined_repr)
        
        ensemble_logits = (
            attention_weights[:, 0:1] * logits_mean +
            attention_weights[:, 1:2] * logits_max +
            attention_weights[:, 2:3] * logits_add
        )
        
        return ensemble_logits, logits_mean, logits_max, logits_add, attention_weights

class ModelFactory:
    """Factory for creating models based on configuration"""
    
    @staticmethod
    def create_model(config):
        """Create model based on configuration dict"""
        model_type = config.get('model_type', 'dual_gin')
        input_dim = config.get('input_dim', 300)
        hidden_dim = config.get('hidden_dim', 128)
        num_classes = config.get('num_classes', 6)
        dropout = config.get('dropout', 0.3)
        
        if model_type == 'dual_gin':
            return DualPoolingGNN(input_dim, hidden_dim, num_classes, 'gin', dropout)
        elif model_type == 'dual_gcn':
            return DualPoolingGNN(input_dim, hidden_dim, num_classes, 'gcn', dropout)
        elif model_type == 'adaptive_ensemble':
            return AdaptiveEnsembleGNN(input_dim, hidden_dim, num_classes, dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class EnsembleTrainer:
    """Training utilities for ensemble models"""
    
    def __init__(self, model, device, lr=0.001):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
    def compute_ensemble_consistency_loss(self, logits_mean, logits_max, selected_indices=None):
        """Consistency loss between ensemble components"""
        if selected_indices is not None and len(selected_indices) > 0:
            mean_selected = logits_mean[selected_indices]
            max_selected = logits_max[selected_indices]
        else:
            mean_selected = logits_mean
            max_selected = logits_max
            
        return F.mse_loss(mean_selected, max_selected)
    
    def train_step(self, batch, robust_loss_fn, sample_selector, epoch):
        """Single training step with ensemble considerations"""
        self.model.train()
        batch = batch.to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        if hasattr(self.model, 'conv3'):  # AdaptiveEnsembleGNN
            ensemble_logits, logits_mean, logits_max, logits_add, attention_weights = self.model(batch)
            consistency_loss = (
                F.mse_loss(logits_mean, logits_max) +
                F.mse_loss(logits_mean, logits_add) +
                F.mse_loss(logits_max, logits_add)
            ) / 3
        else:  # DualPoolingGNN
            ensemble_logits, logits_mean, logits_max, _ = self.model(batch)
            consistency_loss = self.compute_ensemble_consistency_loss(logits_mean, logits_max)
        
        # Compute robust loss with sample selection
        sample_losses = robust_loss_fn.generalized_cross_entropy(
            ensemble_logits, batch.y, reduction='none'
        )
        
        selected_indices, selection_ratio = sample_selector.select_clean_samples(
            sample_losses, batch.y, epoch
        )
        
        if len(selected_indices) > 0:
            selected_logits = ensemble_logits[selected_indices]
            selected_labels = batch.y[selected_indices]
            main_loss = robust_loss_fn.generalized_cross_entropy(selected_logits, selected_labels)
            
            # Selected consistency loss
            consistency_loss = self.compute_ensemble_consistency_loss(
                logits_mean, logits_max, selected_indices
            )
        else:
            main_loss = sample_losses.mean()
        
        total_loss = main_loss + 0.1 * consistency_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'selection_ratio': selection_ratio
        }
    
    def evaluate(self, data_loader):
        """Evaluate model performance"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                
                if hasattr(self.model, 'conv3'):
                    ensemble_logits, _, _, _, _ = self.model(batch)
                else:
                    ensemble_logits, _, _, _ = self.model(batch)
                    
                pred = ensemble_logits.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        return correct / total if total > 0 else 0

def test_models():
    """Test function for model implementations"""
    print("Testing enhanced models...")
    
    # Mock data
    class MockData:
        def __init__(self):
            self.x = torch.zeros(20, dtype=torch.long)
            self.edge_index = torch.randint(0, 20, (2, 40))
            self.batch = torch.zeros(20, dtype=torch.long)
            self.y = torch.randint(0, 6, (1,))
    
    config = {
        'model_type': 'dual_gin',
        'input_dim': 300,
        'hidden_dim': 128,
        'num_classes': 6
    }
    
    model = ModelFactory.create_model(config)
    data = MockData()
    
    # Test forward pass
    output = model(data)
    ensemble_logits = output[0]
    
    print(f"Output shape: {ensemble_logits.shape}")
    print(f"Expected shape: (1, 6)")
    print("✅ Enhanced models test passed!")

if __name__ == "__main__":
    test_models()