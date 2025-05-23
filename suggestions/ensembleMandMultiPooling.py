import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool, global_add_pool
import numpy as np

class MultiPoolingGraphEncoder(nn.Module):
    """
    Multi-pooling strategy for robust graph representation
    
    Key insight: Different pooling methods capture different aspects:
    - Mean pooling: Average behavior across all nodes
    - Max pooling: Most prominent/extreme features  
    - Add pooling: Total accumulated features
    
    When labels are noisy, having multiple perspectives helps triangulate
    the true signal, similar to how multiple witnesses provide better evidence
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Create multiple GIN layers with skip connections
        self.gin_layers = nn.ModuleList()
        
        # First layer: input transformation
        self.gin_layers.append(GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )))
        
        # Subsequent layers with residual connections
        for i in range(num_layers - 1):
            self.gin_layers.append(GINConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim)
            )))
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass with skip connections and multiple pooling
        
        Skip connections are crucial for noisy label scenarios because:
        1. They prevent gradient degradation
        2. They allow lower-level features to influence final predictions
        3. They provide multiple paths for information flow
        """
        # Store intermediate representations for skip connections
        hidden_states = [x]
        
        current_h = x
        
        for i, (gin_layer, layer_norm) in enumerate(zip(self.gin_layers, self.layer_norms)):
            # GIN transformation
            new_h = gin_layer(current_h, edge_index)
            
            # Apply layer normalization
            new_h = layer_norm(new_h)
            
            # Skip connection (if dimensions match)
            if current_h.size(-1) == new_h.size(-1):
                new_h = new_h + current_h  # Residual connection
            
            current_h = new_h
            hidden_states.append(current_h)
        
        # Multiple pooling strategies
        if batch is not None:
            # Mean pooling: captures average node behavior
            mean_pool = global_mean_pool(current_h, batch)
            
            # Max pooling: captures most extreme features
            max_pool = global_max_pool(current_h, batch)
            
            # Add pooling: captures total accumulated features
            add_pool = global_add_pool(current_h, batch)
        else:
            # Single graph case (for testing)
            mean_pool = current_h.mean(dim=0, keepdim=True)
            max_pool = current_h.max(dim=0, keepdim=True)[0]
            add_pool = current_h.sum(dim=0, keepdim=True)
        
        return {
            'mean_pool': mean_pool,
            'max_pool': max_pool, 
            'add_pool': add_pool,
            'final_node_features': current_h,
            'all_hidden_states': hidden_states
        }

class EnsembleClassifier(nn.Module):
    """
    Ensemble classifier that combines multiple pooling strategies
    
    This is where the magic happens for noise robustness:
    - Each pooling method sees the data differently
    - Ensemble weighting learns which view is most reliable
    - Disagreement between methods can indicate noisy samples
    """
    
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        
        # Separate classification heads for each pooling method
        self.mean_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.max_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.add_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Learnable ensemble weights
        # These start equal but adapt based on which pooling method is most reliable
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Attention mechanism for dynamic weighting
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)
        )
        
    def forward(self, pooled_features, use_attention=True):
        """
        Forward pass with ensemble prediction
        
        Args:
            pooled_features: Dictionary with 'mean_pool', 'max_pool', 'add_pool'
            use_attention: Whether to use attention-based weighting
        """
        mean_pool = pooled_features['mean_pool']
        max_pool = pooled_features['max_pool']
        add_pool = pooled_features['add_pool']
        
        # Get predictions from each classifier
        mean_logits = self.mean_classifier(mean_pool)
        max_logits = self.max_classifier(max_pool)
        add_logits = self.add_classifier(add_pool)
        
        if use_attention:
            # Dynamic weighting based on current input
            combined_features = torch.cat([mean_pool, max_pool, add_pool], dim=1)
            attention_weights = self.attention_net(combined_features)
            
            # Weighted ensemble
            ensemble_logits = (
                attention_weights[:, 0:1] * mean_logits +
                attention_weights[:, 1:2] * max_logits +
                attention_weights[:, 2:3] * add_logits
            )
        else:
            # Static weighting using learnable parameters
            weights = F.softmax(self.ensemble_weights, dim=0)
            ensemble_logits = (
                weights[0] * mean_logits +
                weights[1] * max_logits +
                weights[2] * add_logits
            )
        
        return {
            'ensemble_logits': ensemble_logits,
            'mean_logits': mean_logits,
            'max_logits': max_logits,
            'add_logits': add_logits,
            'attention_weights': attention_weights if use_attention else weights
        }

class RobustEnsembleModel(nn.Module):
    """Complete robust ensemble model"""
    
    def __init__(self, input_dim, hidden_dim, num_classes, loss_type='gce'):
        super().__init__()
        
        self.encoder = MultiPoolingGraphEncoder(input_dim, hidden_dim)
        self.classifier = EnsembleClassifier(hidden_dim, num_classes)
        self.loss_type = loss_type
        self.num_classes = num_classes
        
        # For tracking ensemble agreement (key for noise detection)
        self.agreement_history = []
        
    def forward(self, x, edge_index, batch=None, return_all=False):
        """Forward pass with optional detailed outputs"""
        # Encode graph with multiple pooling
        pooled_features = self.encoder(x, edge_index, batch)
        
        # Classify with ensemble
        classifier_outputs = self.classifier(pooled_features)
        
        if return_all:
            return classifier_outputs, pooled_features
        else:
            return classifier_outputs['ensemble_logits']
    
    def compute_ensemble_agreement(self, classifier_outputs):
        """
        Measure agreement between different ensemble components
        
        High agreement = confident prediction
        Low agreement = uncertain prediction (potential noise)
        """
        mean_probs = F.softmax(classifier_outputs['mean_logits'], dim=1)
        max_probs = F.softmax(classifier_outputs['max_logits'], dim=1)
        add_probs = F.softmax(classifier_outputs['add_logits'], dim=1)
        
        # Calculate pairwise KL divergences
        kl_mean_max = F.kl_div(
            F.log_softmax(classifier_outputs['mean_logits'], dim=1),
            max_probs,
            reduction='none'
        ).sum(dim=1)
        
        kl_mean_add = F.kl_div(
            F.log_softmax(classifier_outputs['mean_logits'], dim=1),
            add_probs,
            reduction='none'
        ).sum(dim=1)
        
        kl_max_add = F.kl_div(
            F.log_softmax(classifier_outputs['max_logits'], dim=1),
            add_probs,
            reduction='none'
        ).sum(dim=1)
        
        # Average disagreement (lower = more agreement)
        avg_disagreement = (kl_mean_max + kl_mean_add + kl_max_add) / 3
        
        return avg_disagreement

class EnsembleTrainer:
    """Specialized trainer for ensemble models with noise detection"""
    
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )
        
        # For tracking training dynamics
        self.training_history = {
            'losses': [],
            'agreements': [],
            'attention_weights': []
        }
        
    def train_epoch(self, train_loader, epoch, warmup_epochs=10):
        """Training with ensemble-specific considerations"""
        self.model.train()
        epoch_loss = 0
        epoch_agreements = []
        epoch_attention_weights = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass with all outputs
            classifier_outputs, pooled_features = self.model(
                batch.x, batch.edge_index, batch.batch, return_all=True
            )
            
            # Compute primary loss
            ensemble_logits = classifier_outputs['ensemble_logits']
            
            if self.model.loss_type == 'gce':
                loss = self.compute_gce_loss(ensemble_logits, batch.y)
            else:
                loss = F.cross_entropy(ensemble_logits, batch.y)
            
            # Add consistency regularization between ensemble components
            consistency_loss = self.compute_consistency_loss(classifier_outputs)
            
            # Total loss
            total_loss = loss + 0.1 * consistency_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track statistics
            epoch_loss += total_loss.item()
            
            with torch.no_grad():
                # Measure ensemble agreement
                agreements = self.model.compute_ensemble_agreement(classifier_outputs)
                epoch_agreements.append(agreements.mean().item())
                
                # Track attention weights
                if 'attention_weights' in classifier_outputs:
                    epoch_attention_weights.append(
                        classifier_outputs['attention_weights'].mean(dim=0).cpu()
                    )
        
        self.scheduler.step()
        
        # Store epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        avg_agreement = np.mean(epoch_agreements)
        
        self.training_history['losses'].append(avg_loss)
        self.training_history['agreements'].append(avg_agreement)
        
        if epoch_attention_weights:
            avg_attention = torch.stack(epoch_attention_weights).mean(dim=0)
            self.training_history['attention_weights'].append(avg_attention)
        
        return {
            'loss': avg_loss,
            'agreement': avg_agreement,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def compute_gce_loss(self, logits, labels, q=0.7):
        """Generalized Cross Entropy loss"""
        probs = F.softmax(logits, dim=1)
        true_probs = probs.gather(1, labels.view(-1, 1)).squeeze()
        true_probs = torch.clamp(true_probs, min=1e-8, max=1.0)
        
        if q == 0:
            return F.cross_entropy(logits, labels)
        else:
            loss = (1 - torch.pow(true_probs, q)) / q
            return loss.mean()
    
    def compute_consistency_loss(self, classifier_outputs):
        """Encourage consistency between ensemble components"""
        mean_logits = classifier_outputs['mean_logits']
        max_logits = classifier_outputs['max_logits']
        add_logits = classifier_outputs['add_logits']
        
        # MSE between logits (encourages similar predictions)
        consistency_loss = (
            F.mse_loss(mean_logits, max_logits) +
            F.mse_loss(mean_logits, add_logits) +
            F.mse_loss(max_logits, add_logits)
        ) / 3
        
        return consistency_loss
    
    def evaluate(self, test_loader):
        """Evaluation with ensemble analysis"""
        self.model.eval()
        correct = 0
        total = 0
        all_agreements = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                # Get predictions
                classifier_outputs, _ = self.model(
                    batch.x, batch.edge_index, batch.batch, return_all=True
                )
                
                # Ensemble prediction
                ensemble_logits = classifier_outputs['ensemble_logits']
                pred = ensemble_logits.argmax(dim=1)
                
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
                
                # Track agreement
                agreements = self.model.compute_ensemble_agreement(classifier_outputs)
                all_agreements.extend(agreements.cpu().numpy())
        
        accuracy = correct / total
        avg_agreement = np.mean(all_agreements)
        
        return {
            'accuracy': accuracy,
            'agreement': avg_agreement,
            'total_samples': total
        }

def analyze_ensemble_behavior(model, dataset, device):
    """
    Analyze how the ensemble behaves on your specific dataset
    This helps you understand if the ensemble is providing value
    """
    print("\n=== ANALYZING ENSEMBLE BEHAVIOR ===")
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    
    all_agreements = []
    individual_accuracies = {'mean': 0, 'max': 0, 'add': 0, 'ensemble': 0}
    total_samples = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            classifier_outputs, _ = model(
                batch.x, batch.edge_index, batch.batch, return_all=True
            )
            
            # Individual predictions
            mean_pred = classifier_outputs['mean_logits'].argmax(dim=1)
            max_pred = classifier_outputs['max_logits'].argmax(dim=1)
            add_pred = classifier_outputs['add_logits'].argmax(dim=1)
            ensemble_pred = classifier_outputs['ensemble_logits'].argmax(dim=1)
            
            # Accuracies
            individual_accuracies['mean'] += (mean_pred == batch.y).sum().item()
            individual_accuracies['max'] += (max_pred == batch.y).sum().item()
            individual_accuracies['add'] += (add_pred == batch.y).sum().item()
            individual_accuracies['ensemble'] += (ensemble_pred == batch.y).sum().item()
            
            total_samples += batch.y.size(0)
            
            # Agreement analysis
            agreements = model.compute_ensemble_agreement(classifier_outputs)
            all_agreements.extend(agreements.cpu().numpy())
    
    # Calculate final accuracies
    for key in individual_accuracies:
        individual_accuracies[key] /= total_samples
    
    print("Individual Component Accuracies:")
    print(f"  Mean pooling: {individual_accuracies['mean']:.4f}")
    print(f"  Max pooling:  {individual_accuracies['max']:.4f}")
    print(f"  Add pooling:  {individual_accuracies['add']:.4f}")
    print(f"  Ensemble:     {individual_accuracies['ensemble']:.4f}")
    
    # Ensemble benefit analysis
    best_individual = max(individual_accuracies['mean'], 
                         individual_accuracies['max'], 
                         individual_accuracies['add'])
    
    ensemble_benefit = individual_accuracies['ensemble'] - best_individual
    
    print(f"\nEnsemble benefit: {ensemble_benefit:+.4f}")
    
    if ensemble_benefit > 0.01:
        print("✅ Ensemble provides clear benefit")
    elif ensemble_benefit > 0:
        print("⚠️  Ensemble provides marginal benefit")
    else:
        print("❌ Ensemble may not be helping - consider simplifying")
    
    # Agreement analysis
    avg_agreement = np.mean(all_agreements)
    agreement_std = np.std(all_agreements)
    
    print(f"\nAgreement Statistics:")
    print(f"  Average disagreement: {avg_agreement:.4f}")
    print(f"  Disagreement std:     {agreement_std:.4f}")
    
    # Identify potential noisy samples (high disagreement)
    high_disagreement_threshold = avg_agreement + 2 * agreement_std
    potential_noisy = sum(1 for a in all_agreements if a > high_disagreement_threshold)
    
    print(f"  Potential noisy samples: {potential_noisy} ({potential_noisy/len(all_agreements)*100:.1f}%)")
    
    return {
        'individual_accuracies': individual_accuracies,
        'ensemble_benefit': ensemble_benefit,
        'avg_disagreement': avg_agreement,
        'potential_noisy_count': potential_noisy
    }

def run_phase3_ensemble(dataset, device, phase2_results):
    """Complete Phase 3: Ensemble Methods"""
    print("🚀 STARTING PHASE 3: ENSEMBLE METHODS")
    
    # Initialize ensemble model
    model = RobustEnsembleModel(
        input_dim=dataset.num_node_features,
        hidden_dim=128,
        num_classes=dataset.num_classes,
        loss_type=phase2_results['best_loss_function']
    ).to(device)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train model
    trainer = EnsembleTrainer(model, device)
    
    print("Training ensemble model...")
    for epoch in range(100):
        train_stats = trainer.train_epoch(train_loader, epoch)
        
        if epoch % 20 == 0:
            eval_stats = trainer.evaluate(test_loader)
            print(f"Epoch {epoch}: Loss {train_stats['loss']:.4f}, "
                  f"Acc {eval_stats['accuracy']:.4f}, "
                  f"Agreement {eval_stats['agreement']:.4f}")
    
    # Final evaluation and analysis
    final_eval = trainer.evaluate(test_loader)
    ensemble_analysis = analyze_ensemble_behavior(model, dataset, device)
    
    print(f"\n📋 PHASE 3 SUMMARY:")
    print(f"   - Final accuracy: {final_eval['accuracy']:.4f}")
    print(f"   - Ensemble benefit: {ensemble_analysis['ensemble_benefit']:+.4f}")
    print(f"   - Potential noisy samples detected: {ensemble_analysis['potential_noisy_count']}")
    
    return {
        'model': model,
        'final_accuracy': final_eval['accuracy'],
        'ensemble_benefit': ensemble_analysis['ensemble_benefit'],
        'training_history': trainer.training_history,
        'ready_for_phase4': True
    }