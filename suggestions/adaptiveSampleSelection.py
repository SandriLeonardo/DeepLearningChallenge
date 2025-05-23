import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class AdaptiveSampleSelector:
    """
    Intelligent sample selection based on multiple criteria
    
    Core philosophy: Good samples should be:
    1. Consistent with model predictions (small loss)
    2. Stable across training epochs (consistent predictions)
    3. Agreed upon by ensemble components (high consensus)
    4. Following expected learning patterns (early vs late learning)
    """
    
    def __init__(self, dataset_size, num_classes, warmup_epochs=20, 
                 selection_strategy='hybrid'):
        self.dataset_size = dataset_size
        self.num_classes = num_classes
        self.warmup_epochs = warmup_epochs
        self.selection_strategy = selection_strategy
        
        # Historical tracking for each sample
        self.sample_losses = defaultdict(list)  # Loss history per sample
        self.sample_predictions = defaultdict(list)  # Prediction history
        self.sample_agreements = defaultdict(list)  # Ensemble agreement history
        self.sample_confidence = defaultdict(list)  # Confidence history
        
        # Meta-statistics
        self.global_loss_trajectory = []
        self.selection_history = []
        
    def update_sample_statistics(self, sample_indices, losses, predictions, 
                               agreements, confidences, epoch):
        """
        Update historical statistics for each sample
        
        This is the heart of adaptive selection - we're building a profile
        of each training sample's behavior over time
        """
        for idx, loss, pred, agreement, conf in zip(
            sample_indices, losses, predictions, agreements, confidences
        ):
            idx = idx.item() if torch.is_tensor(idx) else idx
            
            self.sample_losses[idx].append(loss)
            self.sample_predictions[idx].append(pred)
            self.sample_agreements[idx].append(agreement)
            self.sample_confidence[idx].append(conf)
            
            # Keep only recent history to avoid memory issues
            max_history = 50
            if len(self.sample_losses[idx]) > max_history:
                self.sample_losses[idx] = self.sample_losses[idx][-max_history:]
                self.sample_predictions[idx] = self.sample_predictions[idx][-max_history:]
                self.sample_agreements[idx] = self.sample_agreements[idx][-max_history:]
                self.sample_confidence[idx] = self.sample_confidence[idx][-max_history:]
    
    def compute_sample_quality_scores(self, sample_indices, current_epoch):
        """
        Compute quality score for each sample based on historical behavior
        
        Higher scores indicate higher quality (more likely to be correctly labeled)
        """
        if current_epoch < self.warmup_epochs:
            # During warmup, treat all samples equally
            return torch.ones(len(sample_indices))
        
        quality_scores = []
        
        for idx in sample_indices:
            idx = idx.item() if torch.is_tensor(idx) else idx
            
            if idx not in self.sample_losses or len(self.sample_losses[idx]) < 5:
                # New or insufficient history - neutral score
                quality_scores.append(0.5)
                continue
            
            # 1. Loss consistency: Good samples should have stable, low losses
            losses = np.array(self.sample_losses[idx])
            loss_trend = np.mean(losses[-5:])  # Recent average
            loss_stability = 1.0 / (1.0 + np.std(losses[-10:]))  # Stability score
            
            # 2. Prediction consistency: Good samples should have stable predictions
            predictions = np.array(self.sample_predictions[idx])
            if len(predictions) > 1:
                pred_changes = np.sum(predictions[1:] != predictions[:-1])
                pred_stability = 1.0 / (1.0 + pred_changes / len(predictions))
            else:
                pred_stability = 1.0
            
            # 3. Ensemble agreement: Good samples should have high agreement
            agreements = np.array(self.sample_agreements[idx])
            avg_agreement = np.mean(agreements[-5:])  # Recent agreement
            agreement_score = 1.0 / (1.0 + avg_agreement)  # Lower disagreement is better
            
            # 4. Confidence evolution: Good samples should maintain high confidence
            confidences = np.array(self.sample_confidence[idx])
            avg_confidence = np.mean(confidences[-5:])
            
            # Combine scores with weights
            quality_score = (
                0.3 * (1.0 / (1.0 + loss_trend)) +  # Lower loss is better
                0.25 * loss_stability +
                0.25 * pred_stability +
                0.15 * agreement_score +
                0.05 * avg_confidence
            )
            
            quality_scores.append(quality_score)
        
        return torch.tensor(quality_scores, dtype=torch.float32)
    
    def select_clean_samples(self, sample_indices, quality_scores, labels, 
                           current_epoch, noise_rate=0.3):
        """
        Select clean samples based on quality scores and current training phase
        
        This implements the core "small loss" trick with additional sophistication
        """
        if current_epoch < self.warmup_epochs:
            # During warmup, use all samples
            return torch.arange(len(sample_indices))
        
        # Adaptive selection rate
        # Start conservative, become more aggressive as training progresses
        progress = min(1.0, (current_epoch - self.warmup_epochs) / 50)
        
        if self.selection_strategy == 'conservative':
            # Conservative: Select more samples, risk including some noise
            keep_ratio = 1.0 - noise_rate * 0.5 * progress
        elif self.selection_strategy == 'aggressive':
            # Aggressive: Select fewer samples, risk losing some clean data
            keep_ratio = 1.0 - noise_rate * 1.5 * progress
        else:  # hybrid
            # Hybrid: Balanced approach
            keep_ratio = 1.0 - noise_rate * progress
        
        # Ensure we keep at least some samples
        keep_ratio = max(0.3, keep_ratio)
        num_keep = max(1, int(len(sample_indices) * keep_ratio))
        
        # Select samples with highest quality scores
        _, top_indices = torch.topk(quality_scores, num_keep, largest=True)
        
        # Store selection statistics
        self.selection_history.append({
            'epoch': current_epoch,
            'keep_ratio': keep_ratio,
            'num_selected': num_keep,
            'avg_quality': quality_scores[top_indices].mean().item()
        })
        
        return top_indices

class CompleteHybridModel(nn.Module):
    """
    Complete hybrid model integrating all components:
    - Robust loss functions
    - Ensemble methods
    - Adaptive sample selection
    - Progressive training strategy
    """
    
    def __init__(self, input_dim, hidden_dim, num_classes, dataset_size, 
                 loss_type='gce', selection_strategy='hybrid'):
        super().__init__()
        
        # Import the ensemble components from Phase 3
        from previous_phases import MultiPoolingGraphEncoder, EnsembleClassifier
        
        self.encoder = MultiPoolingGraphEncoder(input_dim, hidden_dim)
        self.classifier = EnsembleClassifier(hidden_dim, num_classes)
        
        # Sample selector
        self.sample_selector = AdaptiveSampleSelector(
            dataset_size, num_classes, 
            selection_strategy=selection_strategy
        )
        
        # Loss configuration
        self.loss_type = loss_type
        self.num_classes = num_classes
        
        # Training phase tracking
        self.current_epoch = 0
        self.training_phases = {
            'warmup': (0, 20),      # Learn basic patterns
            'selection': (20, 80),   # Apply sample selection
            'refinement': (80, 150)  # Fine-tune on selected samples
        }
        
    def forward(self, x, edge_index, batch=None, sample_indices=None, return_all=False):
        """Forward pass with sample tracking"""
        # Encode with multiple pooling
        pooled_features = self.encoder(x, edge_index, batch)
        
        # Classify with ensemble
        classifier_outputs = self.classifier(pooled_features)
        
        # Compute ensemble agreement for sample selection
        if sample_indices is not None:
            agreements = self.compute_ensemble_agreement(classifier_outputs)
            
            # Update sample statistics
            with torch.no_grad():
                losses = self.compute_sample_losses(
                    classifier_outputs['ensemble_logits'], 
                    None  # Labels will be provided separately
                )
                
                predictions = classifier_outputs['ensemble_logits'].argmax(dim=1)
                confidences = F.softmax(classifier_outputs['ensemble_logits'], dim=1).max(dim=1)[0]
                
                # Note: We'll update statistics in the training loop where we have labels
        
        if return_all:
            return classifier_outputs, pooled_features, agreements if sample_indices is not None else None
        else:
            return classifier_outputs['ensemble_logits']
    
    def compute_ensemble_agreement(self, classifier_outputs):
        """Compute agreement between ensemble components"""
        mean_probs = F.softmax(classifier_outputs['mean_logits'], dim=1)
        max_probs = F.softmax(classifier_outputs['max_logits'], dim=1)
        add_probs = F.softmax(classifier_outputs['add_logits'], dim=1)
        
        # Compute pairwise KL divergences
        kl_div = lambda p, q: F.kl_div(torch.log(p + 1e-8), q, reduction='none').sum(dim=1)
        
        disagreement = (
            kl_div(mean_probs, max_probs) +
            kl_div(mean_probs, add_probs) +
            kl_div(max_probs, add_probs)
        ) / 3
        
        return disagreement
    
    def compute_sample_losses(self, logits, labels):
        """Compute per-sample losses"""
        if labels is None:
            return torch.zeros(logits.size(0))
        
        if self.loss_type == 'gce':
            return self.compute_gce_loss(logits, labels, reduction='none')
        else:
            return F.cross_entropy(logits, labels, reduction='none')
    
    def compute_gce_loss(self, logits, labels, q=0.7, reduction='mean'):
        """Generalized Cross Entropy loss"""
        probs = F.softmax(logits, dim=1)
        true_probs = probs.gather(1, labels.view(-1, 1)).squeeze()
        true_probs = torch.clamp(true_probs, min=1e-8, max=1.0)
        
        if q == 0:
            loss = -torch.log(true_probs)
        else:
            loss = (1 - torch.pow(true_probs, q)) / q
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            return loss.sum()

class ProgressiveTrainer:
    """
    Progressive trainer that adapts strategy based on training phase
    
    This is the orchestrator that brings everything together:
    - Manages training phases
    - Coordinates sample selection
    - Balances different objectives
    - Monitors training health
    """
    
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model
        self.device = device
        
        # Different optimizers for different phases
        self.optimizers = {
            'warmup': torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4),
            'selection': torch.optim.Adam(model.parameters(), lr=learning_rate * 0.5, weight_decay=1e-3),
            'refinement': torch.optim.Adam(model.parameters(), lr=learning_rate * 0.1, weight_decay=1e-2)
        }
        
        # Schedulers for each phase
        self.schedulers = {
            phase: torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=phase_end - phase_start, eta_min=1e-6
            )
            for phase, (phase_start, phase_end) in model.training_phases.items()
            for optimizer in [self.optimizers[phase]]
        }
        
        # Training monitoring
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'selection_ratios': [],
            'sample_quality': [],
            'phase_transitions': []
        }
        
    def get_current_phase(self, epoch):
        """Determine current training phase"""
        for phase, (start, end) in self.model.training_phases.items():
            if start <= epoch < end:
                return phase
        return 'refinement'  # Default to final phase
    
    def train_epoch(self, train_loader, epoch, validation_loader=None):
        """
        Train one epoch with phase-appropriate strategy
        
        This is where all the magic happens - the coordination of:
        - Sample selection
        - Loss computation
        - Ensemble training
        - Progressive difficulty
        """
        self.model.current_epoch = epoch
        current_phase = self.get_current_phase(epoch)
        
        # Phase transition logging
        if epoch > 0:
            prev_phase = self.get_current_phase(epoch - 1)
            if current_phase != prev_phase:
                print(f"\n🔄 Transitioning from {prev_phase} to {current_phase} phase")
                self.training_history['phase_transitions'].append({
                    'epoch': epoch,
                    'from_phase': prev_phase,
                    'to_phase': current_phase
                })
        
        # Get appropriate optimizer and scheduler
        optimizer = self.optimizers[current_phase]
        scheduler = self.schedulers[current_phase]
        
        self.model.train()
        epoch_loss = 0
        epoch_selected_samples = 0
        epoch_total_samples = 0
        sample_quality_scores = []
        
        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            
            # Create sample indices for this batch
            batch_size = batch.y.size(0)
            sample_indices = torch.arange(
                batch_idx * batch_size, 
                batch_idx * batch_size + batch_size
            )
            
            # Forward pass with full outputs
            classifier_outputs, pooled_features, agreements = self.model(
                batch.x, batch.edge_index, batch.batch, 
                sample_indices=sample_indices, return_all=True
            )
            
            # Compute sample-wise losses and statistics
            ensemble_logits = classifier_outputs['ensemble_logits']
            sample_losses = self.model.compute_sample_losses(ensemble_logits, batch.y)
            
            predictions = ensemble_logits.argmax(dim=1)
            confidences = F.softmax(ensemble_logits, dim=1).max(dim=1)[0]
            
            # Update sample selector with current batch statistics
            self.model.sample_selector.update_sample_statistics(
                sample_indices, 
                sample_losses.detach().cpu().numpy(),
                predictions.detach().cpu().numpy(),
                agreements.detach().cpu().numpy(),
                confidences.detach().cpu().numpy(),
                epoch
            )
            
            # Get quality scores and select samples
            quality_scores = self.model.sample_selector.compute_sample_quality_scores(
                sample_indices, epoch
            )
            
            selected_indices = self.model.sample_selector.select_clean_samples(
                sample_indices, quality_scores, batch.y, epoch
            )
            
            # Train only on selected samples
            if len(selected_indices) > 0:
                selected_logits = ensemble_logits[selected_indices]
                selected_labels = batch.y[selected_indices]
                
                # Compute loss on selected samples
                if self.model.loss_type == 'gce':
                    loss = self.model.compute_gce_loss(selected_logits, selected_labels)
                else:
                    loss = F.cross_entropy(selected_logits, selected_labels)
                
                # Add ensemble consistency regularization
                consistency_loss = self.compute_ensemble_consistency(
                    classifier_outputs, selected_indices
                )
                
                total_loss = loss + 0.1 * consistency_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += total_loss.item()
                epoch_selected_samples += len(selected_indices)
            
            epoch_total_samples += batch_size
            sample_quality_scores.extend(quality_scores.numpy())
        
        scheduler.step()
        
        # Epoch statistics
        avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
        selection_ratio = epoch_selected_samples / epoch_total_samples if epoch_total_samples > 0 else 0
        avg_quality = np.mean(sample_quality_scores) if sample_quality_scores else 0
        
        # Store training history
        self.training_history['losses'].append(avg_loss)
        self.training_history['selection_ratios'].append(selection_ratio)
        self.training_history['sample_quality'].append(avg_quality)
        
        # Validation if provided
        val_accuracy = 0
        if validation_loader is not None:
            val_accuracy = self.evaluate(validation_loader)
            self.training_history['accuracies'].append(val_accuracy)
        
        return {
            'loss': avg_loss,
            'selection_ratio': selection_ratio,
            'sample_quality': avg_quality,
            'validation_accuracy': val_accuracy,
            'phase': current_phase,
            'learning_rate': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else 0
        }
    
    def compute_ensemble_consistency(self, classifier_outputs, selected_indices):
        """Compute consistency loss between ensemble components"""
        if len(selected_indices) == 0:
            return torch.tensor(0.0, device=self.device)
        
        mean_logits = classifier_outputs['mean_logits'][selected_indices]
        max_logits = classifier_outputs['max_logits'][selected_indices]
        add_logits = classifier_outputs['add_logits'][selected_indices]
        
        # Encourage similar predictions from all components
        consistency = (
            F.mse_loss(mean_logits, max_logits) +
            F.mse_loss(mean_logits, add_logits) +
            F.mse_loss(max_logits, add_logits)
        ) / 3
        
        return consistency
    
    def evaluate(self, test_loader):
        """Evaluate model performance"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                
                # Use ensemble prediction
                ensemble_logits = self.model(batch.x, batch.edge_index, batch.batch)
                pred = ensemble_logits.argmax(dim=1)
                
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
        
        return correct / total if total > 0 else 0

def analyze_training_dynamics(trainer, save_plots=True):
    """
    Analyze how the training progressed through different phases
    This helps you understand if your hybrid approach is working
    """
    print("\n=== ANALYZING TRAINING DYNAMICS ===")
    
    history = trainer.training_history
    
    # Basic statistics
    print(f"Training completed with {len(history['losses'])} epochs")
    print(f"Final loss: {history['losses'][-1]:.4f}")
    print(f"Best validation accuracy: {max(history['accuracies']):.4f}")
    print(f"Final selection ratio: {history['selection_ratios'][-1]:.4f}")
    
    # Phase analysis
    phase_transitions = history['phase_transitions']
    for transition in phase_transitions:
        epoch = transition['epoch']
        from_phase = transition['from_phase']
        to_phase = transition['to_phase']
        
        if epoch < len(history['losses']):
            loss_before = history['losses'][max(0, epoch-5):epoch]
            loss_after = history['losses'][epoch:min(len(history['losses']), epoch+5)]
            
            print(f"\nPhase transition at epoch {epoch} ({from_phase} → {to_phase}):")
            print(f"  Loss before: {np.mean(loss_before):.4f}")
            print(f"  Loss after: {np.mean(loss_after):.4f}")
    
    # Sample selection analysis
    selection_ratios = history['selection_ratios']
    avg_selection = np.mean(selection_ratios[-20:])  # Last 20 epochs
    
    print(f"\nSample Selection Analysis:")
    print(f"  Average selection ratio (final 20 epochs): {avg_selection:.4f}")
    
    if avg_selection > 0.8:
        print("  ✅ High selection ratio - dataset appears relatively clean")
    elif avg_selection > 0.5:
        print("  ⚠️  Moderate selection ratio - some noise detected and handled")
    else:
        print("  ❌ Low selection ratio - significant noise detected")
    
    # Sample quality evolution
    quality_scores = history['sample_quality']
    if quality_scores:
        quality_improvement = quality_scores[-1] - quality_scores[0]
        print(f"  Sample quality improvement: {quality_improvement:+.4f}")
        
        if quality_improvement > 0.1:
            print("  ✅ Significant quality improvement - model learning to distinguish clean samples")
        elif quality_improvement > 0:
            print("  ⚠️  Modest quality improvement - some learning happening")
        else:
            print("  ❌ No quality improvement - check if sample selection is working")
    
    return {
        'final_accuracy': max(history['accuracies']) if history['accuracies'] else 0,
        'final_selection_ratio': avg_selection,
        'quality_improvement': quality_improvement if quality_scores else 0,
        'training_stability': np.std(history['losses'][-20:]) if len(history['losses']) >= 20 else float('inf')
    }

def run_complete_pipeline(dataset, device, noise_rate=0.3, epochs=150):
    """
    Run the complete hybrid approach pipeline
    This is your main entry point for the full system
    """
    print("🚀 STARTING COMPLETE HYBRID PIPELINE")
    print(f"Dataset size: {len(dataset)}")
    print(f"Expected noise rate: {noise_rate}")
    print(f"Training epochs: {epochs}")
    
    # Initialize complete model
    model = CompleteHybridModel(
        input_dim=dataset.num_node_features,
        hidden_dim=128,
        num_classes=dataset.num_classes,
        dataset_size=len(dataset),
        loss_type='gce',  # Use GCE as default robust loss
        selection_strategy='hybrid'
    ).to(device)
    
    # Dataset splits
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Progressive trainer
    trainer = ProgressiveTrainer(model, device, learning_rate=0.001)
    
    # Training loop with monitoring
    print("\n📈 Starting training with progressive phases...")
    
    best_val_acc = 0
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train epoch
        train_stats = trainer.train_epoch(train_loader, epoch, val_loader)
        
        # Early stopping based on validation accuracy
        if train_stats['validation_accuracy'] > best_val_acc:
            best_val_acc = train_stats['validation_accuracy']
            patience_counter = 0
            
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        # Progress reporting
        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch:3d} [{train_stats['phase']:>10s}]: "
                  f"Loss {train_stats['loss']:.4f}, "
                  f"Val Acc {train_stats['validation_accuracy']:.4f}, "
                  f"Selection {train_stats['selection_ratio']:.3f}, "
                  f"Quality {train_stats['sample_quality']:.3f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏰ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Final evaluation
    final_test_acc = trainer.evaluate(test_loader)
    
    # Analyze training dynamics
    analysis = analyze_training_dynamics(trainer)
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   - Best validation accuracy: {best_val_acc:.4f}")
    print(f"   - Final test accuracy: {final_test_acc:.4f}")
    print(f"   - Training completed in {epoch + 1} epochs")
    print(f"   - Final sample selection ratio: {analysis['final_selection_ratio']:.4f}")
    
    return {
        'model': model,
        'trainer': trainer,
        'best_val_accuracy': best_val_acc,
        'final_test_accuracy': final_test_acc,
        'training_analysis': analysis,
        'epochs_completed': epoch + 1
    }