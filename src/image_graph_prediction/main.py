## Importing libraries
from image_graph_prediction.model_config import Config
from image_graph_prediction.model import PGATClassifier
from image_graph_prediction.utils import set_up_media_eval_dataset, set_up_multimodal_dataset
import numpy as np

import math


import torch
from torch import nn

from torch.optim import AdamW

from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

from collections import Counter


from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
from image_graph_prediction.utils import plot_acc_graph, plot_loss_graph

import wandb

# Initialize wandb
wandb.init(project="multimodal-graph-classification", entity="", name="multimodal-experiment")

# =====================================================
# CLASS IMBALANCE HANDLING SOLUTIONS
# =====================================================
class AUCMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
    
    def update(self, preds, targets):
        """Update with batch predictions and targets"""
        if torch.is_tensor(preds):
            # Apply sigmoid if not already applied
            if preds.min() < 0 or preds.max() > 1:
                preds = torch.sigmoid(preds)
            preds = preds.detach().cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.detach().cpu().numpy()
        
        self.predictions.extend(preds.flatten())
        self.targets.extend(targets.flatten())
    
    def compute_metrics(self):
        """Compute all AUC-related metrics"""
        if len(self.predictions) == 0:
            return {}
        
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(targets, preds)
        except ValueError:
            roc_auc = 0.5  # If only one class present
        
        # Precision-Recall AUC
        try:
            precision, recall, _ = precision_recall_curve(targets, preds)
            pr_auc = auc(recall, precision)
        except ValueError:
            pr_auc = 0.0
        
        # ROC Curve data for plotting
        try:
            fpr, tpr, _ = roc_curve(targets, preds)
            roc_data = (fpr, tpr)
        except ValueError:
            roc_data = None
        
        # PR Curve data for plotting
        try:
            precision, recall, _ = precision_recall_curve(targets, preds)
            pr_data = (precision, recall)
        except ValueError:
            pr_data = None
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'roc_data': roc_data,
            'pr_data': pr_data
        }

def log_auc_to_wandb(metrics_dict, epoch, phase='train'):
    """Log AUC metrics to wandb with custom plots"""
    
    # Log scalar metrics
    wandb.log({
        f'{phase}/roc_auc': metrics_dict['roc_auc'],
        f'{phase}/pr_auc': metrics_dict['pr_auc'],
        'epoch': epoch
    })
    
    # Create and log ROC curve plot
    if metrics_dict['roc_data'] is not None:
        fpr, tpr = metrics_dict['roc_data']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {metrics_dict["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{phase.capitalize()} ROC Curve - Epoch {epoch}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Log to wandb
        wandb.log({f'{phase}/roc_curve': wandb.Image(plt)})
        plt.close()
    
    # Create and log Precision-Recall curve
    if metrics_dict['pr_data'] is not None:
        precision, recall = metrics_dict['pr_data']
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {metrics_dict["pr_auc"]:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{phase.capitalize()} Precision-Recall Curve - Epoch {epoch}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        # Log to wandb
        wandb.log({f'{phase}/pr_curve': wandb.Image(plt)})
        plt.close()

def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced dataset"""
    # Get all labels
    labels = []
    for data in dataset:
        labels.append(data.y.item())
    
    # Count classes
    class_counts = Counter(labels)
    print(f"Class distribution: {class_counts}")
    
    # Calculate weights - inverse frequency
    total_samples = len(labels)
    weights = {}
    for class_label, count in class_counts.items():
        weights[class_label] = total_samples / (len(class_counts) * count)
    
    print(f"Class weights: {weights}")
    return weights

def get_weighted_loss_function(dataset, device):
    """Create weighted BCE loss function"""
    class_weights = calculate_class_weights(dataset)
    
    # Convert to tensor - make sure order is correct [weight_for_0, weight_for_1]
    if 0.0 in class_weights and 1.0 in class_weights:
        pos_weight = torch.tensor([class_weights[1.0] / class_weights[0.0]], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight: {pos_weight.item():.4f}")
    else:
        # Fallback to regular BCE
        criterion = nn.BCEWithLogitsLoss()
        print("Using regular BCEWithLogitsLoss")
    
    return criterion

def create_weighted_sampler(dataset):
    """Create weighted random sampler for balanced batching"""
    # Get all labels
    labels = []
    for data in dataset:
        labels.append(data.y.item())
    
    # Calculate class weights
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate sample weights (inverse frequency)
    sample_weights = []
    for label in labels:
        weight = total_samples / (len(class_counts) * class_counts[label])
        sample_weights.append(weight)
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow replacement to balance classes
    )
    
    print(f"Created weighted sampler with {len(sample_weights)} samples")
    return sampler

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate focal loss
        ce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def find_optimal_threshold(model, dataloader, device):
    """Find optimal classification threshold for imbalanced data"""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            probs = torch.sigmoid(out).cpu().numpy()
            targets = batch.y.cpu().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(targets)
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Try different thresholds
    best_f1 = 0.0
    best_threshold = 0.5

    
    thresholds = np.arange(0.1, 0.9, 0.05)
    for threshold in thresholds:
        preds = (all_probs > threshold).astype(int)
        
        # Calculate F1 score
        tp = np.sum((preds == 1) & (all_targets == 1))
        fp = np.sum((preds == 1) & (all_targets == 0))
        fn = np.sum((preds == 0) & (all_targets == 1))
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
            
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
            
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
            
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f} with F1: {best_f1:.4f}")
    return best_threshold

# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_for_nan_inf(tensor, name):
    """Helper function to check for NaN or Inf values"""
    if tensor is None:
        return False
    if torch.isnan(tensor).any():
        print(f"WARNING: NaN found in {name}")
        return True
    if torch.isinf(tensor).any():
        print(f"WARNING: Inf found in {name}")
        return True
    return False

def create_fresh_model():
    """Create a completely fresh model instance to avoid state retention"""
    return PGATClassifier()

# =====================================================
# BALANCED TRAINING FUNCTION
# =====================================================

def train_func_epoch_balanced(epoch, model, dataloader, device, optimizer, scheduler, criterion, optimal_threshold=0.5):
    """Training function with balanced loss and better metrics"""
    model.train()
    total_loss = 0
    valid_batches = 0
    
    # Initialize metrics
    train_acc = BinaryAccuracy().to(device)
    train_prec = BinaryPrecision().to(device)
    train_rec = BinaryRecall().to(device)
    train_f1 = BinaryF1Score().to(device)

    # Initialize AUC metrics
    auc_metrics = AUCMetrics()
    
    # Track class-specific predictions
    all_preds = []
    all_targets = []
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            optimizer.zero_grad()
            
            batch = batch.to(device)
            
            # Validate batch data
            if check_for_nan_inf(batch.x, f"batch.x at batch {batch_idx}"):
                print(f"Skipping batch {batch_idx} due to NaN/Inf in input features")
                continue
            
            if batch.edge_index.dtype != torch.long:
                batch.edge_index = batch.edge_index.long()
            
            # CRITICAL: Detach all inputs to ensure no gradient tracking from previous iterations
            x_input = batch.x.detach().requires_grad_(True)
            edge_index_input = batch.edge_index.detach()
            batch_input = batch.batch.detach() if hasattr(batch, 'batch') else None
            
            # Forward pass - completely fresh computation graph
            if batch_input is not None:
                out = model(x_input, edge_index_input, batch_input)
            else:
                out = model(x_input, edge_index_input)
            
            if check_for_nan_inf(out, f"model output at batch {batch_idx}"):
                print(f"Skipping batch {batch_idx} due to NaN/Inf in model output")
                continue
            
            # Prepare targets
            targets = batch.y.float().detach()
            if targets.dim() > 1:
                targets = targets.view(-1)
            
            if check_for_nan_inf(targets, f"targets at batch {batch_idx}"):
                print(f"Skipping batch {batch_idx} due to NaN/Inf in targets")
                continue
            
            if out.shape != targets.shape:
                print(f"Shape mismatch at batch {batch_idx}: out {out.shape}, targets {targets.shape}")
                continue
            
            # Calculate weighted loss
            loss = criterion(out, targets)
            
            if check_for_nan_inf(loss, f"loss at batch {batch_idx}"):
                print(f"Skipping batch {batch_idx} due to NaN/Inf in loss")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Skipping batch {batch_idx} due to NaN/Inf gradients")
                optimizer.zero_grad()
                continue
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            valid_batches += 1
            
            # Calculate metrics with optimal threshold
            with torch.no_grad():
                preds = torch.sigmoid(out.detach()).cpu()
                targets_cpu = targets.detach().cpu()

                 # Update AUC metrics (use raw probabilities, not binary)
                auc_metrics.update(preds, targets_cpu)
                
                preds_binary = (preds > optimal_threshold).float()
                
                # Store for detailed analysis
                all_preds.extend(preds_binary.numpy())
                all_targets.extend(targets_cpu.numpy())
                
                # Update metrics
                train_acc.update(preds_binary, targets_cpu)
                train_prec.update(preds_binary, targets_cpu)
                train_rec.update(preds_binary, targets_cpu)
                train_f1.update(preds_binary, targets_cpu)
            
            # if batch_idx % 10 == 0:
            #     print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
                
        except RuntimeError as e:
            if "backward through the graph a second time" in str(e):
                print(f"CRITICAL: Graph reuse error in batch {batch_idx}")
                print("Forcing cleanup and continuing...")
                
                # Force cleanup
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            else:
                print(f"Runtime error in batch {batch_idx}: {str(e)}")
                optimizer.zero_grad()  # Clear gradients on any error
            
            continue
            
        except Exception as e:
            print(f"Unexpected error in batch {batch_idx}: {str(e)}")
            optimizer.zero_grad()  # Clear gradients on any error
            continue
        
        finally:
            # CRITICAL: Explicit cleanup after each batch
            if 'out' in locals():
                del out
            if 'loss' in locals():
                del loss
            if 'targets' in locals():
                del targets
            if 'x_input' in locals():
                del x_input
            if 'edge_index_input' in locals():
                del edge_index_input
            if 'batch_input' in locals():
                del batch_input
            
            # Clear CUDA cache periodically
            if batch_idx % 20 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Calculate detailed metrics
    if valid_batches > 0:
        # Class-specific metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Per-class metrics
        class_0_mask = all_targets == 0
        class_1_mask = all_targets == 1
        
        if class_0_mask.sum() > 0:
            class_0_acc = np.mean(all_preds[class_0_mask] == all_targets[class_0_mask])
            class_0_recall = np.mean(all_preds[class_0_mask] == 0)  # True negatives
        else:
            class_0_acc = class_0_recall = 0.0
            
        if class_1_mask.sum() > 0:
            class_1_acc = np.mean(all_preds[class_1_mask] == all_targets[class_1_mask])
            class_1_recall = np.mean(all_preds[class_1_mask] == 1)  # True positives
        else:
            class_1_acc = class_1_recall = 0.0

         # Compute AUC metrics
        auc_results = auc_metrics.compute_metrics()
        
        train_report = {
            "accuracy": train_acc.compute().item(),
            "precision": train_prec.compute().item(),
            "recall": train_rec.compute().item(),
            "f1_score": train_f1.compute().item(),
            "class_0_accuracy": class_0_acc,
            "class_1_accuracy": class_1_acc,
            "class_0_recall": class_0_recall,
            "class_1_recall": class_1_recall,
            "roc_auc": auc_results.get('roc_auc', 0.0),
            "pr_auc": auc_results.get('pr_auc', 0.0)
        }
        avg_loss = total_loss / valid_batches
        
        # print(f"\nDetailed Training Metrics:")
        # print(f"Real News (0) Accuracy: {class_0_acc:.4f}, Recall: {class_0_recall:.4f}")
        # print(f"Fake News (1) Accuracy: {class_1_acc:.4f}, Recall: {class_1_recall:.4f}")
        # print(f"ROC AUC: {auc_results.get('roc_auc', 0.0):.4f}, PR AUC: {auc_results.get('pr_auc', 0.0):.4f}")
        
    else:
        train_report = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0,
            "class_0_accuracy": 0.0, "class_1_accuracy": 0.0,
            "class_0_recall": 0.0, "class_1_recall": 0.0,
            "roc_auc": 0.0, "pr_auc": 0.0
        }
        avg_loss = float('inf')
        auc_results = {}
    
    return avg_loss, train_report, auc_results

# =====================================================
# BALANCED EVALUATION FUNCTION
# =====================================================

def eval_func_balanced(model, dataloader, device, epoch, criterion, optimal_threshold=0.5):
    """Evaluation function with improved error handling and optimal threshold"""
    model.eval()
    total_loss = 0
    valid_batches = 0
    
    # Initialize metrics
    val_acc = BinaryAccuracy().to(device)
    val_prec = BinaryPrecision().to(device)
    val_rec = BinaryRecall().to(device)
    val_f1 = BinaryF1Score().to(device)

    # Initialize AUC metrics
    auc_metrics = AUCMetrics()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            try:
                batch = batch.to(device)
                
                # Validate input data
                if check_for_nan_inf(batch.x, f"val batch.x at batch {batch_idx}"):
                    continue
                
                # Ensure edge_index is properly formatted
                if batch.edge_index.dtype != torch.long:
                    batch.edge_index = batch.edge_index.long()
                
                # Forward pass
                out = model(batch.x, batch.edge_index, batch.batch)
                
                # Validate output
                if check_for_nan_inf(out, f"val model output at batch {batch_idx}"):
                    continue
                
                # Prepare targets
                targets = batch.y.float()
                if targets.dim() > 1:
                    targets = targets.view(-1)
                
                # Ensure shapes match
                if out.shape != targets.shape:
                    continue
                
                loss = criterion(out, targets)
                
                if not check_for_nan_inf(loss, f"val loss at batch {batch_idx}"):
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    # Calculate metrics with optimal threshold
                    preds = torch.sigmoid(out).cpu()
                    targets_cpu = targets.cpu()

                    # Update AUC metrics (use raw probabilities, not binary)
                    auc_metrics.update(preds, targets_cpu)
                    
                    preds = torch.clamp(preds, 0.0, 1.0)
                    preds_binary = (preds > optimal_threshold).long()

                    all_preds.append(preds_binary)
                    all_targets.append(targets_cpu.long())
                    
                    val_acc.update(preds_binary.float(), targets_cpu)
                    val_prec.update(preds_binary.float(), targets_cpu)
                    val_rec.update(preds_binary.float(), targets_cpu)
                    val_f1.update(preds_binary.float(), targets_cpu)
                    
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    # Compute final metrics
    if valid_batches > 0:
        acc = val_acc.compute().item()
        prec = val_prec.compute().item()
        rec = val_rec.compute().item()
        f1_score = val_f1.compute().item()
        avg_loss = total_loss / valid_batches
        
    else:
        acc = prec = rec = f1_score = 0.0
        avg_loss = float('inf')

    auc_results = auc_metrics.compute_metrics()
    
    # Calculate per-class metrics
    class_0_acc = class_1_acc = 0.0
    class_0_prec = class_1_prec = 0.0
    class_0_rec = class_1_rec = 0.0
    class_0_f1 = class_1_f1 = 0.0
    
    if all_preds:
        all_preds_arr = torch.cat(all_preds).numpy()
        all_targets_arr = torch.cat(all_targets).numpy()
        
        # Calculate confusion matrix components
        tp = np.sum((all_preds_arr == 1) & (all_targets_arr == 1))  # True Positives
        tn = np.sum((all_preds_arr == 0) & (all_targets_arr == 0))  # True Negatives
        fp = np.sum((all_preds_arr == 1) & (all_targets_arr == 0))  # False Positives
        fn = np.sum((all_preds_arr == 0) & (all_targets_arr == 1))  # False Negatives
        
       # ✅ treat class 0 (real) as “positive” for its own metrics
        class_0_total = tn + fp            # all true real + misclassified real
        class_0_predicted = tn + fn        # everything predicted real

        if class_0_total > 0:
            class_0_acc  = tn / class_0_total
            class_0_rec  = tn / class_0_total   # recall for real
        if class_0_predicted > 0:
            class_0_prec = tn / (tn + fn)       # precision for real
        if class_0_prec + class_0_rec > 0:
            class_0_f1   = 2 * class_0_prec * class_0_rec / (class_0_prec + class_0_rec)
        
        # Class 1 metrics (considering class 1 as positive)
        class_1_total = tp + fn  # Total actual class 1 samples
        class_1_predicted = tp + fp  # Total predicted class 1 samples
        
        if class_1_total > 0:
            class_1_acc = tp / class_1_total  # True Positive Rate (same as recall)
            class_1_rec = tp / class_1_total  # Recall for class 1
        
        if class_1_predicted > 0:
            class_1_prec = tp / (tp + fp)  # Precision for class 1
        
        if class_1_prec + class_1_rec > 0:
            class_1_f1 = 2 * (class_1_prec * class_1_rec) / (class_1_prec + class_1_rec)
            
    else:
        all_preds_arr = np.array([])
        all_targets_arr = np.array([])
    
    report = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1_score,
        "class_0_accuracy": class_0_acc,
        "class_0_precision": class_0_prec,
        "class_0_recall": class_0_rec,
        "class_0_f1_score": class_0_f1,
        "class_1_accuracy": class_1_acc,
        "class_1_precision": class_1_prec,
        "class_1_recall": class_1_rec,
        "class_1_f1_score": class_1_f1,
        "roc_auc": auc_results.get('roc_auc', 0.0),
        "pr_auc": auc_results.get('pr_auc', 0.0),
    }
    
    return avg_loss, report, acc, prec, rec, f1_score, all_targets_arr, all_preds_arr, auc_results

# =====================================================
# MAIN TRAINING CODE
# =====================================================

# Configuration
config = Config()

if __name__ == '__main__':
    
    ## Setup the dataset 
    dataset_name = "multimodal"
    set_seed(42) 

    if dataset_name == "multimodal":  
        dataset_train, dataset_val, dataset_test = set_up_multimodal_dataset()
    else:
        dataset_train, dataset_val, dataset_test = set_up_media_eval_dataset()
    
    # Print class distribution
    print("\n=== CLASS DISTRIBUTION ANALYSIS ===")
    train_labels = [data.y.item() for data in dataset_train]
    val_labels = [data.y.item() for data in dataset_val]
    print(f"Training set: {Counter(train_labels)}")
    print(f"Validation set: {Counter(val_labels)}")
    
    # =====================================================
    # SETUP BALANCED TRAINING COMPONENTS
    # =====================================================
    
    # Option 1: Create weighted sampler for balanced batching
    print("\n=== SETTING UP WEIGHTED SAMPLER ===")
    weighted_sampler = create_weighted_sampler(dataset_train)
    
    # Option 2: Setup weighted loss function
    print("\n=== SETTING UP WEIGHTED LOSS FUNCTION ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Choose your loss function (uncomment one):
    # Option A: Weighted BCE Loss
    # criterion = get_weighted_loss_function(dataset_train, device)
    
    # Option B: Focal Loss (alternative)
    # fake_weight = 0.25            
    # real_weight = 0.75
    criterion = FocalLoss(alpha=0.2, gamma=2.0)
    # criterion = FocalLoss(alpha=None, gamma=4.0)
    # print("Using Focal Loss")
    
    ## Setup the dataloaders with weighted sampler
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        sampler=weighted_sampler,  # Use weighted sampler instead of shuffle
        drop_last=False,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=0
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=0
    ) 
    
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=0
    )
    
    ## Initialize the fixed model
    gnn_model = PGATClassifier()
    print("Total number of parameters:", sum(p.numel() for p in gnn_model.parameters()))
    gnn_model.to(device)
    
    ## Calculate number of train steps
    num_update_steps_per_epoch = math.ceil(len(dataloader_train) / config.gradient_accumulation_steps)
    num_train_steps = num_update_steps_per_epoch * config.epochs
    
    # Use a more conservative learning rate and optimizer settings
    optimizer = AdamW(
        gnn_model.parameters(), 
        lr=config.lr,  # Reduce learning rate
        weight_decay=config.weight_decay,   # Slightly less weight decay
        # eps=1e-8             # Epsilon for numerical stability
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=10,  # More warmup steps
        num_training_steps=num_train_steps
    )

    # Test the model thoroughly before training
    print("Testing model with multiple batches...")
    try:
        test_batches = 0
        for test_batch in dataloader_train:
            test_batch = test_batch.to(device)
            
            # Inference-only pass
            gnn_model.eval()
            with torch.no_grad():
                _ = gnn_model(test_batch.x, test_batch.edge_index, test_batch.batch)

            # Train-like pass for testing backward
            gnn_model.train()
            optimizer.zero_grad()

            # Detach + clone test_batch.x to avoid potential reuse of autograd graph
            fresh_x = test_batch.x.detach().clone().requires_grad_(True)

            test_out = gnn_model(fresh_x, test_batch.edge_index, test_batch.batch)
            test_targets = test_batch.y.float()
            if test_targets.dim() > 1:
                test_targets = test_targets.view(-1)

            test_loss = criterion(test_out, test_targets)
            test_loss.backward()
            optimizer.step()

            del test_out, test_targets, test_loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

                
        print(f"Successfully tested {test_batches} batches")
        
    except Exception as e:
        print(f"Error in model testing: {e}")
        exit(1)

    best_loss = np.inf
    optimal_threshold = 0.5  # Will be updated after first few epochs
    
    def reinitialize_model_if_needed():
        """Reinitialize model if graph issues persist"""
        global gnn_model
        print("Reinitializing model to clear any retained state...")
        gnn_model = create_fresh_model()
        gnn_model.to(device)
        
        # Reinitialize optimizer as well
        optimizer = AdamW(
            gnn_model.parameters(), 
            lr=config.lr ,
            weight_decay=config.weight_decay,
            # eps=1e-8
        )
        return optimizer
        
    # Add error recovery in your epoch loop:
    consecutive_errors = 0
    max_consecutive_errors = 3
    best_f1 = 0.0
    best_acc = 0.0
    epochs_no_improve = 0
    early_stopping_patience = 20
    early_stop = False
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    
    print("\n=== STARTING BALANCED TRAINING ===")
    
    for epoch in range(config.epochs):
        print(f"\n---------------------- Epoch: {epoch+1} ---------------------------------- \n")
        
        # Clear cache before each epoch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            ## Training Loop with balanced approach
            train_loss, train_report, train_auc_results = train_func_epoch_balanced(
                epoch+1, gnn_model, dataloader_train, device, optimizer, scheduler, criterion, optimal_threshold
            )
            consecutive_errors = 0  # Reset error counter on success
            
        except Exception as e:
            consecutive_errors += 1
            print(f"ERROR in epoch {epoch+1}: {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                print("Too many consecutive errors. Reinitializing model...")
                optimizer = reinitialize_model_if_needed()
                consecutive_errors = 0
                continue
            else:
                print("Retrying epoch...")
                continue

        ## Validation loop with balanced evaluation
        val_loss, report, acc, prec, rec, f1_score, val_labels, val_preds, val_auc_results= eval_func_balanced(
            gnn_model, dataloader_val, device, epoch+1, criterion, optimal_threshold
        )

        # Log AUC curves to wandb
        if train_auc_results:
            log_auc_to_wandb(train_auc_results, epoch+1, 'train')
        if val_auc_results:
            log_auc_to_wandb(val_auc_results, epoch+1, 'val')
        
        # Print confusion matrix if we have predictions
        if len(val_labels) > 0 and len(val_preds) > 0:
            print("Confusion Matrix:")
            print(confusion_matrix(val_labels, val_preds))
        
        print(f"\nEpoch: {epoch+1} | Training loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print()
        print("Train Report:")
        print(f"Accuracy: {train_report['accuracy']:.4f} | Precision: {train_report['precision']:.4f} | Recall: {train_report['recall']:.4f} | F1: {train_report['f1_score']:.4f}")
        print(f"Class 0 (Real) Acc: {train_report['class_0_accuracy']:.4f} | Class 1 (Fake) Acc: {train_report['class_1_accuracy']:.4f}")
        print()
        print("Validation Report:")
        print(f"Accuracy: {report['accuracy']:.4f} | Precision: {report['precision']:.4f} | Recall: {report['recall']:.4f} | F1: {report['f1_score']:.4f}")
        # print(f"ROC AUC: {report['roc_auc']:.4f} | PR AUC: {report['pr_auc']:.4f}")
        # Find optimal threshold every few epochs
        if epoch > 0 and epoch % 3 == 0:  # Update threshold every 3 epochs
            try:
                new_threshold = find_optimal_threshold(gnn_model, dataloader_val, device)
                optimal_threshold = new_threshold
                print(f"Updated optimal threshold to: {optimal_threshold:.3f}")
            except Exception as e:
                print(f"Could not update threshold: {e}")
        
        # Only log to wandb if values are valid
        if not any([math.isnan(x) or math.isinf(x) for x in [train_loss, val_loss, acc, prec, rec, f1_score]]):
            wandb.log({
                "train_loss": train_loss, 
                "train-acc": train_report["accuracy"],
                "train-prec": train_report["precision"],
                "train-rec": train_report["recall"],
                "train-f1": train_report["f1_score"],
                "train-class0-acc": train_report["class_0_accuracy"],
                "train-class1-acc": train_report["class_1_accuracy"],
                "val-loss": val_loss, 
                "val-prec": prec, 
                "val-rec": rec, 
                "val-f1score": f1_score, 
                "val-acc": report["accuracy"],
                # "val-roc-auc": report["roc_auc"],
                # "val-pr-auc": report["pr_auc"],
                "optimal_threshold": optimal_threshold,
                "Loss/Train": train_loss,
                "Loss/Validation": val_loss,
                "Accuracy/Train": train_report["accuracy"],
                "Accuracy/Validation": report["accuracy"],
                "epoch": epoch + 1,
                
            })
        
        print(f"\n----------------------------------------------------------------------------")
        
        # if f1_score > best_f1:
        #     best_f1 = f1_score
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1

        if report["accuracy"] > best_acc:
            best_acc = report["accuracy"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1    

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best Acc: {best_acc:.4f}")
            early_stop = True
            break
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': gnn_model.state_dict(),
                'optimal_threshold': optimal_threshold,
                'epoch': epoch,
                'val_loss': val_loss,
                'class_weights': calculate_class_weights(dataset_train)
            }, config.best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        
      
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_report["accuracy"])
        val_accs.append(report["accuracy"])



    print("Training completed!")

    # Final threshold optimization
    print("\n=== FINAL THRESHOLD OPTIMIZATION ===")
    try:
        final_threshold = find_optimal_threshold(gnn_model, dataloader_val, device)
        print(f"Final optimal threshold: {final_threshold:.3f}")
        
        # Save final model with optimal threshold
        torch.save({
            'model_state_dict': gnn_model.state_dict(),
            'optimal_threshold': final_threshold,
            'epoch': epoch,
            'class_weights': calculate_class_weights(dataset_train)
        }, config.best_model_final_path)
        
    except Exception as e:
        print(f"Could not perform final threshold optimization: {e}")
    
    
    print("\n=== EVALUATING ON TEST DATASET ===")

    # Initialize torchmetrics on the correct device
    test_accuracy = BinaryAccuracy().to(device)
    test_precision = BinaryPrecision().to(device)
    test_recall = BinaryRecall().to(device)
    test_f1 = BinaryF1Score().to(device)

    # Load the best model
    try:
        best_model_path = config.best_model_final_path
        checkpoint = torch.load(best_model_path, map_location=device,weights_only=False)
        
        # Load model state
        gnn_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Use the optimal threshold from training
        if 'optimal_threshold' in checkpoint:
            test_threshold = checkpoint['optimal_threshold']
            print(f"Using optimal threshold from training: {test_threshold:.3f}")
        else:
            test_threshold = final_threshold if 'final_threshold' in locals() else 0.5
            print(f"Using threshold: {test_threshold:.3f}")
        
        print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
        
    except FileNotFoundError:
        print("Best model not found, using current model state")
        test_threshold = final_threshold if 'final_threshold' in locals() else optimal_threshold

    # Evaluate on test set
    test_loss, test_report, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds , test_auc_results= eval_func_balanced(
        gnn_model, dataloader_test, device, 0, criterion, test_threshold
    )


    if len(test_labels) > 0 and len(test_preds) > 0:
            print("Confusion Matrix:")
            print(confusion_matrix(test_labels, test_preds))
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f}")
    print(
        f"Class 0 (Real) Acc: {test_report['class_0_accuracy']:.4f} | "
        f"Class 1 (Fake) Acc: {test_report['class_1_accuracy']:.4f} | "
        f"Class 0 (Real) Precision: {test_report['class_0_precision']:.4f} | "
        f"Class 1 (Fake) Precision: {test_report['class_1_precision']:.4f} | "
        f"Class 0 (Real) Recall: {test_report['class_0_recall']:.4f} | "
        f"Class 1 (Fake) Recall: {test_report['class_1_recall']:.4f} | "
        f"Class 0 (Real) F1: {test_report['class_0_f1_score']:.4f} | "
        f"Class 1 (Fake) F1: {test_report['class_1_f1_score']:.4f} | "
        f"ROC AUC: {test_report['roc_auc']:.4f} | PR AUC: {test_report['pr_auc']:.4f}"
    )
   

    wandb.finish()

    plot_loss_graph(train_losses, val_losses)
    plot_acc_graph(train_accs, val_accs)