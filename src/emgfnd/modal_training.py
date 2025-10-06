## Importing libraries
from emgfnd.model_config import Config
from emgfnd.pgat_model import PGATClassifier
from emgfnd.utils import set_up_media_eval_dataset, set_up_all_data_dataset
from emgfnd.evaluation_utils import eval_func
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
import wandb

# Initialize wandb
wandb.init(project="multimodal-graph-classification", entity="", name="multimodal-experiment")

# =====================================================
# AUC METRICS CLASS
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

# =====================================================
# CLASS IMBALANCE HANDLING FUNCTIONS
# =====================================================

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
#  TRAINING FUNCTION
# =====================================================

def train_func_epoch(epoch, model, dataloader, device, optimizer, scheduler, criterion, 
                     optimal_threshold=0.5, handle_imbalance=False):
    """Training function for both balanced and imbalanced datasets
    
    Args:
        handle_imbalance: If True, includes class-specific metrics and uses optimal threshold
    """
    model.train()
    total_loss = 0
    valid_batches = 0
    
    # Initialize metrics
    train_acc = BinaryAccuracy().to(device)
    train_prec = BinaryPrecision().to(device)
    train_rec = BinaryRecall().to(device)
    train_f1 = BinaryF1Score().to(device)

    # Initialize AUC metrics if handling imbalance
    if handle_imbalance:
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
            
            #Detach all inputs to ensure no gradient tracking from previous iterations
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
            
            # Calculate loss
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
            
            # Calculate metrics
            with torch.no_grad():
                preds = torch.sigmoid(out.detach()).cpu()
                targets_cpu = targets.detach().cpu()

                # Update AUC metrics if handling imbalance (use raw probabilities, not binary)
                if handle_imbalance:
                    auc_metrics.update(preds, targets_cpu)
                
                # Use optimal threshold if handling imbalance, otherwise use 0.5
                threshold = optimal_threshold if handle_imbalance else 0.5
                preds_binary = (preds > threshold).float()
                
                # Store for detailed analysis
                all_preds.extend(preds_binary.numpy())
                all_targets.extend(targets_cpu.numpy())
                
                # Update metrics
                train_acc.update(preds_binary, targets_cpu)
                train_prec.update(preds_binary, targets_cpu)
                train_rec.update(preds_binary, targets_cpu)
                train_f1.update(preds_binary, targets_cpu)
                
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
                optimizer.zero_grad()
            
            continue
            
        except Exception as e:
            print(f"Unexpected error in batch {batch_idx}: {str(e)}")
            optimizer.zero_grad()
            continue
        
        finally:
            # Explicit cleanup after each batch
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
        train_report = {
            "accuracy": train_acc.compute().item(),
            "precision": train_prec.compute().item(),
            "recall": train_rec.compute().item(),
            "f1_score": train_f1.compute().item()
        }
        
        # Add class-specific metrics if handling imbalance
        if handle_imbalance:
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # Per-class metrics
            class_0_mask = all_targets == 0
            class_1_mask = all_targets == 1
            
            if class_0_mask.sum() > 0:
                class_0_acc = np.mean(all_preds[class_0_mask] == all_targets[class_0_mask])
                class_0_recall = np.mean(all_preds[class_0_mask] == 0)
            else:
                class_0_acc = class_0_recall = 0.0
                
            if class_1_mask.sum() > 0:
                class_1_acc = np.mean(all_preds[class_1_mask] == all_targets[class_1_mask])
                class_1_recall = np.mean(all_preds[class_1_mask] == 1)
            else:
                class_1_acc = class_1_recall = 0.0

            # Compute AUC metrics
            auc_results = auc_metrics.compute_metrics()
            
            train_report.update({
                "class_0_accuracy": class_0_acc,
                "class_1_accuracy": class_1_acc,
                "class_0_recall": class_0_recall,
                "class_1_recall": class_1_recall,
                "roc_auc": auc_results.get('roc_auc', 0.0),
                "pr_auc": auc_results.get('pr_auc', 0.0)
            })
        else:
            auc_results = {}
        
        avg_loss = total_loss / valid_batches
    else:
        train_report = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0
        }
        if handle_imbalance:
            train_report.update({
                "class_0_accuracy": 0.0, "class_1_accuracy": 0.0,
                "class_0_recall": 0.0, "class_1_recall": 0.0,
                "roc_auc": 0.0, "pr_auc": 0.0
            })
        avg_loss = float('inf')
        auc_results = {}
    
    return avg_loss, train_report, auc_results if handle_imbalance else (avg_loss, train_report)


# =====================================================
# MAIN TRAINING CODE
# =====================================================

# Configuration
config = Config()

if __name__ == '__main__':
    
    # =====================================================
    # CONFIGURATION: SET THESE PARAMETERS
    # =====================================================
    HANDLE_IMBALANCE = True  # Set to True for imbalanced datasets, False for balanced
    dataset_name = "all_data"  # "all_data" or "media_eval"
    
    ## Setup the dataset 
    set_seed(42) 

    if dataset_name == "all_data":  
        dataset_train, dataset_val, dataset_test = set_up_all_data_dataset()
    else:
        dataset_train, dataset_val, dataset_test = set_up_media_eval_dataset()
    
    # Print class distribution
    print("\n=== CLASS DISTRIBUTION ANALYSIS ===")
    train_labels = [data.y.item() for data in dataset_train]
    val_labels = [data.y.item() for data in dataset_val]
    print(f"Training set: {Counter(train_labels)}")
    print(f"Validation set: {Counter(val_labels)}")
    print(f"Test set: {len(dataset_test)} samples")
    
    # =====================================================
    # SETUP TRAINING COMPONENTS BASED ON IMBALANCE FLAG
    # =====================================================
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Handling imbalance: {HANDLE_IMBALANCE}")
    
    if HANDLE_IMBALANCE:
        print("\n=== SETTING UP FOR IMBALANCED DATASET ===")
        
        # Create weighted sampler for balanced batching
        print("\n=== SETTING UP WEIGHTED SAMPLER ===")
        weighted_sampler = create_weighted_sampler(dataset_train)
        
        # Setup weighted loss function
        print("\n=== SETTING UP WEIGHTED LOSS FUNCTION ===")
        # Choose your loss function (uncomment one):
        # Option A: Weighted BCE Loss
        # criterion = get_weighted_loss_function(dataset_train, device)
        
        # Option B: Focal Loss (alternative)
        criterion = FocalLoss(alpha=0.2, gamma=2.0)
        
        # Setup dataloader with weighted sampler
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            sampler=weighted_sampler,  # Use weighted sampler instead of shuffle
            drop_last=False,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0
        )
    else:
        print("\n=== SETTING UP FOR BALANCED DATASET ===")
        
        # Standard BCE loss for balanced datasets
        criterion = nn.BCEWithLogitsLoss()
        print("Using standard BCEWithLogitsLoss")
        
        # Setup dataloader with standard shuffling
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0
        )

    # Validation and test dataloaders (same for both cases)
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
    
    ## Initialize the model
    gnn_model = PGATClassifier()
    print("Total number of parameters:", sum(p.numel() for p in gnn_model.parameters()))
    gnn_model.to(device)
    
    ## Calculate number of train steps
    num_update_steps_per_epoch = math.ceil(len(dataloader_train) / config.gradient_accumulation_steps)
    num_train_steps = num_update_steps_per_epoch * config.epochs
    
    # Optimizer settings
    optimizer = AdamW(
        gnn_model.parameters(), 
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=10,
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
            break
                
        print(f"Successfully tested model")
        
    except Exception as e:
        print(f"Error in model testing: {e}")
        exit(1)

    best_loss = np.inf
    optimal_threshold = 0.5  # Will be updated if HANDLE_IMBALANCE is True
    best_acc = 0.0
    epochs_no_improve = 0
    early_stopping_patience = 20 if HANDLE_IMBALANCE else 3
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    def reinitialize_model_if_needed():
        """Reinitialize model if graph issues persist"""
        global gnn_model
        print("Reinitializing model to clear any retained state...")
        gnn_model = create_fresh_model()
        gnn_model.to(device)
        
        # Reinitialize optimizer as well
        optimizer = AdamW(
            gnn_model.parameters(), 
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        return optimizer
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\n=== STARTING TRAINING (IMBALANCE: {HANDLE_IMBALANCE}) ===")
    
    for epoch in range(config.epochs):
        print(f"\n---------------------- Epoch: {epoch+1} ---------------------------------- \n")
        
        # Clear cache before each epoch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            ## Training Loop
            if HANDLE_IMBALANCE:
                train_loss, train_report, train_auc_results = train_func_epoch(
                    epoch+1, gnn_model, dataloader_train, device, optimizer, scheduler, 
                    criterion, optimal_threshold, handle_imbalance=True
                )
            else:
                train_loss, train_report = train_func_epoch(
                    epoch+1, gnn_model, dataloader_train, device, optimizer, scheduler, 
                    criterion, handle_imbalance=False
                )
                train_auc_results = {}
            
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

        ## Validation loop
        val_loss, report, acc, prec, rec, f1_score, val_labels, val_preds, val_auc_results = eval_func(
            gnn_model, dataloader_val, device, epoch+1, criterion, 
            optimal_threshold if HANDLE_IMBALANCE else 0.5
        )

        # Log AUC curves to wandb if handling imbalance
        if HANDLE_IMBALANCE:
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
        if HANDLE_IMBALANCE:
            print(f"Class 0 (Real) Acc: {train_report['class_0_accuracy']:.4f} | Class 1 (Fake) Acc: {train_report['class_1_accuracy']:.4f}")
        print()
        print("Validation Report:")
        print(f"Accuracy: {report['accuracy']:.4f} | Precision: {report['precision']:.4f} | Recall: {report['recall']:.4f} | F1: {report['f1_score']:.4f}")

        # Update optimal threshold periodically if handling imbalance
        if HANDLE_IMBALANCE and epoch > 0 and epoch % 3 == 0:
            try:
                new_threshold = find_optimal_threshold(gnn_model, dataloader_val, device)
                optimal_threshold = new_threshold
                print(f"Updated optimal threshold to: {optimal_threshold:.3f}")
            except Exception as e:
                print(f"Could not update threshold: {e}")
        
        # Build wandb log dict
        wandb_log = {
            "train_loss": train_loss, 
            "train-acc": train_report["accuracy"],
            "train-prec": train_report["precision"],
            "train-rec": train_report["recall"],
            "train-f1": train_report["f1_score"],
            "val-loss": val_loss, 
            "val-prec": prec, 
            "val-rec": rec, 
            "val-f1score": f1_score, 
            "val-acc": report["accuracy"],
            "Loss/Train": train_loss,
            "Loss/Validation": val_loss,
            "Accuracy/Train": train_report["accuracy"],
            "Accuracy/Validation": report["accuracy"],
            "epoch": epoch + 1
        }
        
        # Add imbalance-specific metrics if applicable
        if HANDLE_IMBALANCE:
            wandb_log.update({
                "train-class0-acc": train_report["class_0_accuracy"],
                "train-class1-acc": train_report["class_1_accuracy"],
                "optimal_threshold": optimal_threshold
            })
        
        # Only log to wandb if values are valid
        if not any([math.isnan(x) or math.isinf(x) for x in [train_loss, val_loss, acc, prec, rec, f1_score]]):
            wandb.log(wandb_log)
        
        print(f"\n----------------------------------------------------------------------------")

        # Early stopping check
        if report["accuracy"] > best_acc:
            best_acc = report["accuracy"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1    

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best Acc: {best_acc:.4f}")
            break

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_dict = {
                'model_state_dict': gnn_model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }
            if HANDLE_IMBALANCE:
                save_dict['optimal_threshold'] = optimal_threshold
                save_dict['class_weights'] = calculate_class_weights(dataset_train)
            
            torch.save(save_dict, config.best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_report["accuracy"])
        val_accs.append(report["accuracy"])

    print("Training completed!")

    # Final threshold optimization for imbalanced datasets
    if HANDLE_IMBALANCE:
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
        best_model_path = config.best_model_final_path if HANDLE_IMBALANCE else config.best_model_path
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        
        # Load model state
        gnn_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Use the optimal threshold from training if available
        if HANDLE_IMBALANCE and 'optimal_threshold' in checkpoint:
            test_threshold = checkpoint['optimal_threshold']
            print(f"Using optimal threshold from training: {test_threshold:.3f}")
        elif HANDLE_IMBALANCE and 'final_threshold' in locals():
            test_threshold = final_threshold
            print(f"Using final threshold: {test_threshold:.3f}")
        else:
            test_threshold = 0.5
            print(f"Using threshold: {test_threshold:.3f}")
        
        print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
        
    except FileNotFoundError:
        print("Best model not found, using current model state")
        test_threshold = 0.5

    # Evaluate on test set
    test_loss, test_report, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds, test_auc_results = eval_func(
        gnn_model, dataloader_test, device, 0, criterion, test_threshold
    )

    if len(test_labels) > 0 and len(test_preds) > 0:
        print("Test Confusion Matrix:")
        print(confusion_matrix(test_labels, test_preds))
        
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f}")
    
    if HANDLE_IMBALANCE:
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