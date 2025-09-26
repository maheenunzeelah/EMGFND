## Importing libraries
from image_graph_prediction.evaluation_utils import eval_func
from image_graph_prediction.model_config import Config
from image_graph_prediction.model import PGATClassifier
from image_graph_prediction.utils import set_up_media_eval_dataset, set_up_multimodal_dataset
import numpy as np

import math


import torch
from torch import nn

from torch.optim import AdamW

from torch_geometric.loader import DataLoader


from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc
import wandb
from image_graph_prediction.utils import plot_acc_graph, plot_loss_graph


# Initialize wandb
wandb.init(project="multimodal-graph-classification", entity="", name="multimodal-experiment")


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
# TRAINING FUNCTION
# =====================================================

def train_func_epoch(epoch, model, dataloader, device, optimizer, scheduler, criterion):
    """Training function for balanced datasets"""
    model.train()
    total_loss = 0
    valid_batches = 0
    
    # Initialize metrics
    train_acc = BinaryAccuracy().to(device)
    train_prec = BinaryPrecision().to(device)
    train_rec = BinaryRecall().to(device)
    train_f1 = BinaryF1Score().to(device)
    
    # Track predictions
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
            
            # Detach all inputs to ensure no gradient tracking from previous iterations
            x_input = batch.x.detach().requires_grad_(True)
            edge_index_input = batch.edge_index.detach()
            batch_input = batch.batch.detach() if hasattr(batch, 'batch') else None
            
            # Forward pass
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
                
                preds_binary = (preds > 0.5).float()
                
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
    
    # Calculate final metrics
    if valid_batches > 0:
        train_report = {
            "accuracy": train_acc.compute().item(),
            "precision": train_prec.compute().item(),
            "recall": train_rec.compute().item(),
            "f1_score": train_f1.compute().item()
        }
        avg_loss = total_loss / valid_batches
    else:
        train_report = {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0
        }
        avg_loss = float('inf')
    
    return avg_loss, train_report


# =====================================================
# MAIN TRAINING CODE
# =====================================================

# Configuration
config = Config()

if __name__ == '__main__':
    
    ## Setup the dataset 
    dataset_name = "media_eval"
    set_seed(42) 

    if dataset_name == "multimodal":  
        dataset_train, dataset_val, dataset_test = set_up_multimodal_dataset()
    else:
        dataset_train, dataset_val, dataset_test = set_up_media_eval_dataset()
    
    # Print dataset sizes
    print(f"\nDataset sizes:")
    print(f"Training set: {len(dataset_train)} samples")
    print(f"Validation set: {len(dataset_val)} samples")
    print(f"Test set: {len(dataset_test)} samples")
    
    # =====================================================
    # SETUP TRAINING COMPONENTS
    # =====================================================
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Standard BCE loss for balanced datasets
    criterion = nn.BCEWithLogitsLoss()
    print("Using standard BCEWithLogitsLoss")
    
    ## Setup the dataloaders
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        shuffle=True,
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

    # Test the model before training
    print("Testing model with sample batch...")
    try:
        for test_batch in dataloader_train:
            test_batch = test_batch.to(device)
            
            # Inference test
            gnn_model.eval()
            with torch.no_grad():
                _ = gnn_model(test_batch.x, test_batch.edge_index, test_batch.batch)

            # Training test
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
                
        print("Model test successful!")
        
    except Exception as e:
        print(f"Error in model testing: {e}")
        exit(1)

    best_loss = np.inf
    best_acc = 0.0
    epochs_no_improve = 0
    early_stopping_patience = 3
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
    
    print("\n=== STARTING TRAINING ===")
    
    for epoch in range(config.epochs):
        print(f"\n---------------------- Epoch: {epoch+1} ---------------------------------- \n")
        
        # Clear cache before each epoch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            ## Training Loop
            train_loss, train_report = train_func_epoch(
                epoch+1, gnn_model, dataloader_train, device, optimizer, scheduler, criterion
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

        ## Validation loop
        val_loss, report, acc, prec, rec, f1_score, val_labels, val_preds, val_auc_results= eval_func(
            gnn_model, dataloader_val, device, epoch+1, criterion
        )

        
        # Print confusion matrix if we have predictions
        if len(val_labels) > 0 and len(val_preds) > 0:
            print("Confusion Matrix:")
            print(confusion_matrix(val_labels, val_preds))
        
        print(f"\nEpoch: {epoch+1} | Training loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print()
        print("Train Report:")
        print(f"Accuracy: {train_report['accuracy']:.4f} | Precision: {train_report['precision']:.4f} | Recall: {train_report['recall']:.4f} | F1: {train_report['f1_score']:.4f}")
        print()
        print("Validation Report:")
        print(f"Accuracy: {report['accuracy']:.4f} | Precision: {report['precision']:.4f} | Recall: {report['recall']:.4f} | F1: {report['f1_score']:.4f}")
        
        # Log to wandb if values are valid
        if not any([math.isnan(x) or math.isinf(x) for x in [train_loss, val_loss, acc, prec, rec, f1_score]]):
            wandb.log({
                "train_loss": train_loss, 
                "train-acc": train_report["accuracy"],
                "train-prec": train_report["precision"],
                "train-rec": train_report["recall"],
                "train-f1": train_report["f1_score"],
                "val-loss": val_loss, 
                "val-prec": prec, 
                "val-rec": rec, 
                "val-f1score": f1_score, 
                "val-acc": report["accuracy"]
            })
        
        print(f"\n----------------------------------------------------------------------------")
        
        # Early stopping based on accuracy
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
            torch.save({
                'model_state_dict': gnn_model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, config.best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accs.append(train_report["accuracy"])
        val_accs.append(report["accuracy"])    

    print("Training completed!")
    
    print("\n=== EVALUATING ON TEST DATASET ===")

    # Initialize torchmetrics on the correct device
    test_accuracy = BinaryAccuracy().to(device)
    test_precision = BinaryPrecision().to(device)
    test_recall = BinaryRecall().to(device)
    test_f1 = BinaryF1Score().to(device)

    # Load the best model
    try:
        best_model_path = config.best_model_path
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        
        # Load model state
        gnn_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
        
    except FileNotFoundError:
        print("Best model not found, using current model state")

    # Evaluate on test set
    test_loss, test_report, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds , test_auc_results = eval_func(
        gnn_model, dataloader_test, device, 0, criterion
    )

    if len(test_labels) > 0 and len(test_preds) > 0:
        print("Test Confusion Matrix:")
        print(confusion_matrix(test_labels, test_preds))
        
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} | Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f}")
    # print(f"Test Report: {test_report}")
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