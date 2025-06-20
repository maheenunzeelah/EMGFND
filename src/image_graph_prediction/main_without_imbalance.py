## Importing libraries
from image_graph_prediction.config import Config
from image_graph_prediction.model import PGATClassifier
from image_graph_prediction.utils import set_up_multimodal_dataset
import numpy as np
import pandas as pd
import math
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

import torchmetrics
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score

from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix
import wandb

# Initialize wandb
wandb.init(project="multimodal-graph-classification", entity="", name="multimodal-experiment")


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


def train_func_epoch(epoch, model, dataloader, device, optimizer, scheduler, criterion):
    """Training function for one epoch with proper graph management"""
    model.train()
    total_loss = 0
    valid_batches = 0
    
    # Initialize metrics once per epoch
    train_acc = BinaryAccuracy().to(device)
    train_prec = BinaryPrecision().to(device)
    train_rec = BinaryRecall().to(device)
    train_f1 = BinaryF1Score().to(device)
   
    
    # criterion = nn.BCEWithLogitsLoss()
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # CRITICAL: Always zero gradients FIRST and clear any retained graphs
            optimizer.zero_grad()
            
            # Move batch to device
            batch = batch.to(device)
            
            # Validate batch data
            if check_for_nan_inf(batch.x, f"batch.x at batch {batch_idx}"):
                print(f"Skipping batch {batch_idx} due to NaN/Inf in input features")
                continue
            
            # Ensure edge_index is properly formatted
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
            
            # Validate model output
            if check_for_nan_inf(out, f"model output at batch {batch_idx}"):
                print(f"Skipping batch {batch_idx} due to NaN/Inf in model output")
                continue
            
            # Prepare targets - ensure proper shape and type
            targets = batch.y.float().detach()
            if targets.dim() > 1:
                targets = targets.view(-1)
            
            # Validate targets
            if check_for_nan_inf(targets, f"targets at batch {batch_idx}"):
                print(f"Skipping batch {batch_idx} due to NaN/Inf in targets")
                continue
            
            # Ensure output and targets have the same shape
            if out.shape != targets.shape:
                print(f"Shape mismatch at batch {batch_idx}: out {out.shape}, targets {targets.shape}")
                continue
            
            # Calculate loss
            loss = criterion(out, targets)
            
            # Validate loss
            if check_for_nan_inf(loss, f"loss at batch {batch_idx}"):
                print(f"Skipping batch {batch_idx} due to NaN/Inf in loss")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping after backward pass
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check for gradient issues
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Skipping batch {batch_idx} due to NaN/Inf gradients (norm: {grad_norm})")
                optimizer.zero_grad()  # Clear bad gradients
                continue
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            valid_batches += 1
            
            # Calculate metrics with proper detachment
            with torch.no_grad():
                # Detach and move to CPU immediately
                preds = torch.sigmoid(out.detach()).cpu()
                targets_cpu = targets.detach().cpu()
                
                # Clamp predictions to valid range
                preds = torch.clamp(preds, 0.0, 1.0)
                
                # Update metrics
                train_acc.update(preds, targets_cpu)
                train_prec.update(preds, targets_cpu)
                train_rec.update(preds, targets_cpu)
                train_f1.update(preds, targets_cpu)
            
            # Print progress
            # if batch_idx % 10 == 0:
            #     print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Grad Norm: {grad_norm:.4f}")
            
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
    
    # Compute final metrics
    if valid_batches > 0:
        train_report = {
            "accuracy": train_acc.compute().item(),
            "precision": train_prec.compute().item(),
            "recall": train_rec.compute().item(),
            "f1_score": train_f1.compute().item()
        }
        avg_loss = total_loss / valid_batches
        print(f"Processed {valid_batches}/{len(dataloader)} valid batches")
    else:
        print("WARNING: No valid batches processed!")
        train_report = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        avg_loss = float('inf')
    
    return avg_loss, train_report
            

def eval_func(model, dataloader, device, criterion, epoch):
    """Evaluation function with improved error handling"""
    model.eval()
    total_loss = 0
    valid_batches = 0
    
    # Initialize metrics
    val_acc = BinaryAccuracy().to(device)
    val_prec = BinaryPrecision().to(device)
    val_rec = BinaryRecall().to(device)
    val_f1 = BinaryF1Score().to(device)

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
                    
                    # Calculate metrics
                    preds = torch.sigmoid(out).cpu()
                    targets_cpu = targets.cpu()
                    
                    preds = torch.clamp(preds, 0.0, 1.0)

                    all_preds.append((preds > 0.5).long())
                    all_targets.append(targets_cpu.long())
                    
                    val_acc.update(preds, targets_cpu)
                    val_prec.update(preds, targets_cpu)
                    val_rec.update(preds, targets_cpu)
                    val_f1.update(preds, targets_cpu)
                    
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
    
    report = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1_score
    }
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    return avg_loss, report, acc, prec, rec, f1_score,  all_targets, all_preds


# Configuration
config = Config()

if __name__ == '__main__':
    
    ## Setup the dataset 
    dataset_name = "multimodal"
    set_seed(42) 

    if dataset_name == "multimodal":  
        dataset_train, dataset_val, dataset_test = set_up_multimodal_dataset()
    else:
        print("No Data")
    
    ## Setup the dataloaders
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=config.batch_size,
        drop_last=False,
        shuffle=False,  # Don't shuffle validation
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
        lr=config.lr * 0.1,  # Reduce learning rate
        weight_decay=1e-4,   # Slightly less weight decay
        eps=1e-8             # Epsilon for numerical stability
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
            
            # Forward pass
            gnn_model.eval()
            with torch.no_grad():
                test_out = gnn_model(test_batch.x, test_batch.edge_index, test_batch.batch)
            
            # Backward pass
            gnn_model.train()
            optimizer.zero_grad()
            test_out = gnn_model(test_batch.x, test_batch.edge_index, test_batch.batch)
            test_targets = test_batch.y.float()
            if test_targets.dim() > 1:
                test_targets = test_targets.view(-1)
            
            test_loss = nn.BCEWithLogitsLoss()(test_out, test_targets)
            test_loss.backward()
            optimizer.zero_grad()
            
            # Clear everything
            del test_out, test_targets, test_loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            test_batches += 1
            if test_batches >= 3:  # Test first 3 batches
                break
                
        print(f"Successfully tested {test_batches} batches")
        
    except Exception as e:
        print(f"Error in model testing: {e}")
        exit(1)

    best_loss = np.inf
    
    
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
                weight_decay=1e-4,
                # eps=1e-8
            )
            return optimizer
        
        # Add error recovery in your epoch loop:
    consecutive_errors = 0
    max_consecutive_errors = 3

     # Calculate positive and negative samples for class weighting
    num_pos = sum(1 for batch in dataloader_train for y in batch.y if y == 0)
    num_neg = sum(1 for batch in dataloader_train for y in batch.y if y == 1)
    
    pos_weight = torch.tensor([num_pos / num_neg], dtype=torch.float).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    for epoch in range(config.epochs):
        print(f"\n---------------------- Epoch: {epoch+1} ---------------------------------- \n")
        
        # Clear cache before each epoch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            ## Training Loop
            train_loss, train_report = train_func_epoch(epoch+1, gnn_model, dataloader_train, device, optimizer, scheduler, criterion)
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
        val_loss, report, acc, prec, rec, f1_score, val_labels, val_preds = eval_func(gnn_model, dataloader_val, device, criterion, epoch+1)
        print(confusion_matrix(val_labels, val_preds))
        print(f"\nEpoch: {epoch+1} | Training loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        print()
        print("Train Report:")
        print(f"Accuracy: {train_report['accuracy']:.4f} | Precision: {train_report['precision']:.4f} | Recall: {train_report['recall']:.4f} | F1: {train_report['f1_score']:.4f}")
        print()
        print("Validation Report:")
        print(f"Accuracy: {report['accuracy']:.4f} | Precision: {report['precision']:.4f} | Recall: {report['recall']:.4f} | F1: {report['f1_score']:.4f}")
        
        # Only log to wandb if values are valid
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
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(gnn_model.state_dict(), "best_multimodal_model.pth")
            print(f"New best model saved with validation loss: {val_loss:.4f}")

    print("Training completed!")
    wandb.finish()
