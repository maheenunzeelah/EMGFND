import torch
import numpy as np
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, auc

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

def eval_func(model, dataloader, device, epoch, criterion, optimal_threshold=0.5):
    """Evaluation function for balanced datasets"""
    model.eval()
    print(model.training)
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
                    
                    # Calculate metrics
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
