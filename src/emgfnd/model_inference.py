from emgfnd.evaluation_utils import eval_func
import os, random, numpy as np, torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)     # harmless if you only use CPU
torch.use_deterministic_algorithms(True)
# optional: restrict parallelism to avoid non-deterministic reductions
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
# ---- End reproducibility settings ----

# then the rest of your imports
from emgfnd.model_config import Config
from emgfnd.utils import set_up_media_eval_dataset, set_up_all_data_dataset
import torch
from torchmetrics.classification import BinaryF1Score
from torch_geometric.loader import DataLoader
from torch import nn
import emgfnd
from emgfnd.pgat_model import PGATClassifier
from emgfnd.graph_dataset import MultimodalGraphDataset
import sys
sys.modules['image_graph_prediction'] = emgfnd

torch.serialization.add_safe_globals([MultimodalGraphDataset])

def evaluate_model_on_test_set():
  config = Config()
  device = "cpu"

  print(f"Using device: {device}")


  dataset_train, dataset_val, dataset_test = set_up_media_eval_dataset()
  # dataset_test = torch.load('test_datasets/all_data_clip_text_dataset_test.pt', weights_only=False)
  torch.save(dataset_test, 'test_datasets/media_eval_clip_title_dataset_test.pt')


  dataloader_test = DataLoader(
    dataset_test,
    batch_size=config.batch_size,
    drop_last=False,
    shuffle=False,
    pin_memory= True if torch.cuda.is_available() else False,
    num_workers=0
  )
  
  print("\n=== EVALUATING ON TEST DATASET ===")
  torch.set_deterministic_debug_mode("warn")
  gnn_model = PGATClassifier()
  print("Total number of parameters:", sum(p.numel() for p in gnn_model.parameters()))
  gnn_model.to(device)
  # Initialize torchmetrics on the correct device

  test_f1 = BinaryF1Score().to(device)
  criterion = nn.BCEWithLogitsLoss()
   
  # Load the best model
  try:
      best_model_path = 'best_models/best_media_eval_model_clip_title.pth'
      # best_model_path ='best_models/final_all_data_model_clip_text_final.pth'
      checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
      
      # Load model state
      gnn_model.load_state_dict(checkpoint['model_state_dict'])
      # Use the optimal threshold from training
      test_threshold = 0.5
      if 'optimal_threshold' in checkpoint:
          test_threshold = checkpoint['optimal_threshold']
          print(f"Using optimal threshold from training: {test_threshold:.3f}")

      
      print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
      
  except FileNotFoundError:
      print("Best model not found, using current model state")
  print("Model training flag:", next(gnn_model.modules()).training)
  # Evaluate on test set
  test_loss, test_report, test_acc, test_prec, test_rec, test_f1, test_labels, test_preds , test_auc_results = eval_func(
      gnn_model, dataloader_test, device, 0, criterion, test_threshold
  )
      
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

  
evaluate_model_on_test_set()
