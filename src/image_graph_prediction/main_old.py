from image_graph_prediction.model import CosineAttentionGATClassifier, PGATClassifier
import torch
from torch_geometric.data import DataLoader
import torch.nn as nn
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt

# from image_graph_prediction.graph_dataset import create_data_loaders

train_losses = []
val_losses = []
train_accs = []
val_accs = []
train_f1s = []
val_f1s = []

train_graphs = torch.load('src/image_graph_prediction/multimodal_graphs/train_graphs.pt', weights_only=False)
val_graphs = torch.load('src/image_graph_prediction/multimodal_graphs/val_graphs.pt', weights_only=False)
test_graphs = torch.load('src/image_graph_prediction/multimodal_graphs/test_graphs.pt', weights_only=False)

class_sample_counts = [2083, 763]
weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
train_targets = torch.tensor([g.y.item() for g in train_graphs], dtype=torch.long)
sample_weights = weights[train_targets]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# train_loader = DataLoader(train_graphs, batch_size=8, sampler=sampler)
train_loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=8)
test_loader = DataLoader(test_graphs, batch_size=8)
# train_loader, val_loader, test_loader = create_data_loaders(
#     train_graphs, val_graphs, test_graphs, batch_size=32
# )


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PGATClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

train_labels = [graph.y.item() for graph in train_graphs]

# Count class distribution
label_counts = Counter(train_labels)
neg_count = label_counts[1.0]  # Class 1 = negative
pos_count = label_counts[0.0]  # Class 0 = positive

pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def train(model, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = (torch.sigmoid(out) > 0.5).long()
        # y_preds_int = [int(p.item()) for p in preds]
        # print("Predicted label distribution:", Counter(y_preds_int))    
        all_preds.append(preds.cpu())
        all_labels.append(batch.y.cpu().long())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    

    from sklearn.metrics import accuracy_score, f1_score
    return {
        'acc': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds)
    }

@torch.no_grad()
def predict(model, loader):
    model.eval()
    all_logits = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.batch)
        all_logits.append(torch.sigmoid(logits).cpu())  # probs between 0 and 1
    return torch.cat(all_logits)

# ------------------- Training Loop -------------------
num_epochs = 40

for epoch in range(num_epochs):
    train_loss = train(model, train_loader)
    train_metrics = evaluate(model, train_loader)
    val_metrics = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(None)  # If you want to compute val loss, add code for it
    train_accs.append(train_metrics['acc'])
    val_accs.append(val_metrics['acc'])
    train_f1s.append(train_metrics['f1'])
    val_f1s.append(val_metrics['f1'])

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['acc']:.4f} "
          f"Val Acc: {val_metrics['acc']:.4f} | Val F1: {val_metrics['f1']:.4f}")
    scheduler.step(val_metrics['acc'])

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_accs, label='Train Acc')
plt.plot(epochs, val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

plt.subplot(1, 2, 2)
plt.plot(epochs, train_f1s, label='Train F1')
plt.plot(epochs, val_f1s, label='Val F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.title('F1 Score Curve')

plt.tight_layout()
plt.show()

# ------------------- Test Evaluation -------------------
test_metrics = evaluate(model, test_loader)
print(f"Test Accuracy: {test_metrics['acc']:.4f} | Test F1: {test_metrics['f1']:.4f}")

# ------------------- Prediction Example -------------------
# probs = predict(model, test_loader)
# print("Predicted probabilities:", probs)

# ------------------- Save Model -------------------
# torch.save(model.state_dict(), "gat_classifier.pth")
# from collections import Counter
# print(Counter([g.y.item() for g in train_graphs]))
# print(Counter([g.y.item() for g in val_graphs]))
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# x = torch.cat([g.x for g in val_graphs], dim=0).cpu().numpy()
# labels = [g.y.item() for g in val_graphs for _ in range(g.num_nodes)]

# pca = PCA(n_components=2)
# x_pca = pca.fit_transform(x)
# plt.scatter(x_pca[:,0], x_pca[:,1], c=labels, cmap='coolwarm')
# plt.title("PCA of Node Features")
# plt.show()