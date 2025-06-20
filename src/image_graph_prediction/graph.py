from torch_geometric.data import Data
import torch

def create_fully_connected_graph(embeddings, label):
    N = embeddings.size(0)
    if N == 1:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # self-loop
    else:
        row = torch.arange(N).repeat_interleave(N)
        col = torch.arange(N).repeat(N)
        mask = row != col
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
    return Data(x=embeddings, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float))
