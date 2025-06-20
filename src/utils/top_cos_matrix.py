import torch
import torch.nn.functional as F
def top5_cosine_matrix(cosine_sim_matrix):
    N = cosine_sim_matrix.size(0)
    mask = torch.eye(N, device=cosine_sim_matrix.device).bool()
    
    # Zero out diagonal (self similarity) so we can rank neighbors
    masked_sim = cosine_sim_matrix.masked_fill(mask, float('-inf'))
    
    # Compute average similarity per row (excluding self)
    avg_similarity = masked_sim.mean(dim=1)
    
    # Get indices of top 5 rows with highest avg similarity
    top5_rows = torch.topk(avg_similarity, 5).indices
    
    # For each selected row, get top 5 similarity values (excluding self)
    top5_matrix = []
    for i in top5_rows:
        sim_row = cosine_sim_matrix[i].clone()
        sim_row[i] = -float('inf')  # exclude self
        top5_vals = torch.topk(sim_row, 5).values
        top5_matrix.append(top5_vals)

    return torch.stack(top5_matrix)  # shape: (5, 5)