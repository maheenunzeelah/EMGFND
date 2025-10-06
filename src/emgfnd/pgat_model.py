from emgfnd.model_config import Config
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, GATConv, global_max_pool, global_add_pool
from torch_geometric.data import Data, DataLoader

# class CosineAttentionGAT(MessagePassing):
#     def __init__(self, in_channels, out_channels):
#         super().__init__(aggr='add')  # Aggregate messages by summation
#         self.linear = nn.Linear(in_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = self.linear(x)
#         return self.propagate(edge_index, x=x)

#     def message(self, x_i, x_j):
#         # Compute cosine similarity and normalize it to [0, 1]
#         cos_attn = F.cosine_similarity(x_i, x_j, dim=1).unsqueeze(1)  # Shape: [E, 1]
#         attn_weight = (cos_attn + 1) / 2  # Normalize to [0, 1]
#         return cos_attn * x_j

# class CosineAttentionGATClassifier(nn.Module):
#     def __init__(self, in_channels=2048, hidden_channels=256):
#         super().__init__()
#         self.gat1 = CosineAttentionGAT(in_channels, hidden_channels)
#         self.gat2 = CosineAttentionGAT(hidden_channels, hidden_channels)
#         self.dropout = nn.Dropout(0.3)
#         self.classifier = nn.Linear(hidden_channels, 1)  # Binary classification

#     def forward(self, x, edge_index, batch):
#         x = F.elu(self.gat1(x, edge_index))
#         x = F.elu(self.gat2(x, edge_index))
#         x = self.dropout(x)
#         x = global_mean_pool(x, batch)  # Shape: [batch_size, hidden_channels]
#         x = self.classifier(x).view(-1)  # Shape: [batch_size]
#         return x  # Use sigmoid + BCEWithLogitsLoss during training
    
# class CosineAttentionGATConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, dropout=0.0):
#         super().__init__(aggr='add')
#         self.linear = nn.Linear(in_channels, out_channels)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, edge_index):
#         x = self.linear(x)
#         return self.propagate(edge_index, x=x)

#     def message(self, x_i, x_j):
#         # Cosine similarity as attention
#         cos_attn = F.cosine_similarity(x_i, x_j, dim=1).unsqueeze(1)  # [E, 1]
#         attn_weight = (cos_attn + 1) / 2  # Normalize to [0, 1]
#         attn_weight = self.dropout(attn_weight)
#         return attn_weight * x_j
    
    
class PGATClassifier(nn.Module):
    def __init__(self, in_dim=768, proj_dim=768, gat_dim=256, num_classes=1, dropout=Config.dropout):
        super().__init__()
        
        # Projection layer (like mm_embedding_space)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # GAT layers with proper normalization
        self.gat1 = GATConv(proj_dim, gat_dim, heads=4, dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(gat_dim * 4)
        
        self.gat2 = GATConv(gat_dim * 4, gat_dim, heads=4, dropout=dropout, concat=True)
        self.norm2 = nn.LayerNorm(gat_dim * 4)

        self.gat3 = GATConv(gat_dim * 4, gat_dim, heads=1, dropout=dropout, concat=False)
        self.norm3 = nn.LayerNorm(gat_dim)
        
        # Classifier with corrected input size
        # After pooling: 3 pooling methods * gat_dim = 3 * 256 = 768
        self.classifier = nn.Sequential(
            nn.Linear(gat_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights properly
        self._init_weights()
      
    def _init_weights(self):
        """Initialize weights to prevent gradient issues"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, batch):
        # Clear any potential cached computations
        if hasattr(self, '_cached_edge_index'):
            delattr(self, '_cached_edge_index')
            
        # Project input features
        x = self.proj(x)  # [num_nodes, proj_dim]
        
        # First GAT layer
        x1 = self.gat1(x, edge_index)  # [num_nodes, gat_dim * 4]
        x1 = self.norm1(x1)
        x1 = F.elu(x1)
        
        # Second GAT layer with residual connection
        x2 = self.gat2(x1, edge_index)  # [num_nodes, gat_dim * 4]
        x2 = self.norm2(x2)
        x2 = F.elu(x2)
        x2 = x2 + x1  # Residual connection (same dimensions)
        
        # Third GAT layer
        x3 = self.gat3(x2, edge_index)  # [num_nodes, gat_dim]
        x3 = self.norm3(x3)
        x3 = F.elu(x3)
        
        # Global pooling - concatenate different pooling methods
        pooled = torch.cat([
            global_mean_pool(x3, batch),  # [batch_size, gat_dim]
            global_max_pool(x3, batch),   # [batch_size, gat_dim]
            global_add_pool(x3, batch)    # [batch_size, gat_dim]
        ], dim=1)  # [batch_size, gat_dim * 3]
        
        # Classification
        x = self.classifier(pooled)
        
        # Return as 1D tensor for BCE loss
        return x.view(-1)
# class PGATClassifier(nn.Module):
#     def __init__(self, in_dim=768, proj_dim=768, gat_dim=256, num_classes=1, dropout=0.2):
#         super().__init__()
        
#         # Projection layer (like mm_embedding_space)
#         self.proj = nn.Sequential(
#             nn.Linear(in_dim, proj_dim),
#             nn.ELU(),
#             nn.Dropout(dropout)
#         )
        
#         # GAT layers with proper normalization
#         self.gat1 = GATConv(proj_dim, gat_dim, heads=4, dropout=dropout, concat=True)
#         self.norm1 = nn.LayerNorm(gat_dim*4)
        
#         self.gat2 = GATConv(gat_dim * 4, gat_dim, heads=4, dropout=dropout, concat=True)
#         self.norm2 = nn.LayerNorm(gat_dim*4)

#         # self.gat3 = GATConv(gat_dim, gat_dim, heads=1, dropout=dropout, concat=False)
#         # self.norm3 = nn.LayerNorm(gat_dim)
        
#         # Classifier with proper initialization
#         self.classifier = nn.Sequential(
#             nn.Linear(gat_dim * 3 * 4, 128),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, num_classes)
#         )
        
#         # Initialize weights properly
#         self._init_weights()
      
#     def _init_weights(self):
#         """Initialize weights to prevent gradient issues"""
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)
    
#     def forward(self, x, edge_index, batch):
#         # Clear any potential cached computations
#         if hasattr(self, '_cached_edge_index'):
#             delattr(self, '_cached_edge_index')
            
#         # Project input features
#         x = self.proj(x)
        
#         # First GAT layer with residual connection and normalization
#         x_residual = x
#         x = self.gat1(x, edge_index)
#         x = self.norm1(x)
#         x = F.elu(x)
        
#         # Add residual connection if dimensions match
#         if x.size(-1) == x_residual.size(-1):
#             x = x + x_residual
        
#         # Second GAT layer with residual connection and normalization
#         x_residual = x
#         x = self.gat2(x, edge_index)
#         x = self.norm2(x)
#         x = F.elu(x)
#         x = x + x_residual  # Same dimensions, so we can add residual

#         # # Third GAT layer with residual connection and normalization
#         # x_residual = x
#         # x = self.gat3(x, edge_index)
#         # x = self.norm3(x)
#         # x = F.elu(x)
#         # x = x + x_residual  # Same dimensions, so we can add residual
        
#         # Global pooling
#         x = torch.cat([
#             global_mean_pool(x, batch),
#             global_max_pool(x, batch),
#             global_add_pool(x, batch)
#         ], dim=1)
        
#         # Classification
#         x = self.classifier(x)
        
#         # Return as 1D tensor for BCE loss
#         return x.view(-1)
