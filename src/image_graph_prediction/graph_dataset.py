#%%
import os
import config
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

class MultimodalGraphDataset(Dataset):
    def __init__(self, df, img_embeddings, text_embeddings, transform=None, pre_transform=None):
        """
        Create Graph Dataset for Fake News Detection Task using PyTorch Geometric
        
        Args:
            df (pd.DataFrame): Main dataframe with labels and indices
            img_embeddings (list): List of image embedding tensors 
            text_embeddings (list): List of text embedding tensors
            transform: Optional transform to be applied on a sample
            pre_transform: Optional pre-transform to be applied on a sample
        """
        super(MultimodalGraphDataset, self).__init__(None, transform, pre_transform)
        
        self.df = df.reset_index(drop=True)
        self.img_embeddings = img_embeddings
        self.text_embeddings = text_embeddings
        
        # Projection for image embeddings 
        self.img_proj = nn.Linear(config.image_embed_size, 768)
        
        # Process embeddings once during initialization
        self._process_embeddings()
    
    def _process_embeddings(self):
        """Process embeddings to ensure consistent dimensions"""
        self.processed_text_embeddings = []
        self.processed_img_embeddings = []
        
        for i in range(len(self.img_embeddings)):
            # Convert to tensor if not already
            text_emb = torch.tensor(self.text_embeddings[i], dtype=torch.float32) if not isinstance(self.text_embeddings[i], torch.Tensor) else self.text_embeddings[i].float()
            img_emb = torch.tensor(self.img_embeddings[i], dtype=torch.float32) if not isinstance(self.img_embeddings[i], torch.Tensor) else self.img_embeddings[i].float()
            

                # Fallback if img_emb is empty
            if img_emb.numel() == 0 or img_emb.dim() == 0:
                img_emb = torch.randn(1, config.image_embed_size)  # random 2048 or 4096 dim
            img_emb_processed = self.img_proj(img_emb)
            self.processed_text_embeddings.append(text_emb)
            self.processed_img_embeddings.append(img_emb_processed)

    def len(self):
        return len(self.df)
    
    def get(self, idx):
        """
        Create a multimodal graph for a single sample
        
        Args:
            idx: Index of the sample
            
        Returns:
            torch_geometric.data.Data: Graph data object
        """
        # Get processed embeddings for this sample
        text_emb = self.processed_text_embeddings[idx]  
        img_emb = self.processed_img_embeddings[idx]    
        # print(img_emb.shape)
        # Get the number of objects for each modality
        n_text_objects = text_emb.shape[0]
        n_img_objects = img_emb.shape[0]
        
        # Option 1: Concatenate all nodes (text + image)
        # This creates a graph with (n_text_objects + n_img_objects) nodes
        node_features = torch.cat([text_emb, img_emb], dim=0)  # Shape: [total_objects, 768]
        num_nodes = node_features.shape[0]
        
        # Create node type indicators (0 for text, 1 for image)
        node_types = torch.cat([
            torch.zeros(n_text_objects, dtype=torch.long),  # Text nodes
            torch.ones(n_img_objects, dtype=torch.long)     # Image nodes
        ])
        
        # Create edges - you can customize this based on your needs
        edge_index = self._create_edges(n_text_objects, n_img_objects)
        
        # Get label
        label = self._get_label(idx)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.tensor([label], dtype=torch.float)
        )
        
        return data
    
    def _create_edges(self, n_text_objects, n_img_objects):
        """
        Create edges for the multimodal graph
        You can customize this based on your specific requirements
        """
        edges = []
        total_nodes = n_text_objects + n_img_objects
        
        # Option 1: Fully connected graph
        # for i in range(total_nodes):
        #     for j in range(total_nodes):
        #         if i != j:  # Exclude self-loops
        #             edges.append([i, j])
        
        # Option 2: Connect text nodes to image nodes (bipartite-like)
        # Text nodes are indexed 0 to n_text_objects-1
        # Image nodes are indexed n_text_objects to n_text_objects+n_img_objects-1
        for i in range(n_text_objects):
            for j in range(n_text_objects, n_text_objects + n_img_objects):
                edges.append([i, j])  # Text to image
                edges.append([j, i])  # Image to text
        
        # Option 3: Add intra-modality connections
        # Connect text nodes to each other
        for i in range(n_text_objects):
            for j in range(i+1, n_text_objects):
                edges.append([i, j])
                edges.append([j, i])
        
        # Connect image nodes to each other
        for i in range(n_text_objects, n_text_objects + n_img_objects):
            for j in range(i+1, n_text_objects + n_img_objects):
                edges.append([i, j])
                edges.append([j, i])
        
        # Add self-loops
        for i in range(total_nodes):
            edges.append([i, i])
        
        if not edges:  # Fallback: create self-loops if no edges
            edges = [[i, i] for i in range(total_nodes)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _get_label(self, idx):
        """Get label for the sample"""
        label_value = self.df['label'].iloc[idx]
        return int(label_value)


# def create_train_val_test_datasets(df, img_embeddings, text_embeddings, 
#                                  train_size=0.7, val_size=0.15, test_size=0.15, 
#                                  random_state=42):
#     """
#     Split the dataset into train, validation, and test sets
    
#     Args:
#         df: DataFrame with labels
#         img_embeddings: Image embeddings array/list
#         text_embeddings: Text embeddings array/list
#         train_size: Proportion for training set
#         val_size: Proportion for validation set
#         test_size: Proportion for test set
#         random_state: Random seed for reproducibility
        
#     Returns:
#         tuple: (train_dataset, val_dataset, test_dataset)
#     """
#     assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"

#     # First split: train and temp (val + test)
#     train_idx, temp_idx = train_test_split(
#         range(len(df)), 
#         test_size=(val_size + test_size),
#         random_state=random_state,
#         stratify=df['label'] if 'label' in df.columns else None
#     )
    
#     # Second split: val and test from temp
#     val_size_adjusted = val_size / (val_size + test_size)
#     val_idx, test_idx = train_test_split(
#         temp_idx,
#         test_size=(1 - val_size_adjusted),
#         random_state=random_state,
#         stratify=df.iloc[temp_idx]['label'] if 'label' in df.columns else None
#     )
    
#     # Create datasets for each split
#     train_df = df.iloc[train_idx].reset_index(drop=True)
#     val_df = df.iloc[val_idx].reset_index(drop=True)
#     test_df = df.iloc[test_idx].reset_index(drop=True)
    
#     # Split embeddings based on indices
#     train_img_emb = [img_embeddings[i] for i in train_idx]
#     val_img_emb = [img_embeddings[i] for i in val_idx]
#     test_img_emb = [img_embeddings[i] for i in test_idx]
    
#     train_text_emb = [text_embeddings[i] for i in train_idx]
#     val_text_emb = [text_embeddings[i] for i in val_idx]
#     test_text_emb = [text_embeddings[i] for i in test_idx]
    
#     # Create dataset objects
#     train_dataset = MultimodalGraphDataset(train_df, train_img_emb, train_text_emb)
#     val_dataset = MultimodalGraphDataset(val_df, val_img_emb, val_text_emb)
#     test_dataset = MultimodalGraphDataset(test_df, test_img_emb, test_text_emb)
    
#     return train_dataset, val_dataset, test_dataset


# def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
#     """
#     Create data loaders for training, validation, and testing
    
#     Args:
#         train_dataset: Training dataset
#         val_dataset: Validation dataset
#         test_dataset: Test dataset
#         batch_size: Batch size for data loaders
        
#     Returns:
#         tuple: (train_loader, val_loader, test_loader)
#     """
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
#     return train_loader, val_loader, test_loader

#%%
# df=pd.read_csv("datasets/processed/all_data_df_resolved.csv")
# img_embeddings_df = pd.read_pickle("src/resnet-embeddings/image_embeddings_3954.pkl")
# text_embeddings_df = pd.read_pickle("src/bert-text-embeddings/text_embeddings_3950.pkl")
# # df=pd.read_csv("../../datasets/processed/all_data_df_resolved.csv")
# # img_embeddings_df = pd.read_pickle("../resnet-embeddings/image_embeddings_3954.pkl")
# # text_embeddings_df = pd.read_pickle("../bert-text-embeddings/text_embeddings_3950.pkl")
# valid_df = df[df['resolved_text'].notnull()].reset_index(drop=True)

# # Step 2: Filter embeddings_df to match the valid rows
# # Assuming both dataframes are aligned row-wise (i.e., same original index)
# valid_img_embeddings_df = img_embeddings_df.loc[valid_df.index].reset_index(drop=True)
# valid_text_embeddings_df = text_embeddings_df.loc[valid_df.index].reset_index(drop=True)

# all_img_embeddings = valid_img_embeddings_df['referenced_image_embeddings'].values
# all_text_embeddings = valid_text_embeddings_df['text_embeddings'].values

# valid_df['label'] = valid_df['type'].map({'real': 0, 'fake': 1})

# # Your embeddings (make sure they're numpy arrays)
# all_img_embeddings = valid_img_embeddings_df['referenced_image_embeddings'].values
# all_text_embeddings = valid_text_embeddings_df['text_embeddings'].values
# print(all_text_embeddings[0].shape)

# all_img_embeddings = [torch.tensor(emb) for emb in all_img_embeddings]
# all_text_embeddings = [torch.tensor(emb) for emb in all_text_embeddings]

#%%
# Create train/val/test splits
# train_graphs, val_graphs, test_graphs = create_train_val_test_datasets(
#     valid_df, all_img_embeddings, all_text_embeddings
# )

# # # Create data loaders
# # train_loader, val_loader, test_loader = create_data_loaders(
# #     train_graphs, val_graphs, test_graphs, batch_size=32
# # )
# # Save datsets to disk if needed
# torch.save(train_graphs, 'multimodal_graphs/train_graphs.pt')
# torch.save(val_graphs, 'multimodal_graphs/val_graphs.pt')
# torch.save(test_graphs, 'multimodal_graphs/test_graphs.pt')

#%%
# org_df=pd.read_csv("../../datasets/raw/all_data.csv")
# org_df["type"].value_counts()
# Now you can use these with any PyTorch model and torchmetrics
# print(f"Train samples: {len(train_dataset)}")
# print(f"Val samples: {len(val_dataset)}")
# print(f"Test samples: {len(test_dataset)}")

# # Example of iterating through the data
# for batch in train_loader:
#     print(f"Batch node features shape: {batch.x.shape}")
#     print(f"Batch edge index shape: {batch.edge_index.shape}")
#     print(f"Batch labels shape: {batch.y.shape}")
#     break

# %%
