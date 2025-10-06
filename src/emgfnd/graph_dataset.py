#%%
import config
import torch
from torch import nn
from torch_geometric.data import Data, Dataset

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
                img_emb = torch.randn(1, config.image_embed_size)

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
       
        # Get the number of objects for each modality
        n_text_objects = text_emb.shape[0]
        n_img_objects = img_emb.shape[0]
        
        # This creates a graph with (n_text_objects + n_img_objects) nodes
        node_features = torch.cat([text_emb, img_emb], dim=0)  # Shape: [total_objects, 768]

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
        
        for i in range(n_text_objects):
            for j in range(n_text_objects, n_text_objects + n_img_objects):
                edges.append([i, j])  # Text to image
                edges.append([j, i])  # Image to text
        

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