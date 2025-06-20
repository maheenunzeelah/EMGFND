## Importing libraries
#%%
import random 
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from image_graph_prediction.graph_dataset import MultimodalGraphDataset
from collections import Counter
#%%
def set_up_multimodal_dataset():
    """Setup the multimodal dataset similar to set_up_mediaeval2015()"""
    # Load your data
    df = pd.read_csv("datasets/processed/all_data_df_resolved.csv")

    img_embeddings_df = pd.read_pickle("src/embeddings/clip_img_title_embeddings/image_embeddings.pkl")
    print(img_embeddings_df["referenced_image_embeddings"][0].shape)
    text_embeddings_df = pd.read_pickle("src/embeddings/clip_title_embeddings/title_embeddings.pkl")
    
    # Filter valid data
    valid_df = df[df['resolved_text'].notnull()].reset_index(drop=True)
    valid_img_embeddings_df = img_embeddings_df.loc[valid_df.index].reset_index(drop=True)
    valid_text_embeddings_df = text_embeddings_df.loc[valid_df.index].reset_index(drop=True)
    
    all_img_embeddings = valid_img_embeddings_df['referenced_image_embeddings'].values
    all_text_embeddings = valid_text_embeddings_df['title_embeddings'].values
    
    valid_df['label'] = valid_df['type'].map({'real': 0, 'fake': 1})
    
    
    # Create train/test split
    train_idx, temp_idx = train_test_split(
        range(len(valid_df)), 
        test_size=(0.15 + 0.1),
        random_state=42,
        stratify=valid_df['label'] if 'label' in valid_df.columns else None
    )
    
    # Second split: val and test from temp
    val_size_adjusted = 0.15 / (0.15 + 0.1)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_size_adjusted),
        random_state=42,
        stratify=valid_df.iloc[temp_idx]['label'] if 'label' in valid_df.columns else None
    )
    
    # Create datasets for each split
    train_df =valid_df.iloc[train_idx].reset_index(drop=True)
    val_df = valid_df.iloc[val_idx].reset_index(drop=True)
    test_df = valid_df.iloc[test_idx].reset_index(drop=True)
    
    # Split embeddings based on indices
    train_img_emb = [all_img_embeddings[i] for i in train_idx]
    val_img_emb = [all_img_embeddings[i] for i in val_idx]
    test_img_emb = [all_img_embeddings[i] for i in test_idx]
    
    train_text_emb = [all_text_embeddings[i] for i in train_idx]
    val_text_emb = [all_text_embeddings[i] for i in val_idx]
    test_text_emb = [all_text_embeddings[i] for i in test_idx]
    
    # Create dataset objects
    train_dataset = MultimodalGraphDataset(train_df, train_img_emb, train_text_emb)
    val_dataset = MultimodalGraphDataset(val_df, val_img_emb, val_text_emb)
    test_dataset = MultimodalGraphDataset(test_df, test_img_emb, test_text_emb)
    
    return train_dataset, val_dataset, test_dataset
#  #%%
# train_dataset, val_dataset, test_dataset = set_up_multimodal_dataset()
# # valid_df = pd.read_csv("../../datasets/processed/all_data_df_resolved.csv")
# labels = [data.y.item() for data in test_dataset]
# label_counts = Counter(labels)
# print(label_counts)
# %%
