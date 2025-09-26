## Importing libraries
#%%
import random 
import config
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from image_graph_prediction.graph_dataset import MultimodalGraphDataset
from collections import Counter
import matplotlib.pyplot as plt
#%%
def set_up_multimodal_dataset():
    """Setup the multimodal dataset"""
    # Load your data
    df = pd.read_csv(config.all_data_path)
    img_embeddings_df = pd.read_pickle(config.clip_img_title_embeddings_path)
    print(len(img_embeddings_df["referenced_image_embeddings"]))
    text_embeddings_df = pd.read_pickle(config.clip_title_embeddings_path)
    print(text_embeddings_df["title_embeddings"][0].shape)
    
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
        test_size=(0.30),
        random_state=42,
        stratify=valid_df['label'] if 'label' in valid_df.columns else None
    )
    
    # Second split: val and test from temp
   
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(0.15),
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

def set_up_media_eval_dataset():
    """Set up the multimodal dataset with separate test set and a train/val split"""

    # Load data and embeddings
    # train_df = pd.read_csv(config.media_eval_train_path)
    # test_df = pd.read_csv(config.media_eval_test_path)
    # print("Train Data Shape:", train_df.shape)
    # print("Test Data Shape:", test_df.shape)
    df= pd.read_csv(config.media_eval_df)

    # train_img_embeddings_df = pd.read_pickle(config.media_eval_clip_img_title_train_embeddings_path)
    # train_text_embeddings_df = pd.read_pickle(config.media_eval_clip_title_train_embedding_path)

    # test_img_embeddings_df = pd.read_pickle(config.media_eval_clip_img_title_test_embeddings_path)
    # test_text_embeddings_df = pd.read_pickle(config.media_eval_clip_title_test_embedding_path)
    img_embedding = pd.read_pickle(config.media_eval_clip_img_title_embedding_path)

    text_embedding = pd.read_pickle(config.media_eval_clip_title_embedding_path)
    print(text_embedding)

    all_img_embeddings = img_embedding['referenced_image_embeddings'].values
    all_text_embeddings = text_embedding['text_embeddings'].values


    print(text_embedding["text_embeddings"].values)

     # Create train/test split
    train_idx, temp_idx = train_test_split(
        range(len(df)), 
        test_size=(0.30),
        random_state=42,
        stratify=df['label'] if 'label' in df.columns else None
    )
    
    # Second split: val and test from temp

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(0.15),
        random_state=42,
        stratify=df.iloc[temp_idx]['label'] if 'label' in df.columns else None
    )

    # Add label column to train and test data
    df['label'] = df['label'].map({'real': 0, 'fake': 1})

    # # Split train into train and validation sets
    # train_idx, val_idx = train_test_split(
    #     range(len(train_df)),
    #     test_size=0.15,
    #     random_state=42,
    #     stratify=train_df['label']
    # )

     # Create datasets for each split
    train_df =df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
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

    # Subset train and val data
    # train_final_df = train_df.iloc[train_idx].reset_index(drop=True)
    # val_df = train_df.iloc[val_idx].reset_index(drop=True)

    # train_img_emb = [train_img_embeddings_df['referenced_image_embeddings'].iloc[i] for i in train_idx]
    # val_img_emb = [train_img_embeddings_df['referenced_image_embeddings'].iloc[i] for i in val_idx]

    # train_text_emb = [train_text_embeddings_df['text_embeddings'].iloc[i] for i in train_idx]
    # val_text_emb = [train_text_embeddings_df['text_embeddings'].iloc[i] for i in val_idx]

    # print("Train Image Embeddings Shape:", len(train_img_emb), train_img_emb[0].shape)
    # print("Train Text Embeddings Shape:", len(train_text_emb), train_text_emb[0].shape)

    # print("Val Image Embeddings Shape:", len(val_img_emb), val_img_emb[0].shape)
    # print("Val Text Embeddings Shape:", len(val_text_emb), val_text_emb[0].shape)
    # # Prepare test data
    # test_img_emb = test_img_embeddings_df['referenced_image_embeddings'].tolist()
    # test_text_emb = test_text_embeddings_df['text_embeddings'].tolist()
    # # Create dataset objects
    # train_dataset = MultimodalGraphDataset(train_final_df, train_img_emb, train_text_emb)
    # val_dataset = MultimodalGraphDataset(val_df, val_img_emb, val_text_emb)
    # test_dataset = MultimodalGraphDataset(test_df, test_img_emb, test_text_emb)

    # return train_dataset, val_dataset, test_dataset

#  #%%
# train_dataset, val_dataset, test_dataset = set_up_multimodal_dataset()
# # valid_df = pd.read_csv("../../datasets/processed/all_data_df_resolved.csv")
# labels = [data.y.item() for data in test_dataset]
# label_counts = Counter(labels)
# print(label_counts)
# %%
def plot_loss_graph(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_acc_graph(train_accs, val_accs):
    epochs = range(1, len(train_accs) + 1)

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_accs, label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()