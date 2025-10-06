## Importing libraries
#%%
import config
import pandas as pd
from sklearn.model_selection import train_test_split
from emgfnd.graph_dataset import MultimodalGraphDataset

#%%
def set_up_all_data_dataset():
    """Setup the multimodal dataset"""
    # Load your data
    df = pd.read_csv(config.all_data_path)
    img_embeddings_df = pd.read_pickle(config.clip_img_text_embeddings_path)
    text_embeddings_df = pd.read_pickle(config.clip_text_embeddings_path)
    
    # Filter valid data
    valid_df = df[df['resolved_text'].notnull()].reset_index(drop=True)
    valid_img_embeddings_df = img_embeddings_df.loc[valid_df.index].reset_index(drop=True)
    valid_text_embeddings_df = text_embeddings_df.loc[valid_df.index].reset_index(drop=True)
    
    all_img_embeddings = valid_img_embeddings_df['referenced_image_embeddings'].values
    all_text_embeddings = valid_text_embeddings_df['text_embeddings'].values
    
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
        test_size=(0.10),
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

    df= pd.read_csv(config.media_eval_df)

    img_embedding = pd.read_pickle(config.media_eval_clip_img_title_embedding_path)
    text_embedding = pd.read_pickle(config.media_eval_clip_title_embedding_path)

    all_img_embeddings = img_embedding['referenced_image_embeddings'].values
    all_text_embeddings = text_embedding['text_embeddings'].values

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
