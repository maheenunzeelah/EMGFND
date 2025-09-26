import ast
import os
from PIL import Image as PILImage
import config
import torch
from pipelines.get_embeddings import get_embeddings
from utils.image_utils import get_pil_image_cached
import pandas as pd
from typing import List, Set, Tuple

def save_images_from_batch(image_lists, indices_list, output_dir):
    """
    Saves images with filenames based on provided indices and image position.

    Args:
        image_lists (list of list of PIL.Image): List of lists of images.
        indices_list (list): List of indices corresponding to the image_lists.
        output_dir (str): Directory where images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (row_index, images) in enumerate(zip(indices_list, image_lists)):
        for j, img in enumerate(images):
            filename = f"node_{row_index}_{j}.jpg"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)

def images_exist_for_index(idx, image_dir):
    """Check if at least one image file exists for a given index."""
    
    filepath = os.path.join(image_dir, f"node_{idx}_0.jpg")
    # print(os.path.exists(filepath), "----> file")
    if os.path.exists(filepath):
        return "file exists"
    else:
        print(f"Missing image for index: {idx}")
        return idx

def get_batch(start_index, end_index,df):
    return df.iloc[start_index:end_index]

def load_images_for_batch(batch_df, image_dir):
    """
    Loads images grouped by DataFrame index from files like node_<index>_<j>.jpg.
    Returns a list of lists (one per row), with possibly empty lists if no images found.
    """
    all_img = []
    
    for idx in batch_df.index:
        images = []
        j = 0
        while True:
            filename = f"node_{idx}_{j}.jpg"
            # print(f"Looking for: {filename}")
            filepath = os.path.join(image_dir, filename)
            if os.path.exists(filepath):
                try:
                    images.append(PILImage.open(filepath).convert("RGB"))
                except Exception as e:
                    print(f"Warning: Failed to open {filepath} — {e}")
                j += 1
            else:
                break
        all_img.append(images)  # Can be empty list if no images found
    
    return all_img

def process_img_embeddings_batch(all_img, df, batch_df, IMG_PATH, model):
    """Optimized batch processing"""
    all_embeddings = []
    # Pre-build all image paths to avoid repeated string operations
    img_paths = []
    print(batch_df.index)

    for idx in batch_df.index:
        filename = df.loc[idx, 'image_filename'] + '.jpg'
        # filename = "wp-content_uploads_2016_09_Hillary-cyber-security-600x315"
        # print(f"Index: {idx} | Filename: {filename}")
        path = os.path.join(IMG_PATH, filename).replace("\\", "/")
        img_paths.append(path)

    # print(all_img)
    # Process in batches or all at once depending on memory constraints
    for i, row in enumerate(all_img):
        # Collect all URLs for this row
        imgs = [img for img in row]
        try:
            img_pil = get_pil_image_cached(img_paths[i])
            # Add the main image path
            imgs.append(img_pil)
        except Exception as e:
            print(f"Main Image is corrupted: {e}")

        # Filter out failed loads if needed
        valid_images = [img for img in imgs if img is not None]
        
        if valid_images:
            # Single embedding call per row
            emb = get_embeddings(valid_images, model)
            all_embeddings.append(emb)  # Keep same structure as original
            print(f"Row {i}: processed {len(valid_images)} images")
        else:
            print(f"Row {i}: no valid images")
            random_emb = torch.randn(1, config.image_embed_size)  # Use config.image_embed_size
            random_emb = random_emb / random_emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(random_emb)  # Append a random embedding if no valid images
    
    return all_embeddings

def extract_hashtags_from_df(df: pd.DataFrame, hashtag_column: str = 'hashtag') -> Set[str]:
    """
    Extract unique hashtags from dataframe, removing # symbol and filtering empty arrays.
    
    Args:
        df: DataFrame containing hashtag column
        hashtag_column: Name of the column containing hashtag arrays
    
    Returns:
        Set of unique hashtags without # symbol
    """
    all_hashtags = set()

    for _, row in df.iterrows():
        hashtags = row[hashtag_column]

        # Skip if NaN or empty string
        if pd.isna(hashtags) or hashtags.strip() == '':
            continue

        try:
            # Convert string representation of list to actual list
            hashtags_list = ast.literal_eval(hashtags)
        except (ValueError, SyntaxError):
            continue  # Skip rows that don't parse properly

        for hashtag in hashtags_list:
            if hashtag and isinstance(hashtag, str):
                print(hashtag, "hashtag")
                clean_hashtag = hashtag.lstrip('#')
                if clean_hashtag:
                    all_hashtags.add(clean_hashtag)
    
    return all_hashtags

def check_hashtag_image_exists(hashtag: str, image_dir: str) -> bool:
    """
    Check if image already exists for a given hashtag.
    
    Args:
        hashtag: Hashtag name (without #)
        image_dir: Directory where images are stored
    
    Returns:
        True if image exists, False otherwise
    """
    # Common image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    for ext in extensions:
        image_path = os.path.join(image_dir, f"{hashtag}{ext}")
        if os.path.exists(image_path):
            return True
    
    return False

def get_missing_hashtags(hashtags: Set[str], image_dir: str) -> List[str]:
    """
    Get list of hashtags that don't have corresponding images.
    
    Args:
        hashtags: Set of hashtags to check
        image_dir: Directory where images are stored
    
    Returns:
        List of hashtags missing images
    """
    missing_hashtags = []
    
    for hashtag in hashtags:
        if not check_hashtag_image_exists(hashtag, image_dir):
            missing_hashtags.append(hashtag)
    
    return missing_hashtags

def save_hashtag_images(images, hashtags: List[str], image_dir: str, image_format: str = 'png'):
    """
    Save images for hashtags in the specified directory.
    
    Args:
        images: List of image data from run_async_batch_analysis
        hashtags: List of hashtags corresponding to images
        image_dir: Directory to save images
        image_format: Image format (default: 'png')
    """

    os.makedirs(image_dir, exist_ok=True)
     
    for i, (image_data, hashtag) in enumerate(zip(images, hashtags)):
        print(image_data,hashtag, "image_data, hashtag")
        if image_data is not None:
            image_path = os.path.join(image_dir, f"{hashtag}.jpg")
            image_data.save(image_path)

            
        #     print(f"Saved image for hashtag: {hashtag}")
        # else:
        #     print(f"No image data for hashtag: {hashtag}")