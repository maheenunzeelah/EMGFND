import os
from PIL import Image as PILImage
import torch
from pipelines.get_embeddings import get_embeddings
from utils.image_utils import get_pil_image_cached

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
            print(row_index, "row_index")
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
            random_emb = torch.randn(1, 768)
            random_emb = random_emb / random_emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(random_emb)  # Append a random embedding if no valid images
    
    return all_embeddings