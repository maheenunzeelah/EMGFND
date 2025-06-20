import config
from transformers import AutoTokenizer, AutoModel, CLIPTextModelWithProjection
import torch
import pytesseract
import re
from pytesseract import Output
import config 
from utils.image_utils import get_pil_image_cached
import os
from PIL import Image as PILImage
from pipelines.get_embeddings import get_embeddings
import spacy

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
bert_model.eval()

clip_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")


model_name= config.text_embed_model
nlp = spacy.load("en_core_web_trf")


def chunk_text_by_tokens(text, max_tokens=77, tokenizer=None):
    """
    Split text into chunks based on actual token count
    If tokenizer is provided, uses actual tokenization; otherwise approximates
    """
    if tokenizer is not None:
        # Use actual tokenizer for precise token counting
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    else:
        # Fallback to word-based approximation
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            # Rough approximation: assume average word is ~1.3 tokens
            estimated_tokens = len(current_chunk) * 1.3 + 1.3
            
            if estimated_tokens > max_tokens and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

def split_text_into_chunks(text, nlp, num_chunks=3, max_tokens=77, tokenizer=None, model_name=""):
    """
    Split text into chunks. If model is CLIP and text exceeds 77 tokens, use token-based chunking.
    Otherwise, use sentence-based chunking.
    """
    if model_name.lower() == "clip":
        # Check if text needs token-based chunking for CLIP
        if tokenizer is not None:
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
        else:
            # Rough estimate: ~1.3 tokens per word
            token_count = len(text.split()) * 1.3
        
        if token_count > max_tokens:
            # Use token-based chunking for long text
            return chunk_text_by_tokens(text, max_tokens=max_tokens, tokenizer=tokenizer)
    
    # Default sentence-based chunking (original logic)
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    total = len(sentences)
    if total <= num_chunks:
        return sentences
    chunk_size = total // num_chunks
    chunks = []
    for i in range(0, total, chunk_size):
        chunk_sentences = sentences[i:i + chunk_size]
        if i + chunk_size >= total:
            chunk_sentences = sentences[i:]
        chunks.append(' '.join(chunk_sentences))
        if len(chunks) >= num_chunks:
            break
    return chunks

def process_text_embeddings_batch(all_text, df, get_text_embedding, tokenizer=clip_tokenizer, model_name=config.text_embed_model, include_resolved_text=True):
    """
    Returns: list of tensors, each [num_entities + title_chunks + text_chunks + ocr_chunks, hidden_dim]
    tokenizer: Optional tokenizer for accurate token counting
    model_name: Model name to determine if chunking is needed
    include_resolved_text: Whether to process resolved_text column (requires nlp if True)
    """
    results = []
    for i, entity_list in enumerate(all_text):
        # Entity embeddings
        entity_embeddings = [get_text_embedding(ent) for ent in entity_list]
        
        # Title embedding with token chunking (only for CLIP model)
        resolved_title = df['resolved_title'].iloc[i]
        
        if model_name.lower() == "clip":
            # Check if title needs chunking for CLIP model
            if tokenizer is not None:
                title_token_count = len(tokenizer.encode(resolved_title, add_special_tokens=False))
            else:
                # Rough estimate: ~1.3 tokens per word
                title_token_count = len(resolved_title.split()) * 1.3
            
            if title_token_count > 77:
                # Chunk the title
                title_chunks = chunk_text_by_tokens(resolved_title, max_tokens=77, tokenizer=tokenizer)
                title_embeddings = [get_text_embedding(chunk) for chunk in title_chunks]
            else:
                # Single title embedding
                title_embeddings = [get_text_embedding(resolved_title)]
        else:
            # For non-CLIP models, use title as-is without chunking
            title_embeddings = [get_text_embedding(resolved_title)]
        
        # Text chunk embeddings with token chunking support (optional)
        text_embeddings = []
        if include_resolved_text:
            if nlp is None:
                raise ValueError("nlp parameter is required when include_resolved_text=True")
            
            if 'resolved_text' in df.columns:
                resolved_text = safe_text(df['resolved_text'].iloc[i])
                if resolved_text:  # Only process if text is not empty
                    text_chunks = split_text_into_chunks(resolved_text, nlp, num_chunks=3, max_tokens=77, 
                                                       tokenizer=tokenizer, model_name=model_name)
                    text_embeddings = [get_text_embedding(safe_text(chunk)) for chunk in text_chunks]

        # OCR embedding with token chunking (only for CLIP model)
        img_file = df['image_filename'].iloc[i]
        img_for_ocr = os.path.join(config.IMG_PATH, img_file).replace("\\", "/") + '.jpg'

        ocr_embeddings = []
        try:
            img_pil = get_pil_image_cached(img_for_ocr)
            ocr_text = clean_ocr_text(img_pil)
            if ocr_text:
                if model_name.lower() == "clip":
                    # Check if OCR text needs chunking for CLIP model
                    if tokenizer is not None:
                        ocr_token_count = len(tokenizer.encode(ocr_text, add_special_tokens=False))
                    else:
                        # Rough estimate: ~1.3 tokens per word
                        ocr_token_count = len(ocr_text.split()) * 1.3
                    
                    if ocr_token_count > 77:
                        # Chunk the OCR text
                        ocr_chunks = chunk_text_by_tokens(ocr_text, max_tokens=77, tokenizer=tokenizer)
                        ocr_embeddings = [get_text_embedding(chunk) for chunk in ocr_chunks]
                    else:
                        # Single OCR embedding
                        ocr_embeddings = [get_text_embedding(ocr_text)]
                else:
                    # For non-CLIP models, use OCR text as-is without chunking
                    ocr_embeddings = [get_text_embedding(ocr_text)]
        except Exception as e:
            print(f"OCR failed for {img_for_ocr}: {e}")
            ocr_text = ""

        # Combine all: entities + title chunks + text chunks (if included) + ocr chunks
        all_embeddings = entity_embeddings + title_embeddings + text_embeddings
        if ocr_embeddings:
            all_embeddings.extend(ocr_embeddings)

        if all_embeddings:
            combined_tensor = torch.stack(all_embeddings)
        else:
            print("No valid embeddings found — using random fallback")
            combined_tensor = torch.randn(1, config.text_embed_size)  # Random fallback
            combined_tensor = combined_tensor / combined_tensor.norm(dim=-1, keepdim=True) 
       
        results.append(combined_tensor)
    
    return results

def safe_text(val):
    if isinstance(val, float) or pd.isna(val):
        return ""
    return str(val)
    
def get_text_embedding(text, model_name=config.text_embed_model):
    if model_name == "clip":
        """Get CLIP text embedding for a single text string."""
        inputs = clip_tokenizer(text, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clip_model(**inputs)
            text_embeds = outputs.text_embeds.squeeze(0).cpu() 
            # print(text_embeds.shape, "text_embeds.shape")
        return text_embeds   
    else:        
        """Get BERT [CLS] embedding for a single text string."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]
        return cls_embedding.squeeze(0).cpu()  # [hidden_dim]
    
def clean_ocr_text(image, conf_threshold=90):
    """
    Extracts and cleans OCR text from an image.
    Returns an empty string if the result is likely garbage or low confidence.
    """
    # Step 1: Extract OCR data with confidence
    data = pytesseract.image_to_data(image, output_type=Output.DICT, lang='eng', config='--psm 11 -c preserve_interword_spaces=1')
    text_blocks = data['text']
    confidences = data['conf']

    # Step 2: Filter out low-confidence words
    filtered_text = []
    for txt, conf in zip(text_blocks, confidences):
        if txt.strip() and conf != '-1' and int(conf) >= conf_threshold:
            filtered_text.append(txt)

    raw_text = ' '.join(filtered_text)

    # Step 3: Clean up text
    cleaned_text = re.sub(r'\s+', ' ', raw_text)  # Normalize whitespace
    cleaned_text = re.sub(r'[^\w\s.,!?\'"-]', '', cleaned_text)  # Remove junk symbols
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)  # Remove non-ASCII
    cleaned_text = cleaned_text.strip()

    # Step 4: Use spaCy to validate if this is real text or garbage
    doc = nlp(cleaned_text)
    valid_words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop and len(token.text) >= 3]

    if len(valid_words) < 1:
        return ""  # Treat as garbage
    
    return cleaned_text

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
            random_emb = torch.randn(1, config.image_embed_size)  # Random embedding
            random_emb = random_emb / random_emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(random_emb)  # Append a random embedding if no valid images
    
    return all_embeddings