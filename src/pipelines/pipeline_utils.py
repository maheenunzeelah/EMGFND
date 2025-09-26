import config
from transformers import AutoTokenizer, AutoModel, CLIPTextModelWithProjection
import torch
import pytesseract
import re
from pytesseract import Output
import config 
import ast
import os
from PIL import Image as PILImage
from pipelines.get_embeddings import get_embeddings
import spacy
import pandas as pd
from utils.image_utils import find_image_file, get_pil_image_cached

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
IMG_PATH = config.IMG_PATH


def get_token_count(text, tokenizer=None):
    """Helper function to get token count with safety checks"""
    if not text or not text.strip():
        return 0
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            # Fallback to word estimate if tokenization fails
            return int(len(text.split()) * 1.5)  # Conservative estimate
    else:
        return int(len(text.split()) * 1.5)  # Conservative estimate

def split_sentence_by_tokens(sentence, max_tokens=77, tokenizer=None):
    """
    Split a single sentence into smaller chunks if it exceeds max_tokens.
    Returns list of chunks, each guaranteed to be <= max_tokens.
    """
    if not sentence or not sentence.strip():
        return []
    
    # Check if sentence fits within limit
    if get_token_count(sentence, tokenizer) <= max_tokens:
        return [sentence.strip()]
    
    # If sentence is too long, split by words
    words = sentence.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        # Test adding this word
        test_chunk = current_chunk + [word]
        test_text = ' '.join(test_chunk)
        
        if get_token_count(test_text, tokenizer) <= max_tokens:
            current_chunk = test_chunk
        else:
            # Current chunk is full, save it and start new chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk).strip())
                current_chunk = [word]
            else:
                # Single word exceeds limit (very rare), but include it anyway
                chunks.append(word)
                current_chunk = []
    
    # Add final chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    
    return [chunk for chunk in chunks if chunk.strip()]

def break_text_into_sentences(text, nlp, max_tokens=77, tokenizer=None):
    """
    Break text into sentences, ensuring no sentence exceeds max_tokens.
    Returns list of sentence chunks.
    """
    if not text or not text.strip():
        return []
    
    try:
        # Use spaCy to split into sentences
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:
        print(f"spaCy sentence splitting failed: {e}, using simple splitting")
        # Fallback to simple sentence splitting
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Process each sentence to ensure it doesn't exceed token limit
    sentence_chunks = []
    for sentence in sentences:
        if sentence:
            chunks = split_sentence_by_tokens(sentence, max_tokens=max_tokens, tokenizer=tokenizer)
            sentence_chunks.extend(chunks)
    
    return sentence_chunks

def process_text_embeddings_batch(all_text, df, tokenizer=clip_tokenizer, include_resolved_text=True):
    """
    Process text embeddings with sentence-based chunking.
    All text (resolved_text and others) are broken into sentences with max 77 tokens each.
    """
    results = []
    max_tokens = 75
    
    for i, entity_list in enumerate(all_text):
        all_embeddings = []
        
        # Process entity embeddings
        if entity_list:
            for ent in entity_list:
                if ent and ent.strip():
                    # Split entity into sentences if needed
                    if nlp is not None:
                        ent_sentences = break_text_into_sentences(ent, nlp, max_tokens=max_tokens, tokenizer=tokenizer)
                    else:
                        # Fallback without nlp
                        ent_sentences = split_sentence_by_tokens(ent, max_tokens=max_tokens, tokenizer=tokenizer)
                    
                    for sentence in ent_sentences:
                        if sentence.strip():
                            all_embeddings.append(get_text_embedding(sentence))
        
        # Process title - split into sentences if needed
        resolved_title = df['resolved_title'].iloc[i]
        if isinstance(resolved_title, str) and resolved_title.strip():
            title_sentences = break_text_into_sentences(resolved_title, nlp, max_tokens=max_tokens, tokenizer=tokenizer)
    
            for sentence in title_sentences:
                if sentence.strip():
                    all_embeddings.append(get_text_embedding(sentence))
        
        # Process resolved_text - break into sentences
        if include_resolved_text and 'resolved_text' in df.columns:
            resolved_text = safe_text(df['resolved_text'].iloc[i]) if 'safe_text' in globals() else str(df['resolved_text'].iloc[i])
            if resolved_text and resolved_text.strip():
                text_sentences = break_text_into_sentences(resolved_text, nlp, max_tokens=max_tokens-5, tokenizer=tokenizer)
                print(f"Created {len(text_sentences)} sentence chunks for resolved_text at index {i}")
                
                for sentence in text_sentences:
                    if sentence.strip():
                        all_embeddings.append(get_text_embedding(sentence))
                
        
        # Process OCR text - split into sentences if needed
        if 'image_filename' in df.columns:
            base_filename = df['image_filename'].iloc[i]
            img_for_ocr = find_image_file(base_filename, IMG_PATH)
            if img_for_ocr:
                img_for_ocr = img_for_ocr.replace("\\", "/")
            print(f"Processing OCR for image: {img_for_ocr}")

            try:
                img_pil = get_pil_image_cached(img_for_ocr)
                ocr_text = clean_ocr_text(img_pil)
                print(f"OCR text for {img_for_ocr}: {ocr_text[:100]}...")  # Print first 100 chars
                if ocr_text and ocr_text.strip():    
                    ocr_sentences = break_text_into_sentences(ocr_text, nlp, max_tokens=max_tokens, tokenizer=tokenizer)
                    for sentence in ocr_sentences:
                        if sentence.strip():
                            all_embeddings.append(get_text_embedding(sentence))
            except Exception as e:
                print(f"OCR failed for {img_for_ocr}: {e}")

        # Create combined tensor
        if all_embeddings:
            combined_tensor = torch.stack(all_embeddings)
        else:
            print(f"No valid embeddings found for index {i} — using random fallback")
            # Assuming config.text_embed_size exists, otherwise adjust as needed
            embed_size = getattr(config, 'text_embed_size', 768)  # Default to 768 if not found
            combined_tensor = torch.randn(1, embed_size)
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

def process_img_embeddings_batch(all_img, df, batch_df, model):
    """Optimized batch processing"""
    all_embeddings = []
    # Pre-build all image paths to avoid repeated string operations
    img_paths = []

    for idx in batch_df.index:
        base_filename = df.loc[idx, 'image_filename']
        path = find_image_file(base_filename, IMG_PATH)
        if path:
            path = path.replace("\\", "/")
            img_paths.append(path)
            print(f"Processing index {idx} with file: {path}")
        else:
            print(f"No image found for index {idx} with base filename: {base_filename}")
            img_paths.append(None)  # Add None to keep indexing consistent

    # print(all_img)
    # Process in batches or all at once depending on memory constraints
    for i, row in enumerate(all_img):
        # Collect all URLs for this row
        imgs = [img for img in row]
        try:
            print(img_paths[i],"imgggg")
            if img_paths[i] is not None:
                img_pil = get_pil_image_cached(img_paths[i])
                imgs.append(img_pil)
            else:
                print(f"No image path available for index {i}")
        except Exception as e:
            print(f"Main Image is corrupted: {e}")

        # Filter out failed loads if needed
        valid_images = [img for img in imgs if img is not None]
        
        if valid_images:
            # Single embedding call per row
            emb = get_embeddings(valid_images, model)
            all_embeddings.append(emb)  # Keep same structure as original

        else:
            print(f"Row {i}: no valid images")
            random_emb = torch.randn(1, config.image_embed_size)  # Random embedding
            random_emb = random_emb / random_emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(random_emb)  # Append a random embedding if no valid images
    
    return all_embeddings

def clean_entity_title_for_filename(entity_title):
    """Clean entity title to create valid filename"""
    # Replace spaces with underscores and remove invalid characters
    cleaned = entity_title.replace(' ', '_')
    # Remove or replace invalid filename characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        cleaned = cleaned.replace(char, '_')
    return cleaned

def media_eval_load_images_for_batch(batch_df, image_dir):
    """
    Load images for each row in the batch DataFrame.
    
    Returns:
        List[List[PIL.Image]]: A list where each element is a list of images for a row.
    """
    all_images = []

    for idx, row in batch_df.iterrows():
        
        entity_titles = row.get('entity_titles', [])

        try:
            entity_titles = ast.literal_eval(entity_titles)
        except Exception as e:
            print(f"Error parsing entity_titles for row {idx}: {e}")
            all_images.append([])  # Still keep alignment with rows
            continue

        if not isinstance(entity_titles, list):
            all_images.append([])
            continue

        row_images = []
        for entity_title in entity_titles:
            cleaned_title = clean_entity_title_for_filename(entity_title)
            filepath = os.path.join(image_dir, f"{cleaned_title}.jpg")
            if os.path.exists(filepath):
                try:
                    image = PILImage.open(filepath)
                    row_images.append(image)
                    print(f"Loaded image for entity: {entity_title}")
                except Exception as e:
                    print(f"Error loading image for {entity_title}: {e}")

        all_images.append(row_images)

    return all_images

def extract_text_arrays_from_column(batch_df, column_name):
    """
    Converts stringified list in a DataFrame column to actual list of strings.
    
    Args:
        batch_df (pd.DataFrame): A batch of the main DataFrame.
        column_name (str): The name of the column containing stringified lists.

    Returns:
        List[List[str]]: A list of lists of strings.
    """
    all_texts = []
    
    for raw_val in batch_df[column_name]:
        try:
            text_list = ast.literal_eval(raw_val) if isinstance(raw_val, str) else []
            cleaned_list = [text.replace("_", " ") for text in text_list if isinstance(text, str)]
            all_texts.append(cleaned_list)
        except (ValueError, SyntaxError):
            all_texts.append([])  # fallback if parsing fails
    
    return all_texts