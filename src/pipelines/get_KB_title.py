# %%
import asyncio
import pickle
import aiohttp
import time
from pipelines.get_embeddings import get_embeddings
# from utils.data_cleaning import clean_ocr_text
from utils.image_utils import  get_pil_image_cached
from utils.reference_images import get_batch
import utils.tagMe as tagme
import requests
from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd
import json
import hashlib
import numpy as np
import logging
import torch
import concurrent.futures
from PIL import Image as PILImage
import io
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModel, CLIPTextModelWithProjection
import pytesseract
import re
from pytesseract import Output
import json
# %%
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
IMG_PATH = "allData_images"
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Global caches
annotation_cache = {}  # Cache for TagMe API responses


# %%
load_dotenv(override=True)
GCUBE_TOKEN = os.getenv('TAG_ME_TOKEN')

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModel.from_pretrained(bert_model_name).to(device)
bert_model.eval()

clip_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")


model_name='clip'


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
    
def get_text_embedding(text, model_name="bert"):
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

#%%
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

    # Step 4: Basic validation - check if we have reasonable text
    words = cleaned_text.split()
    if len(words) < 2:
        return ""  # Treat as garbage
    
    return cleaned_text

#%%
def process_title_embeddings_batch(all_text, df, get_text_embedding, tokenizer=clip_tokenizer):
    """
    Returns: list of tensors, each containing embeddings for [entities + title + OCR]
    tokenizer: Optional tokenizer for accurate token counting
    """
    
    results = []
    for i, entity_list in enumerate(all_text):
        print(f"Processing row {i}")
        # Entity embeddings
        entity_embeddings = [get_text_embedding(ent, model_name) for ent in entity_list]

        # Title embedding with chunking (only for CLIP model)
        resolved_title = df['resolved_title'].iloc[i]
        
        if model_name.lower() == "clip":
            # Check if title needs chunking for CLIP model
            if tokenizer is not None:
                token_count = len(tokenizer.encode(resolved_title, add_special_tokens=False))
            else:
                # Rough estimate: ~1.3 tokens per word
                token_count = len(resolved_title.split()) * 1.3
            
            if token_count > 77:
                # Chunk the title
                title_chunks = chunk_text_by_tokens(resolved_title, max_tokens=77, tokenizer=tokenizer)
                title_embeddings = [get_text_embedding(chunk, model_name) for chunk in title_chunks]
            else:
                # Single title embedding
                title_embeddings = [get_text_embedding(resolved_title, model_name)]
        else:
            # For non-CLIP models, use title as-is without chunking
            title_embeddings = [get_text_embedding(resolved_title, model_name)]
        
        # OCR embedding with chunking (only for CLIP model)
        img_file = df['image_filename'].iloc[i]
        img_for_ocr = os.path.join(IMG_PATH, img_file).replace("\\", "/") + '.jpg'
        
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
                        ocr_embeddings = [get_text_embedding(chunk, model_name) for chunk in ocr_chunks]
                    else:
                        # Single OCR embedding
                        ocr_embeddings = [get_text_embedding(ocr_text, model_name)]
                else:
                    # For non-CLIP models, use OCR text as-is without chunking
                    ocr_embeddings = [get_text_embedding(ocr_text, model_name)]
        except Exception as e:
            logger.warning(f"OCR failed for {img_for_ocr}: {e}")

        # Combine embeddings: entities + title chunks + OCR chunks (if available)
        all_embeddings = entity_embeddings + title_embeddings
        if ocr_embeddings:
            all_embeddings.extend(ocr_embeddings)

        if all_embeddings:
            combined_tensor = torch.stack(all_embeddings)
        else:
            print("No valid embeddings found — using random fallback")
            combined_tensor = torch.randn(1, 768)  # Random fallback
            combined_tensor = combined_tensor / combined_tensor.norm(dim=-1, keepdim=True) 
       
        results.append(combined_tensor)
    
    return results

def safe_text(val):
    if isinstance(val, float) or pd.isna(val):
        return ""
    return str(val)

#%%
class AsyncTagMeClient:
    """Async client for TagMe API calls"""
    
    def __init__(self, gcube_token, max_concurrent=10, rate_limit_delay=0.1):
        self.gcube_token = gcube_token
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0
        
    async def __aenter__(self):
        # Configure session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=50,  # Total connection pool size
            limit_per_host=20,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=60, connect=20)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'AsyncTagMeClient/1.0'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Simple rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    async def annotate_async(self, text, cache_key):
        """Async TagMe annotation with caching"""
        # Create cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        full_cache_key = f"{cache_key}_{text_hash}"
        
        if full_cache_key in annotation_cache:
            logger.debug(f"Cache hit for {cache_key}")
            return annotation_cache[full_cache_key]
        
        async with self.semaphore:  # Limit concurrent requests
            await self._rate_limit()
            
            try:
                # Prepare data for TagMe API
                data = {
                    'text': text,
                    'lang': 'en',
                    'gcube-token': self.gcube_token
                }
                
                async with self.session.post(
                    'https://tagme.d4science.org/tagme/tag',
                    data=data
                ) as response:
                    if response.status == 200:
                        json_data = await response.json()
                        # Convert to your AnnotateResponse format (assuming tagme module available)
                        annotations = tagme.AnnotateResponse(json_data) if json_data else None
                        annotation_cache[full_cache_key] = annotations
                        logger.debug(f"API success for {cache_key}")
                        return annotations
                    else:
                        logger.warning(f"TagMe API error {response.status} for {cache_key}")
                        return None
                        
            except Exception as e:
                logger.error(f"TagMe annotation failed for {cache_key}: {e}")
                return None


async def extract_top_tagme_entities_async(annotations, max_count, global_seen_entities):
    """Extract top TagMe entities from title only"""
    if not annotations:
        return []

    entity_scores = []
    min_score_threshold = 0.1

    for ann in annotations.get_annotations(min_score_threshold):
        if ann.entity_title in global_seen_entities:
            continue
        entity_scores.append({
            "score": ann.score,
            "original": ann.mention,
            "linked_title": ann.entity_title,
            "mention": ann.mention,
            "start": getattr(ann, 'start', None),
            "end": getattr(ann, 'end', None)
        })

    # Sort and take top entities
    sorted_entities = sorted(entity_scores, key=lambda x: x["score"], reverse=True)
    top_entities = sorted_entities[:max_count]
    
    # Update global seen entities
    global_seen_entities.update([e["linked_title"] for e in top_entities])
    
    return top_entities

def deduplicate_and_rank_entities(tagme_entities, max_entities=20):
    """Deduplicate entities and return top max_entities with score >= 0.5, ranked by importance."""
    seen_entities = set()
    filtered_entities = []
    all_candidates = []

    def normalize_entity(text):
        return text.lower().strip()

    def add_if_unique(entity, source, priority_score=0, target_list=None):
        normalized = normalize_entity(entity.get('linked_title', entity.get('original', '')))
        if normalized and normalized not in seen_entities:
            seen_entities.add(normalized)
            entity_copy = entity.copy()
            entity_copy['source'] = source
            entity_copy['priority_score'] = priority_score
            if target_list is not None:
                target_list.append(entity_copy)
            return True
        return False

    # Collect all candidates from title only
    for entity in tagme_entities:
        add_if_unique(entity, 'tagme_title', entity.get('score', 0), all_candidates)

    # Filter by score >= 0.1
    filtered_entities = [e for e in all_candidates if e['priority_score'] >= 0.1]

    # If less than 2, add more from all_candidates
    if len(filtered_entities) < 2:
        all_candidates_sorted = sorted(all_candidates, key=lambda x: x['priority_score'], reverse=True)
        for e in all_candidates_sorted:
            if e not in filtered_entities:
                filtered_entities.append(e)
            if len(filtered_entities) >= 2:
                break

    # Sort by priority score (highest first)
    filtered_entities.sort(key=lambda x: x['priority_score'], reverse=True)

    # Limit to max_entities
    return filtered_entities[:max_entities]

async def process_single_text_async(text_data, tagme_client):
    """Async processing of a single text for entity extraction from title only"""
    resolved_title, index = text_data
    entity_dedup_cache = set()  # Local entity deduplication for TagMe
    logger.info(f"Processing text {index + 1}")
    
    try:
        # Process title with TagMe only
        title_annotations = await tagme_client.annotate_async(
            resolved_title, f"title_{index}"
        )
        
        title_tagme_entities = []
        if title_annotations:
            title_tagme_entities = await extract_top_tagme_entities_async(
                title_annotations, 16, entity_dedup_cache
            )
        
        # Deduplicate and rank entities
        final_entities = deduplicate_and_rank_entities(title_tagme_entities, max_entities=16)
        
        logger.info(f"Text {index + 1} completed: Final entities: {len(final_entities)}")
        
        return final_entities
        
    except Exception as e:
        logger.error(f"Error processing text {index + 1}: {e}")
        return []

async def batch_entity_analysis_async(df, gcube_token, max_concurrent_texts=10):
    """Main async batch processing function for entity extraction from titles only"""
    logger.info("=== ASYNC BATCH ENTITY ANALYSIS (TITLE ONLY) ===")
    logger.info(f"Processing {len(df)} texts with max {max_concurrent_texts} concurrent")
    
    start_time = time.time()
    
    async with AsyncTagMeClient(gcube_token, max_concurrent=30) as tagme_client:
        # Prepare data - only title needed
        text_data = [
            (df['resolved_title'].iloc[i], i) 
            for i in range(len(df))
        ]
        
        # Process in batches to avoid overwhelming the APIs
        batch_size = max_concurrent_texts
        all_results = [None] * len(df)  # Pre-allocate to maintain order
        
        with tqdm(total=len(df), desc="Processing texts") as pbar:
            for i in range(0, len(text_data), batch_size):
                batch = text_data[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
                
                # Create tasks for this batch
                tasks = [
                    process_single_text_async(data, tagme_client)
                    for data in batch
                ]
                
                # Process batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Store results in correct positions
                for j, (data, result) in enumerate(zip(batch, batch_results)):
                    original_index = data[1]  # index is second element
                    if isinstance(result, Exception):
                        logger.error(f"Error in text {original_index + 1}: {result}")
                        all_results[original_index] = []
                    else:
                        all_results[original_index] = result
                    pbar.update(1)
                
                # Small delay between batches
                if i + batch_size < len(text_data):
                    await asyncio.sleep(1)
    
    end_time = time.time()
    
    # Calculate statistics
    total_entities = sum(len(result) for result in all_results if result)
    
    logger.info(f"=== ASYNC PROCESSING COMPLETE ===")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
    logger.info(f"Total unique entities found: {total_entities}")
    logger.info(f"Average entities per text: {total_entities / len(df):.1f}")
    logger.info(f"Cache stats - Annotations: {len(annotation_cache)}")
    
    return all_results

# Wrapper functions for different execution environments
def run_async_entity_analysis(df, gcube_token, max_concurrent_texts=20):
    """Wrapper to run async analysis - compatible with existing event loops"""
    try:
        # Try to get existing loop
        loop = asyncio.get_running_loop()
        # If we're in a running loop (like Jupyter), create task
        import nest_asyncio
        nest_asyncio.apply()  # Allows nested event loops
        return asyncio.run(
            batch_entity_analysis_async(df, gcube_token, max_concurrent_texts)
        )
    except RuntimeError:
        # No running loop, use asyncio.run normally
        return asyncio.run(
            batch_entity_analysis_async(df, gcube_token, max_concurrent_texts)
        )
    except ImportError:
        # nest_asyncio not available, try alternative approach
        return run_with_new_loop(df, gcube_token, max_concurrent_texts)

def run_with_new_loop(df, gcube_token, max_concurrent_texts=20):
    """Alternative approach using new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            batch_entity_analysis_async(df, gcube_token, max_concurrent_texts)
        )
    finally:
        loop.close()

# Example usage function
def extract_entities_from_dataframe(df, gcube_token, max_concurrent_texts=20):
    """
    Main function to extract entities from dataframe using resolved_title only
    
    Args:
        df: DataFrame with 'resolved_title' column
        gcube_token: Your TagMe API token
        max_concurrent_texts: Maximum concurrent text processing
    
    Returns:
        List of dictionaries containing entity extraction results
    """
    results = run_async_entity_analysis(df, gcube_token, max_concurrent_texts)
    return [
        [entity['linked_title'] for entity in entity_list]
        for entity_list in results
    ]

#%%

df = pd.read_csv("datasets/processed/all_data_df_resolved.csv")
# embedding_df = pd.read_pickle("src/bert_title_embeddings/title_embeddings_1500.pkl")

#%%
batch_df = get_batch(0,len(df),df)

# Extract entities from titles only
# all_texts = extract_entities_from_dataframe(batch_df, GCUBE_TOKEN, max_concurrent_texts=8)



# %%

# Process embeddings: entities + title + OCR
with open("src/embeddings/bert_title_embeddings/all_texts.json", "r", encoding="utf-8") as f:
    all_texts = json.load(f)
title_embeddings = process_title_embeddings_batch(all_texts, batch_df, get_text_embedding)

# # Create batch embedding dataframe
batch_embedding_df = pd.DataFrame({
    "index": batch_df.index,  # preserves mapping to original df
    "title_embeddings": title_embeddings
})
# print(batch_embedding_df["title_embeddings"][2].shape)
# Combine with existing embeddings
# embedding_df = pd.concat([embedding_df, batch_embedding_df], ignore_index=True)
batch_embedding_df.to_pickle("src/embeddings/clip_title_embeddings/title_embeddings.pkl")

# %%
# with open("src/embeddings/bert_title_embeddings/all_texts.json", "w", encoding="utf-8") as f:
#     json.dump(all_texts, f, ensure_ascii=False, indent=2)