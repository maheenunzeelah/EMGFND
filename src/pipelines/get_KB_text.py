# %%
import asyncio
import pickle
import aiohttp
import time
from pipelines.get_embeddings import get_embeddings
# from utils.data_cleaning import clean_ocr_text
from utils.image_utils import  get_pil_image_cached
import utils.tagMe as tagme
import requests
from pathlib import Path
import os
from dotenv import load_dotenv
import pandas as pd
import spacy
import json
import hashlib
import numpy as np
import logging
import torch
import concurrent.futures
from PIL import Image as PILImage
import io
from tqdm import tqdm 
from transformers import AutoTokenizer, AutoModel
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
# %%
nlp = spacy.load("en_core_web_trf")
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

def get_bert_embedding(text, tokenizer=tokenizer, model=bert_model, device=device):
    """Get BERT [CLS] embedding for a single text string."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]
    return cls_embedding.squeeze(0).cpu()  # [hidden_dim]
#%%
def split_text_into_chunks(text, nlp, num_chunks=3):
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

    # Step 4: Use spaCy to validate if this is real text or garbage
    doc = nlp(cleaned_text)
    valid_words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop and len(token.text) >= 3]

    if len(valid_words) < 1:
        return ""  # Treat as garbage
    
    return cleaned_text
#%%
def process_text_embeddings_batch(all_text, df, get_bert_embedding, nlp):
    """
    Returns: list of tensors, each [num_entities + 1 + num_chunks, hidden_dim]
    """
    results = []
    for i, entity_list in enumerate(all_text):
        # Entity embeddings
        entity_embeddings = [get_bert_embedding(ent) for ent in entity_list]
        # Title embedding
        resolved_title = df['resolved_title'].iloc[i]
        title_embedding = get_bert_embedding(resolved_title)
        # Text chunk embeddings
        resolved_text = safe_text(df['resolved_text'].iloc[i])
        text_chunks = split_text_into_chunks(resolved_text, nlp, num_chunks=3)
        text_embeddings = [get_bert_embedding(safe_text(chunk)) for chunk in text_chunks]

        img_file = df['image_filename'].iloc[i]
        # img_file = "wp-content_uploads_2016_09_Hillary-cyber-security-600x315"

        img_for_ocr = os.path.join(IMG_PATH, img_file).replace("\\", "/") +'.jpg'

        ocr_embedding = None
        try:
            img_pil = get_pil_image_cached(img_for_ocr)
            ocr_text = clean_ocr_text(img_pil)
            if ocr_text:
                ocr_embedding = get_bert_embedding(ocr_text)
        except Exception as e:
            logger.warning(f"OCR failed for {img_for_ocr}: {e}")
            ocr_text = ""
            

        # Combine all: entities + [title] + text chunks
        all_embeddings = entity_embeddings + [title_embedding] + text_embeddings
        if ocr_embedding is not None:
            all_embeddings.append(ocr_embedding)
        combined_tensor = torch.stack(all_embeddings) if all_embeddings else torch.empty((0, bert_model.config.hidden_size))
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


def is_required_entity(entity):
    """Check if entity mention is required (for TagMe results)"""
    return nlp(entity)[0].ent_type_.lower() in ['person', 'gpe', 'fac', 'org', 'work_of_art', 'norp', 'loc']

async def extract_top_tagme_entities_async(annotations, max_count, global_seen_entities):
    """Extract top TagMe entities """
    if not annotations:
        return []

    entity_scores = []
    min_score_threshold = 0.1

    for ann in annotations.get_annotations(min_score_threshold):
        if not is_required_entity(ann.mention):
            continue
        if ann.entity_title in global_seen_entities:
            continue
        entity_scores.append({
            "score": ann.score,
            "original": ann.mention,
            "ent_type": nlp(ann.entity_title)[0].ent_type_,
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

def split_text_into_chunks_optimized(text, num_chunks=3):
    """Optimized text splitting with sentence reuse"""
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    total = len(sentences)
    
    if total <= num_chunks:
        return sentences
    
    chunk_size = total // num_chunks
    chunks = []
    
    for i in range(0, total, chunk_size):
        chunk_sentences = sentences[i:i + chunk_size]
        if i + chunk_size >= total:  # Last chunk gets remaining
            chunk_sentences = sentences[i:]
        chunks.append(' '.join(chunk_sentences))
        if len(chunks) >= num_chunks:
            break
    
    return chunks

def deduplicate_and_rank_entities(tagme_entities, max_entities=16):
    """Deduplicate entities and return top max_entities with score >= 0.5, ranked by importance.
    If fewer than 2, add lower-score entities until at least 2 are present.
    """
    seen_entities = set()
    filtered_entities = []
    all_candidates = []

    def normalize_entity(text):
        return text.lower().strip()

    def add_if_unique(entity, source, priority_score=0, target_list=None):
        normalized = normalize_entity(entity.get('text', entity.get('original', '')))
        if normalized and normalized not in seen_entities:
            seen_entities.add(normalized)
            entity_copy = entity.copy()
            entity_copy['source'] = source
            entity_copy['priority_score'] = priority_score
            if target_list is not None:
                target_list.append(entity_copy)
            return True
        return False

    # Collect all candidates (regardless of score)
    for entity in tagme_entities.get('title', []):
        add_if_unique(entity, 'tagme_title', entity.get('score', 0), all_candidates)
    for entity in tagme_entities.get('text', []):
        add_if_unique(entity, 'tagme_text', entity.get('score', 0), all_candidates)

    # Now filter by score >= 0.5
    filtered_entities = [e for e in all_candidates if e['priority_score'] >= 0.5]

    # If less than 2, add more from all_candidates (sorted by score, not already in filtered_entities)
    if len(filtered_entities) < 2:
        # Sort all candidates by score descending
        all_candidates_sorted = sorted(all_candidates, key=lambda x: x['priority_score'], reverse=True)
        # Add until at least 2 entities
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
    """Async processing of a single text for entity extraction only"""
    resolved_title, resolved_text, index = text_data
    entity_dedup_cache = set()  # Local entity deduplication for TagMe
    logger.info(f"Processing text {index + 1}")
    
    try:
        # Process title with TagMe
        title_tagme_entities = []
        title_annotations = await tagme_client.annotate_async(
            resolved_title, f"title_{index}"
        )
        if title_annotations:
            title_tagme_entities = await extract_top_tagme_entities_async(
                title_annotations, 8, entity_dedup_cache  # Increased limit for initial extraction
            )
        
        # Process text chunks with TagMe
        text_chunks = split_text_into_chunks_optimized(resolved_text, 3)
        
        # Process all chunks concurrently
        chunk_tasks = []
        for chunk_idx, chunk in enumerate(text_chunks):
            task = tagme_client.annotate_async(chunk, f"text_{index}_{chunk_idx}")
            chunk_tasks.append((task, chunk_idx))
        
        # Wait for all annotations
        chunk_annotations = []
        for task, chunk_idx in chunk_tasks:
            annotations = await task
            if annotations:
                chunk_annotations.append(annotations)
        
        # Process entity extraction for chunks
        text_tagme_entities = []
        for annotations in chunk_annotations:
            chunk_entities = await extract_top_tagme_entities_async(
                annotations, 5, entity_dedup_cache  # Increased limit for initial extraction
            )
            text_tagme_entities.extend(chunk_entities)
        
        tagme_entities = {
            'title': title_tagme_entities,
            'text': text_tagme_entities
        }
        
        # Deduplicate and rank entities, limit to 16 total
        final_entities = deduplicate_and_rank_entities(tagme_entities, max_entities=16)
        
        logger.info(f"Text {index + 1} completed: "
                   f"Final entities: {len(final_entities)} "
                   f"TagMe Title: {len(title_tagme_entities)}, "
                   f"TagMe Text: {len(text_tagme_entities)})")
        
        return final_entities
        
    except Exception as e:
        logger.error(f"Error processing text {index + 1}: {e}")
        return []

async def batch_entity_analysis_async(df, gcube_token, max_concurrent_texts=10):
    """Main async batch processing function for entity extraction"""
    logger.info("=== ASYNC BATCH ENTITY ANALYSIS ===")
    logger.info(f"Processing {len(df)} texts with max {max_concurrent_texts} concurrent")
    
    start_time = time.time()
    
    async with AsyncTagMeClient(gcube_token, max_concurrent=30) as tagme_client:
        # Prepare data
        text_data = [
            (df['resolved_title'].iloc[i], df['resolved_text'].iloc[i], i) 
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
                    original_index = data[2]  # index is third element
                    if isinstance(result, Exception):
                        logger.error(f"Error in text {original_index + 1}: {result}")
                        all_results[original_index] = []
                    else:
                        all_results[original_index] = result
                    pbar.update(1)
                
                # Small delay between batches to be nice to APIs
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

async def process_large_dataset_async(df, gcube_token, batch_size=200, max_concurrent=15):
    """Process large dataset with async operations"""
    total_rows = len(df)
    all_results = []
    
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        logger.info(f"Processing batch {start_idx}-{end_idx} of {total_rows}")
        
        # Create batch dataframe
        batch_df = df.iloc[start_idx:end_idx].copy()
        batch_df.reset_index(drop=True, inplace=True)
        
        # Process batch async
        batch_results = await batch_entity_analysis_async(
            batch_df, gcube_token, max_concurrent_texts=max_concurrent
        )
        all_results.extend(batch_results)
        
        # Optional memory management
        if start_idx > 0 and start_idx % (batch_size * 3) == 0:
            logger.info("Clearing annotation cache...")
            annotation_cache.clear()
    
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

def run_async_large_dataset(df, gcube_token, batch_size=200, max_concurrent=15):
    """Wrapper to run async large dataset processing - event loop compatible"""
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(
            process_large_dataset_async(df, gcube_token, batch_size, max_concurrent)
        )
    except RuntimeError:
        return asyncio.run(
            process_large_dataset_async(df, gcube_token, batch_size, max_concurrent)
        )
    except ImportError:
        return run_large_dataset_with_new_loop(df, gcube_token, batch_size, max_concurrent)

def run_large_dataset_with_new_loop(df, gcube_token, batch_size=200, max_concurrent=15):
    """Alternative approach for large dataset processing"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            process_large_dataset_async(df, gcube_token, batch_size, max_concurrent)
        )
    finally:
        loop.close()

# RECOMMENDED: Direct async usage (best for Jupyter/IPython)
async def run_in_existing_loop(df, gcube_token, max_concurrent_texts=20):
    """Use this if you're already in an async environment"""
    return await batch_entity_analysis_async(df, gcube_token, max_concurrent_texts)

async def run_large_dataset_in_existing_loop(df, gcube_token, batch_size=200, max_concurrent=15):
    """Use this for large datasets if you're already in an async environment"""
    return await process_large_dataset_async(df, gcube_token, batch_size, max_concurrent)

# Example usage function
def extract_entities_from_dataframe(df, gcube_token, max_concurrent_texts=20):
    """
    Main function to extract entities from dataframe with resolved_title and resolved_text columns
    
    Args:
        df: DataFrame with 'resolved_title' and 'resolved_text' columns
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

df=pd.read_csv("datasets/processed/all_data_df_resolved.csv")
embedding_df = pd.read_pickle("src/bert-text-embeddings/text_embeddings_1500.pkl")
#%%
def get_batch(df):
    return df.iloc[1500:len(df)]

batch_df = get_batch(df)

all_texts = extract_entities_from_dataframe(batch_df, GCUBE_TOKEN, max_concurrent_texts=8)

with open("src/bert-text-embeddings/all_texts_3950.json", "w", encoding="utf-8") as f:
    json.dump(all_texts, f, ensure_ascii=False, indent=2)

# %%

# with open("../bert-text-embeddings/all_texts_1500.json", "r", encoding="utf-8") as f:
#     all_texts = json.load(f)

text_embeddings = process_text_embeddings_batch(all_texts, df , get_bert_embedding, nlp)


batch_embedding_df = pd.DataFrame({
    "index": batch_df.index,  # preserves mapping to original df
    "text_embeddings": text_embeddings
})

embedding_df = pd.concat([embedding_df, batch_embedding_df], ignore_index=True)
embedding_df.to_pickle("src/bert-text-embeddings/text_embeddings_3950.pkl")
# %%
