# %%
import asyncio
import pickle
import aiohttp
import time
from utils.image_utils import  get_pil_image_cached
from utils.reference_images import get_batch, images_exist_for_index, load_images_for_batch, process_img_embeddings_batch, save_images_from_batch
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
# %%
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
IMG_PATH = "allData_images"
# Global caches
# %%
nlp = spacy.load("en_core_web_trf")
# Global caches
annotation_cache = {}  # Cache for TagMe API responses
image_cache = []  # Your existing image cache


# %%
load_dotenv(override=True)
GCUBE_TOKEN = os.getenv('TAG_ME_TOKEN')

# %%
def get_or_create_event_loop():
    """Get existing event loop or create new one if needed"""
    try:
        loop = asyncio.get_running_loop()
        return loop, False  # Loop exists, don't close it
    except RuntimeError:
        # No running loop, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop, True  # New loop, should close it

async def ensure_loop_running():
    """Ensure we have a running event loop"""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        # This shouldn't happen in an async function, but just in case
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

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
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
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
                        # Convert to your AnnotateResponse format
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

class AsyncWikipediaClient:
    """Async client for Wikipedia image fetching"""
    
    def __init__(self, max_concurrent=15):
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def __aenter__(self):
        connector = aiohttp.TCPConnector(limit=30, limit_per_host=15)
        timeout = aiohttp.ClientTimeout(total=20, connect=5)
        self.session = aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_entity_image_async(self, entity_title, lang='en', image_size='small',return_pil=False):
        """Async version of get_entity_image"""
        async with self.semaphore:
            try:
                from utils.tagMe import wiki_title, _get_image_size_pixels, WikipediaImage
                
                clean_title = wiki_title(entity_title)
                api_url = f"https://{lang}.wikipedia.org/w/api.php"
                
                params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': clean_title,
                    'prop': 'pageimages|pageterms',
                    'pithumbsize': _get_image_size_pixels(image_size),
                    'pilimit': 1,
                    'wbptterms': 'description'
                }
                
                async with self.session.get(api_url, params=params) as response:
                  
                    if response.status != 200:
                        return None
                        
                    data = await response.json()
                 
                    pages = data.get('query', {}).get('pages', {})
                    
                    for page_id, page_data in pages.items():
                        if page_id == '-1':
                            continue
                            
                        thumbnail = page_data.get('thumbnail')
                        if not thumbnail:
                            continue
                        
                        pageimage = page_data.get('pageimage')
                        original_url = None
                        
                        # Get original image URL if needed
                        if pageimage:
                            original_url = await self._get_original_image_url(
                                api_url, pageimage
                            )
                        image_url = original_url or thumbnail['source']    
                        if return_pil:
                            # Download and open as PIL Image
                            async with self.session.get(image_url) as img_response:
                                if img_response.status == 200:
                                    img_bytes = await img_response.read()
                                    pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                                    return pil_img
                                else:
                                    return None
                        else:        
                        # Extract description
                            description = ""
                            terms = page_data.get('terms', {})
                            if 'description' in terms:
                                description = terms['description'][0]
                            
                            return WikipediaImage(
                                url=image_url,
                                title=pageimage or 'Unknown',
                                description=description,
                                width=thumbnail.get('width'),
                                height=thumbnail.get('height'),
                                thumbnail_url=thumbnail['source'],
                                page_url=f"https://{lang}.wikipedia.org/wiki/{entity_title}"
                            )
                        
            except Exception as e:
                logger.warning(f"Error fetching image for {entity_title}: {e}")
                return None
    
    async def _get_original_image_url(self, api_url, pageimage):
        """Get original image URL"""
        try:
            params = {
                'action': 'query',
                'format': 'json',
                'titles': f'File:{pageimage}',
                'prop': 'imageinfo',
                'iiprop': 'url'
            }
            
            async with self.session.get(api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    pages = data.get('query', {}).get('pages', {})
                    
                    for page_data in pages.values():
                        imageinfo = page_data.get('imageinfo', [])
                        if imageinfo:
                            return imageinfo[0].get('url')
        except:
            pass
        return None

async def get_images_batch_async(entity_titles, wiki_client):
    """Async batch image retrieval"""
    if not entity_titles:
        return []
    
    images = []
    uncached_titles = []
    
    # Check cache first
    for title in entity_titles:
        cached = next((item['img'] for item in image_cache if item['query'] == title), None)
        if cached:
            logger.debug(f"Cache hit for image: {title}")
            images.append(cached)
        else:
            uncached_titles.append(title)
    
    # Fetch uncached images concurrently
    if uncached_titles:
        logger.info(f"Fetching {len(uncached_titles)} uncached images")
        
        tasks = [
            wiki_client.get_entity_image_async(title,return_pil=True) 
            for title in uncached_titles
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for title, result in zip(uncached_titles, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching image for {title}: {result}")
            elif result:
                images.append(result)
                image_cache.append({'query': title, 'img': result})
                logger.debug(f"✓ Fetched image: {title}")
    
    return images

async def extract_top_entities_async(annotations, max_count, linked_name_cache, 
                                   global_seen_entities, wiki_client):
    """Async version of extract_top_entities, ensures at least 2 images if possible"""
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
            "ann": ann
        })

    # Sort and take top entities
    sorted_entities = sorted(entity_scores, key=lambda x: x["score"], reverse=True)
    images = []
    used_titles = set()
    i = 0

    # Try to fetch images until at least 2 are found or entities are exhausted
    while len(images) < 2 and i < len(sorted_entities):
        # Try next batch of up to max_count entities
        batch_entities = sorted_entities[i:i+max_count]
        entity_titles = [e["linked_title"] for e in batch_entities if e["linked_title"] not in used_titles]
        if not entity_titles:
            break
        new_images = await get_images_batch_async(entity_titles, wiki_client)
        images.extend([img for img in new_images if img is not None])
        used_titles.update(entity_titles)
        i += max_count

    # Update caches
    global_seen_entities.update(used_titles)

    return images[:max_count] 

def is_required_entity(entity):
    return nlp(entity)[0].ent_type_.lower() in ['person', 'gpe', 'fac', 'org', 'work_of_art', 'norp', 'loc']

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

async def process_single_text_async(text_data, tagme_client, wiki_client):
    """Async processing of a single text"""
    text, index = text_data
    linked_name_cache = set()
    entity_dedup_cache = set()  # Local entity deduplication across all texts
    logger.info(f"Processing text {index + 1}")
    
    try:
        # Split headline and body
        parts = text.split('<body>')
        if len(parts) != 2:
            logger.warning(f"Text {index + 1} doesn't have proper <body> separator")
            return []
            
        headline, body = parts[0], parts[1]
        all_images = []
        
        # Process headline
        headline_annotations = await tagme_client.annotate_async(
            headline, f"headline_{index}"
        )
        if headline_annotations:
            head_images = await extract_top_entities_async(
                headline_annotations, 3, linked_name_cache, 
                entity_dedup_cache, wiki_client
            )
            all_images.extend(head_images)
        
        # Process body chunks
        body_chunks = split_text_into_chunks_optimized(body, 3)
        
        # Process all chunks concurrently
        chunk_tasks = []
        for chunk_idx, chunk in enumerate(body_chunks):
            task = tagme_client.annotate_async(chunk, f"body_{index}_{chunk_idx}")
            chunk_tasks.append((task, chunk_idx))
        
        # Wait for all annotations
        chunk_annotations = []
        for task, chunk_idx in chunk_tasks:
            annotations = await task
            if annotations:
                chunk_annotations.append(annotations)
        
        # Process entity extraction for chunks
        chunk_image_tasks = []
        for annotations in chunk_annotations:
            task = extract_top_entities_async(
                annotations, 2, linked_name_cache, 
                entity_dedup_cache, wiki_client
            )
            chunk_image_tasks.append(task)
        
        # Wait for all chunk images
        if chunk_image_tasks:
            chunk_image_results = await asyncio.gather(*chunk_image_tasks)
            for chunk_images in chunk_image_results:
                all_images.extend(chunk_images)
        
        logger.info(f"Text {index + 1} completed: {len(all_images)} images")
        return all_images
        
    except Exception as e:
        logger.error(f"Error processing text {index + 1}: {e}")
        return []

async def batch_image_analysis_async(texts, gcube_token, max_concurrent_texts=10):
    """Main async batch processing function"""
    logger.info("=== ASYNC BATCH IMAGE ANALYSIS ===")
    logger.info(f"Processing {len(texts)} texts with max {max_concurrent_texts} concurrent")
    
    start_time = time.time()
    
    async with AsyncTagMeClient(gcube_token, max_concurrent=30) as tagme_client, \
               AsyncWikipediaClient(max_concurrent=20) as wiki_client:
        
        # Prepare data
        text_data = [(text, i) for i, text in enumerate(texts)]
        
        # Process in batches to avoid overwhelming the APIs
        batch_size = max_concurrent_texts
        all_results = [None] * len(texts)  # Pre-allocate to maintain order
        
        with tqdm(total=len(texts), desc="Processing texts") as pbar:
            for i in range(0, len(text_data), batch_size):
                batch = text_data[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                # Create tasks for this batch
                tasks = [
                    process_single_text_async(data, tagme_client, wiki_client)
                    for data in batch
                ]
                
                # Process batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Store results in correct positions
                for j, (data, result) in enumerate(zip(batch, batch_results)):
                    original_index = data[1]
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
    total_images = sum(len(result) for result in all_results if result)
    
    logger.info(f"=== ASYNC PROCESSING COMPLETE ===")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
    logger.info(f"Total images found: {total_images}")
    logger.info(f"Cache stats - Annotations: {len(annotation_cache)}, Images: {len(image_cache)}")
    
    return all_results

async def process_large_dataset_async(df, gcube_token, batch_size=200, max_concurrent=15):
    """Process large dataset with async operations"""
    total_rows = len(df)
    all_results = []
    
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        logger.info(f"Processing batch {start_idx}-{end_idx} of {total_rows}")
        
        # Create batch texts
        batch_texts = (
            df['resolved_title'].iloc[start_idx:end_idx] + " <body>" + 
            df['resolved_text'].iloc[start_idx:end_idx]
        ).tolist()
        
        # Process batch async
        batch_results = await batch_image_analysis_async(
            batch_texts, gcube_token, max_concurrent_texts=max_concurrent
        )
        all_results.extend(batch_results)
        
        # Optional memory management
        if start_idx > 0 and start_idx % (batch_size * 3) == 0:
            logger.info("Clearing annotation cache...")
            annotation_cache.clear()
    
    return all_results

# FIXED: Event loop compatible wrapper functions
def run_async_batch_analysis(texts, gcube_token, max_concurrent_texts=20):
    """Wrapper to run async analysis - compatible with existing event loops"""
    try:
        # Try to get existing loop
        loop = asyncio.get_running_loop()
        # If we're in a running loop (like Jupyter), create task
        import nest_asyncio
        nest_asyncio.apply()  # Allows nested event loops
        return asyncio.run(
            batch_image_analysis_async(texts, gcube_token, max_concurrent_texts)
        )
    except RuntimeError:
        # No running loop, use asyncio.run normally
        return asyncio.run(
            batch_image_analysis_async(texts, gcube_token, max_concurrent_texts)
        )
    except ImportError:
        # nest_asyncio not available, try alternative approach
        return run_with_new_loop(texts, gcube_token, max_concurrent_texts)

def run_with_new_loop(texts, gcube_token, max_concurrent_texts=20):
    """Alternative approach using new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            batch_image_analysis_async(texts, gcube_token, max_concurrent_texts)
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
async def run_in_existing_loop(texts, gcube_token, max_concurrent_texts=20):
    """Use this if you're already in an async environment"""
    return await batch_image_analysis_async(texts, gcube_token, max_concurrent_texts)

async def run_large_dataset_in_existing_loop(df, gcube_token, batch_size=200, max_concurrent=15):
    """Use this for large datasets if you're already in an async environment"""
    return await process_large_dataset_async(df, gcube_token, batch_size, max_concurrent)

# %%

df=pd.read_csv("datasets/processed/all_data_df_resolved.csv")
# embedding_df = pd.DataFrame(columns=["index", "referenced_image_embeddings"])

embedding_df = pd.read_pickle("src/embeddings/image_embeddings_400.pkl")


# embedding_df = pd.read_pickle("src/embeddings/resnet_text_embeddings/image_embeddings_3600.pkl")

batch_df = get_batch(0,800,df)


# === Check if analysis is needed ===
image_dir = "reference_images"

# missing_indices = [idx for idx in batch_df.index if  images_exist_for_index(idx, image_dir) != "file exists"]

# print(len(missing_indices))

# if len(missing_indices) > 0:
#     print("Images missing for indices:", missing_indices)
#     texts = batch_df.loc[missing_indices, 'resolved_title'].tolist()
#     texts = (batch_df.loc[missing_indices, 'resolved_title'] + " <body>" + batch_df.loc[missing_indices, 'resolved_text']).tolist()
#     all_imgs = run_async_batch_analysis(texts, GCUBE_TOKEN, max_concurrent_texts=8)
#     save_images_from_batch(all_imgs, missing_indices, image_dir)

# else:
#     print("All images exist. Skipping run_async_batch_analysis.")
all_imgs = load_images_for_batch(batch_df, image_dir)


all_embeddings = process_img_embeddings_batch(all_imgs, df, batch_df, IMG_PATH, mode="clip")
# Create batch embedding dataframe
batch_embedding_df = pd.DataFrame({
    "index": batch_df.index,  # preserves mapping to original df
    "referenced_image_embeddings": all_embeddings
})


# # # Append to the master embedding DataFrame
embedding_df = pd.concat([embedding_df, batch_embedding_df], ignore_index=True)
embedding_df.to_pickle("src/embeddings/image_embeddings_800.pkl")
# # # # Append to the master embedding DataFrame
# embedding_df = pd.concat([embedding_df, batch_embedding_df], ignore_index=True)



