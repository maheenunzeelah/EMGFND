import asyncio
import aiohttp
import time
import config
from pipelines.pipeline_utils import clean_entity_title_for_filename
from utils.reference_images import get_batch
import utils.tagMe as tagme
import os
from dotenv import load_dotenv
import spacy
import hashlib
import numpy as np
import logging
from PIL import Image as PILImage
import io
from tqdm import tqdm 
import pandas as pd
import json
import ast

# %%
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
IMG_PATH = "allData_images"

# Global caches
annotation_cache = {}  # Cache for TagMe API responses
image_cache = []  # Your existing image cache
entity_image_cache = {}  # Cache for entity title -> image mapping
saved_entity_titles = set()  # Track which entity titles have been saved

# %%
nlp = spacy.load("en_core_web_trf")

# %%
load_dotenv(override=True)
GCUBE_TOKEN = os.getenv('TAG_ME_TOKEN')

# %%
def save_entity_image(entity_title, image, image_dir):
    """Save image with entity title as filename"""
    if not image:
        return None
    
    # Create directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)
    
    # Clean entity title for filename
    cleaned_title = clean_entity_title_for_filename(entity_title)
    filename = f"{cleaned_title}.jpg"
    filepath = os.path.join(image_dir, filename)
    
    # Save image
    try:
        image.save(filepath, 'JPEG', quality=85)
        logger.info(f"Saved image for entity: {entity_title} -> {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving image for {entity_title}: {e}")
        return None

def load_entity_metadata(image_dir):
    """Load existing entity metadata from JSON file"""
    metadata_file = os.path.join(image_dir, "text_entity_metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading entity metadata: {e}")
    return {}

def save_entity_metadata(metadata, image_dir):
    """Save entity metadata to JSON file"""
    metadata_file = os.path.join(image_dir, "text_entity_metadata.json")
    os.makedirs(image_dir, exist_ok=True)
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved entity metadata with {len(metadata)} entities")
    except Exception as e:
        logger.error(f"Error saving entity metadata: {e}")

def update_dataframe_with_entities(df, row_index, entity_titles):
    """Update dataframe with entity titles for a specific row"""
    if 'entity_titles' not in df.columns:
        # Initialize column with empty lists for all rows
        df['entity_titles'] = [[] for _ in range(len(df))]
    
    # Get current entities for this row
    current_entities = df.at[row_index, 'entity_titles']
    
    # Convert to list if it's not already (handle NaN or other types)
    if not isinstance(current_entities, list):
        current_entities = []
    
    # Add new entity titles (avoid duplicates)
    for entity_title in entity_titles:
        if entity_title not in current_entities:
            current_entities.append(entity_title)
    
    # Use .at for single value assignment
    df.at[row_index, 'entity_titles'] = current_entities
    return df

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

async def get_entity_image_async(entity_title, wiki_client, image_dir):
    """Get image for a specific entity, checking cache first"""
    global entity_image_cache, saved_entity_titles
    
    # Check if already saved
    if entity_title in saved_entity_titles:
        logger.debug(f"Entity {entity_title} already saved, skipping")
        return None
    
    # Check memory cache
    if entity_title in entity_image_cache:
        logger.debug(f"Memory cache hit for entity: {entity_title}")
        return entity_image_cache[entity_title]
    
    # Check if file exists on disk
    cleaned_title = clean_entity_title_for_filename(entity_title)
    filepath = os.path.join(image_dir, f"{cleaned_title}.jpg")
    
    if os.path.exists(filepath):
        logger.debug(f"File exists for entity: {entity_title}")
        saved_entity_titles.add(entity_title)
        return None
    
    # Fetch image
    logger.info(f"Fetching new image for entity: {entity_title}")
    image = await wiki_client.get_entity_image_async(entity_title, return_pil=True)
    
    if image:
        # Save image
        saved_path = save_entity_image(entity_title, image, image_dir)
        if saved_path:
            entity_image_cache[entity_title] = image
            saved_entity_titles.add(entity_title)
            logger.info(f"Successfully saved image for: {entity_title}")
            return image
    
    return None

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
            timeout=timeout,
            headers={
                'User-Agent': 'AsyncTagMeClient/1.0'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_entity_image_async(self, entity_title, lang='en', image_size='small', return_pil=False):
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
                                    try:
                                        img_bytes = await img_response.read()
                                    except aiohttp.ClientPayloadError as e:
                                        logger.warning(f"Incomplete image payload for {entity_title}: {e}")
                                        return None
                                    except aiohttp.ContentLengthError as e:
                                        logger.warning(f"Content length error for {entity_title}: {e}")
                                        return None
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
            wiki_client.get_entity_image_async(title, return_pil=True) 
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

async def get_image_for_entity_async(entity_title, wiki_client, image_dir):
    """Get image for a specific entity, checking cache first"""
    global entity_image_cache, saved_entity_titles
    
    
    # Check if already saved
    if entity_title in saved_entity_titles:
        logger.debug(f"Entity {entity_title} already saved, skipping")
        return None
    
    # Check memory cache
    if entity_title in entity_image_cache:
        logger.debug(f"Memory cache hit for entity: {entity_title}")
        return entity_image_cache[entity_title]
    
    # Check if file exists on disk
    cleaned_title = clean_entity_title_for_filename(entity_title)
    filepath = os.path.join(image_dir, f"{cleaned_title}.jpg")
    
    if os.path.exists(filepath):
        logger.debug(f"File exists for entity: {entity_title}")
        saved_entity_titles.add(entity_title)
        return None
    
    # Fetch image
    logger.info(f"Fetching new image for entity: {entity_title}")
    image = await wiki_client.get_entity_image_async(entity_title, return_pil=True)
    
    if image:
        # Save image
        saved_path = save_entity_image(entity_title, image, image_dir)
        print(f"Saved path: {saved_path}")
        if saved_path:
            entity_image_cache[entity_title] = image
            saved_entity_titles.add(entity_title)
            logger.info(f"Successfully saved image for: {entity_title}")
            return image
    
    return None

async def extract_top_entities_async(annotations, max_count, row_index, 
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
    entity_titles = []
    images_saved = 0
    min_images_required = 2

    for i, entity in enumerate(sorted_entities):
        # Stop if we've reached max_count AND have at least min_images_required
        if i >= max_count and images_saved >= min_images_required:
            break
        
        # Also stop if we've exhausted all entities
        if i >= len(sorted_entities):
            break
        
        entity_title = clean_entity_title_for_filename(entity["linked_title"])
        entity_titles.append(entity_title)
        
        # Try to get/save image
        image = await get_image_for_entity_async(entity_title, wiki_client, image_dir)
        if image:
            images_saved += 1

    logger.info(f"Row {row_index}: Found {len(entity_titles)} entities, saved {images_saved} new images")
    return entity_titles
    
async def process_single_text_async_with_saving(text_data, tagme_client, wiki_client, image_dir):
    """Async processing of a single text with entity tracking and image saving"""
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
        all_entity_titles = []
        
        # Process headline
        headline_annotations = await tagme_client.annotate_async(
            headline, f"headline_{index}"
        )
        if headline_annotations:
            head_entity_titles = await extract_top_entities_async(
                headline_annotations, 3, index, image_dir, wiki_client
            )
            all_entity_titles.extend(head_entity_titles)
            entity_dedup_cache.update(head_entity_titles)
        
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
        for annotations in chunk_annotations:
            chunk_entity_titles = await extract_top_entities_async(
                annotations, 2, index, image_dir, wiki_client
            )
            # Add only new entity titles not seen before
            for entity_title in chunk_entity_titles:
                if entity_title not in entity_dedup_cache:
                    all_entity_titles.append(entity_title)
                    entity_dedup_cache.add(entity_title)
        
        logger.info(f"Text {index + 1} completed: {len(all_entity_titles)} entity titles found")
        return all_entity_titles
        
    except Exception as e:
        logger.error(f"Error processing text {index + 1}: {e}")
        return []

async def batch_image_analysis_async_with_saving(texts, indices, df, gcube_token, image_dir, max_concurrent_texts=10):
    """Main async batch processing function with entity tracking and image saving"""
    logger.info("=== ASYNC BATCH IMAGE ANALYSIS WITH ENTITY TRACKING AND SAVING ===")
    logger.info(f"Processing {len(texts)} texts with max {max_concurrent_texts} concurrent")
    
    # Load existing entity metadata
    entity_metadata = load_entity_metadata(image_dir)
    global saved_entity_titles
    saved_entity_titles.update(entity_metadata.keys())
    
    start_time = time.time()
    
    async with AsyncTagMeClient(gcube_token, max_concurrent=30) as tagme_client, \
               AsyncWikipediaClient(max_concurrent=20) as wiki_client:
        
        # Prepare data
        text_data = [(text, indices[i]) for i, text in enumerate(texts)]
        
        # Process in batches
        batch_size = max_concurrent_texts
        
        with tqdm(total=len(texts), desc="Processing texts") as pbar:
            for i in range(0, len(text_data), batch_size):
                batch = text_data[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                # Create tasks for this batch
                tasks = [
                    process_single_text_async_with_saving(data, tagme_client, wiki_client, image_dir)
                    for data in batch
                ]
                
                # Process batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update dataframe with entity titles
                for j, (data, result) in enumerate(zip(batch, batch_results)):
                    row_index = data[1]
                    if isinstance(result, Exception):
                        logger.error(f"Error in row {row_index}: {result}")
                        result = []
                    
                    if result:
                        df = update_dataframe_with_entities(df, row_index, result)
                        
                        # Update metadata
                        for entity_title in result:
                            if entity_title not in entity_metadata:
                                entity_metadata[entity_title] = {
                                    'filename': f"{clean_entity_title_for_filename(entity_title)}.jpg",
                                    'rows': []
                                }
                            if row_index not in entity_metadata[entity_title]['rows']:
                                entity_metadata[entity_title]['rows'].append(row_index)
                    
                    pbar.update(1)
                
                # Small delay between batches
                if i + batch_size < len(text_data):
                    await asyncio.sleep(1)
    
    # Save updated metadata
    save_entity_metadata(entity_metadata, image_dir)
    
    end_time = time.time()
    logger.info(f"=== ASYNC PROCESSING COMPLETE ===")
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")
    logger.info(f"Total unique entities: {len(entity_metadata)}")
    
    return df

# FIXED: Event loop compatible wrapper functions
def run_async_batch_analysis_with_saving(texts, indices, df, gcube_token, image_dir, max_concurrent_texts=20):
    """Wrapper to run async analysis with saving - compatible with existing event loops"""
    try:
        # Try to get existing loop
        loop = asyncio.get_running_loop()
        # If we're in a running loop (like Jupyter), create task
        import nest_asyncio
        nest_asyncio.apply()  # Allows nested event loops
        return asyncio.run(
            batch_image_analysis_async_with_saving(texts, indices, df, gcube_token, image_dir, max_concurrent_texts)
        )
    except RuntimeError:
        # No running loop, use asyncio.run normally
        return asyncio.run(
            batch_image_analysis_async_with_saving(texts, indices, df, gcube_token, image_dir, max_concurrent_texts)
        )
    except ImportError:
        # nest_asyncio not available, try alternative approach
        return run_with_new_loop_saving(texts, indices, df, gcube_token, image_dir, max_concurrent_texts)

def run_with_new_loop_saving(texts, indices, df, gcube_token, image_dir, max_concurrent_texts=20):
    """Alternative approach using new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            batch_image_analysis_async_with_saving(texts, indices, df, gcube_token, image_dir, max_concurrent_texts)
        )
    finally:
        loop.close()

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

async def batch_image_analysis_async(texts, gcube_token, max_concurrent_texts=10):
    """Main async batch processing function (original version without saving)"""
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

async def process_single_text_async(text_data, tagme_client, wiki_client):
    """Async processing of a single text (original version)"""
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

#%%
# Updated usage example
df = pd.read_csv(config.all_data_path)
batch_df = get_batch(3200, len(df), df)
image_dir = config.all_data_reference_images_dir # Changed directory name to be more descriptive

# Load existing processed data
try:
    df_existing = pd.read_csv(config.all_data_text_entities_df)
except FileNotFoundError:
    df_existing = pd.DataFrame()  

# Check if all_data_text_entities_df has the entity_titles column
if 'entity_titles' in df_existing.columns:
    # Find which indices from batch_df are missing or have empty entity_titles in all_data_text_entities_df
    missing_indices = []
    
    for idx in batch_df.index:
        # Check if this index exists in all_data_text_entities_df

        if idx in df_existing.index:
            entity_titles_value = ast.literal_eval(df_existing.at[idx, 'entity_titles'])

            # Check if entity_titles is missing, not a list, or empty
            if not isinstance(entity_titles_value, list) or len(entity_titles_value) == 0:
                missing_indices.append(idx)
        else:
            # Index doesn't exist in all_data_text_entities_df, so it's missing
            missing_indices.append(idx)
else:
    # all_data_text_entities_df doesn't exist or doesn't have entity_titles column
    # Process all rows in batch_df
    missing_indices = batch_df.index.tolist()


texts = (batch_df.loc[missing_indices, 'resolved_title'] + " <body>" + batch_df.loc[missing_indices, 'resolved_text']).tolist()

# Process batch
updated_df = run_async_batch_analysis_with_saving(
    texts, missing_indices, batch_df, GCUBE_TOKEN, image_dir, max_concurrent_texts=8
)

if not df_existing.empty:
    updated_df = updated_df[~updated_df.index.isin(df_existing.index)]

combined_df = pd.concat([df_existing, updated_df], ignore_index=True)
combined_df.to_csv(config.all_data_text_entities_df, index=False)