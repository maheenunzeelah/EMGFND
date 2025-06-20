# %%
import spacy
import utils.tagMe as tagme
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Tuple, Optional
import threading
from functools import lru_cache
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
# Global variables for thread-safe operations
tagme_lock = threading.Lock()
nlp = None
# %%
def setup_models():
    """Initialize spaCy and TagMe models with GPU support"""
    global nlp
    
    # Load spaCy model with GPU support if available
    try:
        nlp = spacy.load("en_core_web_trf")
        if spacy.prefer_gpu():
            print("GPU enabled for spaCy")
        else:
            print("Using CPU for spaCy")
    except IOError:
        print("Please install spaCy English model: python -m spacy download en_core_web_sm")
        return False
    
    # Setup TagMe (ensure you have your API token set)
    load_dotenv(override=True)
    GCUBE_TOKEN = os.getenv('TAG_ME_TOKEN') 
    tagme.GCUBE_TOKEN = GCUBE_TOKEN
    print(tagme.GCUBE_TOKEN)
    
    return True

def extract_spacy_entities(text: str, entity_types: List[str] = None) -> List[Dict]:
    """
    Extract entities from text using spaCy
    
    Args:
        text: Input text
        entity_types: List of entity types to filter (e.g., ['PERSON', 'ORG', 'GPE'])
    
    Returns:
        List of entity dictionaries with text, label, and start/end positions
    """
    if not nlp:
        raise ValueError("spaCy model not loaded. Call setup_models() first.")
    
    if entity_types is None:
        entity_types = ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'PRODUCT']
    
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        if ent.label_ in entity_types:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'normalized': ent.text.lower().strip()
            })
    
    return entities

def get_tagme_annotations_batch(texts: List[str], min_rho: float = 0.1) -> List[List]:
    """
    Get TagMe annotations for multiple texts with thread safety
    
    Args:
        texts: List of texts to annotate
        min_rho: Minimum confidence score for annotations
    
    Returns:
        List of annotation lists for each text
    """
    all_annotations = []
    
    for text in texts:
        with tagme_lock:
            try:
                annotations = tagme.annotate(text)
                if annotations:
                    text_annotations = list(annotations.get_annotations(min_rho=min_rho))
                    all_annotations.append(text_annotations)
                else:
                    all_annotations.append([])
            except Exception as e:
                print(f"TagMe error for text: {e}")
                all_annotations.append([])
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
    
    return all_annotations

@lru_cache(maxsize=1000)
def normalize_entity_text(text: str) -> str:
    """Normalize entity text for matching (cached for performance)"""
    return text.lower().strip().replace("'s", "").replace("'", "")

def match_entities(spacy_entities: List[Dict], tagme_annotations: List) -> List[Dict]:
    """
    Match spaCy entities with TagMe annotations
    
    Args:
        spacy_entities: List of spaCy entity dictionaries
        tagme_annotations: List of TagMe annotation objects
    
    Returns:
        List of matched entities with TagMe annotation data
    """
    matched_entities = []
    
    # Create normalized lookup for spaCy entities
    spacy_lookup = {normalize_entity_text(ent['text']): ent for ent in spacy_entities}
    
    for ann in tagme_annotations:
        # Try exact match first
        ann_normalized = normalize_entity_text(ann.entity_title)
        
        if ann_normalized in spacy_lookup:
            matched_entity = spacy_lookup[ann_normalized].copy()
            matched_entity.update({
                'tagme_title': ann.entity_title,
                'tagme_score': ann.score,
                'tagme_annotation': ann
            })
            matched_entities.append(matched_entity)
            continue
        
        # Try partial matching
        for spacy_norm, spacy_ent in spacy_lookup.items():
            if (ann_normalized in spacy_norm or spacy_norm in ann_normalized or
                any(word in spacy_norm.split() for word in ann_normalized.split() if len(word) > 3)):
                
                matched_entity = spacy_ent.copy()
                matched_entity.update({
                    'tagme_title': ann.entity_title,
                    'tagme_score': ann.score,
                    'tagme_annotation': ann
                })
                matched_entities.append(matched_entity)
                break
    
    return matched_entities

def get_entity_image_parallel(annotation) -> Optional[Dict]:
    """Get image for a single annotation (for parallel processing)"""
    try:
        image = annotation.get_image()
        if image:
            return {
                'entity': annotation.entity_title,
                'image': image,
                'score': annotation.score
            }
    except Exception as e:
        print(f"Error getting image for {annotation.entity_title}: {e}")
    return None

def fetch_images_for_entities(matched_entities: List[Dict], max_workers: int = 10) -> List[Dict]:
    """
    Fetch images for matched entities using parallel processing
    
    Args:
        matched_entities: List of matched entity dictionaries
        max_workers: Number of parallel workers for image fetching
    
    Returns:
        List of entities with images
    """
    entities_with_images = []
    
    # Extract annotations for parallel processing
    annotations = [ent['tagme_annotation'] for ent in matched_entities if 'tagme_annotation' in ent]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all image fetching tasks
        future_to_entity = {
            executor.submit(get_entity_image_parallel, ann): (ann, matched_entities[i])
            for i, ann in enumerate(annotations)
        }
        
        # Collect results
        for future in as_completed(future_to_entity):
            ann, matched_entity = future_to_entity[future]
            try:
                result = future.result()
                if result:
                    entity_with_image = matched_entity.copy()
                    entity_with_image.update(result)
                    entities_with_images.append(entity_with_image)
            except Exception as e:
                print(f"Error processing {ann.entity_title}: {e}")
    
    return entities_with_images

def select_top_entities(headline_entities: List[Dict], text_entities: List[Dict], 
                       ratio_headline: int = 1, ratio_text: int = 3) -> List[Dict]:
    """
    Select top entities based on TagMe scores in specified ratio
    
    Args:
        headline_entities: Entities from headline with images
        text_entities: Entities from text with images
        ratio_headline: Number of entities to select from headline
        ratio_text: Number of entities to select from text
    
    Returns:
        List of selected entities (max 8 total)
    """
    # Sort by TagMe score (descending)
    headline_sorted = sorted(headline_entities, key=lambda x: x.get('tagme_score', 0), reverse=True)
    text_sorted = sorted(text_entities, key=lambda x: x.get('tagme_score', 0), reverse=True)
    
    # Select based on ratio
    selected_headline = headline_sorted[:ratio_headline]
    selected_text = text_sorted[:ratio_text]
    
    # Combine and ensure max 8 entities
    all_selected = selected_headline + selected_text
    
    # Remove duplicates based on entity title
    seen_entities = set()
    unique_selected = []
    for entity in all_selected:
        entity_key = entity.get('tagme_title', entity.get('text', ''))
        if entity_key not in seen_entities:
            seen_entities.add(entity_key)
            unique_selected.append(entity)
    
    return unique_selected[:8]

def process_news_item(headline: str, text: str, entity_types: List[str] = None) -> Dict:
    """
    Process a single news item (headline + text) and extract matched entities with images
    
    Args:
        headline: News headline
        text: News text content
        entity_types: spaCy entity types to extract
    
    Returns:
        Dictionary with processing results
    """
    start_time = time.time()
    
    try:
        # Extract spaCy entities
        headline_spacy_entities = extract_spacy_entities(headline, entity_types)
        text_spacy_entities = extract_spacy_entities(text, entity_types)
        
        # Get TagMe annotations
        tagme_annotations = get_tagme_annotations_batch([headline, text])
        headline_tagme = tagme_annotations[0] if len(tagme_annotations) > 0 else []
        text_tagme = tagme_annotations[1] if len(tagme_annotations) > 1 else []
        
        # Match entities
        headline_matched = match_entities(headline_spacy_entities, headline_tagme)
        text_matched = match_entities(text_spacy_entities, text_tagme)
        
        # Fetch images in parallel
        headline_with_images = fetch_images_for_entities(headline_matched)
        text_with_images = fetch_images_for_entities(text_matched)
        
        # Select top entities based on ratio
        selected_entities = select_top_entities(headline_with_images, text_with_images)
        
        processing_time = time.time() - start_time
        
        return {
            'headline': headline,
            'text': text,
            'headline_entities_count': len(headline_spacy_entities),
            'text_entities_count': len(text_spacy_entities),
            'matched_headline_count': len(headline_matched),
            'matched_text_count': len(text_matched),
            'entities_with_images_count': len(selected_entities),
            'selected_entities': selected_entities,
            'processing_time': processing_time
        }
        
    except Exception as e:
        return {
            'headline': headline,
            'text': text,
            'error': str(e),
            'processing_time': time.time() - start_time
        }

def process_news_batch(news_data: List[Tuple[str, str]], max_workers: int = 5) -> List[Dict]:
    """
    Process multiple news items in parallel
    
    Args:
        news_data: List of (headline, text) tuples
        max_workers: Number of parallel workers
    
    Returns:
        List of processing results
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all processing tasks
        future_to_news = {
            executor.submit(process_news_item, headline, text): (headline, text)
            for headline, text in news_data
        }
        
        # Collect results with progress tracking
        for i, future in enumerate(as_completed(future_to_news)):
            try:
                result = future.result()
                results.append(result)
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(news_data)} news items")
            except Exception as e:
                headline, text = future_to_news[future]
                results.append({
                    'headline': headline,
                    'text': text,
                    'error': str(e)
                })
    
    return results

# Example usage function
# def main_processing_example(news_items):
#     """Example of how to use the system"""
    
#     # Setup models
#     if not setup_models():
#         return
    
 
    # Process news items
   

# %%
# main_processing_example()
# %%
df=pd.read_csv("../../datasets/processed/all_data_df_resolved.csv")
sample_news = list(zip(df['resolved_title'].iloc[:10], df['resolved_text'].iloc[:10]))
setup_models()
results = process_news_batch(sample_news, max_workers=3)

# Print results
for result in results:
    if 'error' not in result:
        print(f"\nHeadline: {result['headline'][:100]}...")
        print(f"Entities with images: {result['entities_with_images_count']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
        for entity in result['selected_entities']:
            print(f"  - {entity['tagme_title']} (score: {entity['tagme_score']:.3f})")
    else:
        print(f"Error processing: {result['error']}")
# %%
