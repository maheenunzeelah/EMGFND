# %%
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import spacy
# import requests
# from bs4 import BeautifulSoup
# import json
# import os
# from PIL import Image
# from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import tagme
from utils.data_cleaning import replace_unicode_placeholders
from utils.img_web_scraper import google_image_search
from utils.image_utils import get_pil_image
import re
load_dotenv(override=True)
GCUBE_TOKEN = os.getenv('TAG_ME_TOKEN')
print(os.getenv('TAG_ME_TOKEN'))
# %%
nlp = spacy.load("en_core_web_trf")
image_cache = {}
tagme.GCUBE_TOKEN='bfcbde68-cc51-43a9-983c-5fe6383f7920-843339462'

# %%
def extract_entities(headline):
    """Extract named entities from the headline using spaCy."""
    doc = nlp(headline)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
        
    return entities

def entity_linking(headline, entities):
    """Link entities to knowledge bases or add more context."""
    linked_entities = []
    appended_titles = []
    linked_title = ''
    annotations = tagme.annotate(headline)
    # entities = [ent.text for ent in entities.ents]
    for ann in annotations.get_annotations(0.1):
        for ent in entities:
            print( ann.mention, ent['text'])
            
            if ann.mention.lower() in ent['text'].lower():
                # print(f'Score : {ann.score}')
                # print(f'Mention : {ann.mention}')
                # print(f'Title : {ann.entity_title}')
                linked_title = ann.entity_title
                if linked_title not in appended_titles:
                    appended_titles.append(linked_title)
                    linked_entities.append({
                        "original": ann.mention,
                        "ent_type": ent['label'],
                        "linked_title": linked_title,
                        # "vector": vector.tolist()  # Convert numpy array to list for JSON serialization
                    })
        # link to Wikidata, DBpedia, or other knowledge bases
    
        

        
        # annotations_list = [ann for ann in annotations.get_annotations(0.1) if ann.score is not None]

        # if annotations_list:
        #     best_ann = max(annotations_list, key=lambda ann: ann.score)
        #     linked_title = best_ann.entity_title
        #     print(f'Highest Score: {best_ann.score}, Title: {linked_title}')
        # else:
        #     linked_title = ''
        
    
        
        
    # print(linked_entities)    
    return linked_entities
    


def search_images(query, num_images=5):
    """Search for images using a search engine API."""
    ref_images = []
    for part in query:
        print(part,"query_part")
        ref_images.append(google_image_search(part,2))
    # # Check cache first
    # if query in image_cache:
    #     return image_cache[query]
    
    # # If using Google Custom Search API
    # url = "https://www.googleapis.com/customsearch/v1"
    # params = {
    #     "q": query,
    #     "cx": search_engine_cx,
    #     "key": search_engine_api_key,
    #     "searchType": "image",
    #     "num": num_images
    # }
    
    # # For demonstration, we'll use a mock response
    # # In a real implementation, use: response = requests.get(url, params=params)
    # mock_images = mock_image_search(query, num_images)
    # image_cache[query] = mock_images

    rows = len(ref_images)
    array=np.array(ref_images)
    print(array.ndim,rows)
    return ref_images



# def download_image(image_url):
#     """Download an image from a URL."""
#     try:
#         response = requests.get(image_url)
#         return Image.open(BytesIO(response.content))
#     except Exception as e:
#         print(f"Error downloading image: {e}")
#         return None

def get_headline_query(headline, entities):
    """Build a search query based on headline and entities."""
    if not entities:
        return headline
    
    # Prioritize named entities
    # important_entities = [e for e in entities if e["original"]["label"] in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT"]]
    
    # if not important_entities:
    #     return headline
    
    # Construct query from most important entities and their context
    query_parts = []

    for entity in entities:  # Limit to top 2 entities
        entity_text = re.sub(r'\s*\(.*?\)', '', entity["linked_title"])
        # context = entity["context"]
        query_parts.append(f"{entity_text}")
    
    # return " ".join(query_parts)
    return query_parts

def match_images_to_reference_images(image, ref_images, model):
    #Encode an image:
    img_emb = model.encode(get_pil_image(image))
    ref_imgs_pil = []
    for img in ref_images:
        ref_imgs_pil.append(get_pil_image(img))
    

    #Encode ref images
    ref_emb = model.encode(ref_imgs_pil)

    #Compute cosine similarities 
    cos_scores = util.cos_sim(img_emb, ref_emb)
    return cos_scores
    # """Score images based on relevance to headline and entities."""
    # # Create a text representation of the headline and entities
    # headline_text = headline.lower()
    # entity_texts = [e["original"]["text"].lower() for e in entities]
    # combined_text = headline_text + " " + " ".join(entity_texts)
    
    # # Score each image
    # scored_images = []
    # for image in images:
    #     image_text = (image["title"] + " " + query).lower()
        
    #     # Calculate text similarity (simplified)
    #     # In a real implementation, you would use more sophisticated methods
    #     similarity = calculate_text_similarity(combined_text, image_text)
        
    #     scored_images.append({
    #         "image": image,
    #         "score": similarity,
    #         "entities_matched": count_entity_matches(entity_texts, image_text)
    #     })
    
    # # Sort by score
    # return sorted(scored_images, key=lambda x: x["score"], reverse=True)



def process_headlines(df):
    """Process a list of headlines to find matching images."""
    results = []
    model = SentenceTransformer('clip-ViT-B-32')
    

    for idx, row in df.iterrows():
        # print(f"Processing headline: '{row['title']}'")
        headline = row['resolved_title']
        # Extract named entities

        entities = extract_entities(headline)
        # print(f"Extracted entities: {entities}")
        
        # Entity linking
        linked_entities = entity_linking(headline,entities)
        
        # print(linked_entities,"lined")
        # Build search query
        query = get_headline_query(headline, linked_entities)
        print(f"Search query: '{query}'")
        
        # Search for images
        ref_images = search_images(query)
        # print(ref_images)
        # Match images to headline
        # cos_score = match_images_to_reference_images(row['main_img_url'], ref_images,model)
        
        # results.append({
        #     "headline": headline,
        #     "entities": entities,
        #     "linked_entities": linked_entities,
        #     "search_query": query,
        #     "matched_images": matched_images
        # })
        
    return 'abs'

def visualize_results(results):
    """Visualize the top matched image for each headline."""
    num_headlines = len(results)
    fig, axes = plt.subplots(num_headlines, 1, figsize=(10, 5 * num_headlines))
    
    if num_headlines == 1:
        axes = [axes]
        
    for i, result in enumerate(results):
        headline = result["headline"]
        top_image = result["matched_images"][0]["image"] if result["matched_images"] else None
        
        axes[i].text(0.5, 0.9, headline, fontsize=12, ha='center', transform=axes[i].transAxes)
        
        if top_image:
            # In a real implementation, download and display the actual image
            # For demonstration, we'll just show a placeholder
            axes[i].text(0.5, 0.5, f"Image: {top_image['title']}", 
                            fontsize=10, ha='center', transform=axes[i].transAxes)
            axes[i].text(0.5, 0.4, f"Score: {result['matched_images'][0]['score']:.2f}", 
                            fontsize=10, ha='center', transform=axes[i].transAxes)
            
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

# %%
df=pd.read_csv("../../datasets/processed/all_data_df_resolved.csv")
# %%

# df=pd.read_csv("../../datasets/processed/all_data_df_resolved.csv")







cos_score= process_headlines(df[7:8])


print(df["main_img_url"][7])
# %%
ref_images = google_image_search('Donald trump',2)
arr= np.array(ref_images)
print(arr.ndim,len(arr))
# print(os.getenv('TAG_ME_TOKEN'))
# text = "Hello World !"
# print(df['title'][2])
# headline= matcher.nlp("Remember When 'Figaro' Was Set in Trump Tower?")
# annotations = tagme.annotate(headline)
# print(headline)
# headline=matcher.nlp("Luring Chinese Investors With Trump's Name, and Little Else")

# annotations = tagme.annotate("Remember When 'Figaro' Was Set in Trump Tower?")
# for ann in annotations.get_annotations(0.1):
#     print(f'Score : {ann.score}')
#     print(f'Begin : {ann.begin}')
#     print(f'End : {ann.end}')
#     print(f'Id : {ann.entity_id}')
#     print(f'Mention : {ann.mention}')
#     print(f'Title : {ann.entity_title}')
#     print('--------------')
# df.head(20)

# %%
ref_images
# %%
