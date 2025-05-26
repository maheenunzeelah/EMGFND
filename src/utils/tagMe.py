'''
This module provides a wrapper for the TagMe API.
'''

import json
import logging
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import unquote
from dataclasses import dataclass
from typing import Optional, Union
import dateutil.parser
import requests
from PIL import Image as PILImage

__all__ = [
    'annotate', 'mentions', 'relatedness_wid', 'relatedness_title', 'Annotation',
    'AnnotateResponse', 'Mention', 'MentionsResponse', 'Relatedness', 'RelatednessResponse',
    'normalize_title', 'title_to_uri', 'get_entity_image', 'get_entity_images', 'WikipediaImage',
    ]

__author__ = 'Marco Cornolti <cornolti@di.unipi.it>'

DEFAULT_TAG_API = "https://tagme.d4science.org/tagme/tag"
DEFAULT_SPOT_API = "https://tagme.d4science.org/tagme/spot"
DEFAULT_REL_API = "https://tagme.d4science.org/tagme/rel"
DEFAULT_LANG = "en"
DEFAULT_LONG_TEXT = 3
WIKIPEDIA_URI_BASE = "https://{}.wikipedia.org/wiki/{}"
WIKIPEDIA_API_BASE = "https://{}.wikipedia.org/w/api.php"
MAX_RELATEDNESS_PAIRS_PER_REQUEST = 100
GCUBE_TOKEN = None
HTML_PARSER = HTMLParser()

@dataclass
class WikipediaImageResult:
    """Extended result that includes both PIL image and metadata"""
    pil_image: PILImage.Image
    title: str
    description: str
    width: Optional[int]
    height: Optional[int]
    thumbnail_url: str
    page_url: str
    original_url: Optional[str] = None
    
class WikipediaImage:
    '''
    A Wikipedia image with metadata.
    '''
    def __init__(self, url, title=None, description=None, width=None, height=None, 
                 thumbnail_url=None, page_url=None):
        self.url = url
        self.title = title or ""
        self.description = description or ""
        self.width = width
        self.height = height
        self.thumbnail_url = thumbnail_url
        self.page_url = page_url
        
        
    def __str__(self):
        return f"WikipediaImage(title='{self.title}', url='{self.url}')"
    
    def __repr__(self):
        return self.__str__()


class Annotation:
    '''
    An annotation, i.e. a link of a part of text to an entity.
    '''
    def __init__(self, ann_json):
        self.begin = int(ann_json.get("start"))
        self.end = int(ann_json.get("end"))
        self.entity_id = int(ann_json.get("id"))
        self.entity_title = ann_json.get("title")
        self.score = float(ann_json.get("rho"))
        self.mention = ann_json.get("spot")

    def __str__(self):
        return "{} -> {} (score: {})".format(self.mention, self.entity_title, self.score)

    def uri(self, lang=DEFAULT_LANG):
        '''
        Get the URI of this annotation entity.
        :param lang: the Wikipedia language.
        '''
        return title_to_uri(self.entity_title, lang)
    
    def get_image(self, lang=DEFAULT_LANG, image_size='small'):
        '''
        Get the main image for this entity from Wikipedia.
        :param lang: the Wikipedia language.
        :param image_size: 'small', 'medium', 'large', or 'original'
        '''
        return get_entity_image(self.entity_title, lang, image_size)
    
    def get_images(self, lang=DEFAULT_LANG, limit=5):
        '''
        Get multiple images for this entity from Wikipedia.
        :param lang: the Wikipedia language.
        :param limit: maximum number of images to return
        '''
        return get_entity_images(self.entity_title, lang, limit)


class AnnotateResponse:
    '''
    A response to a call to the annotation (/tag) service. It contains the list of annotations
    found.
    '''
    def __init__(self, json_content):
        self.annotations = [Annotation(ann_json) for ann_json in json_content["annotations"] if "title" in ann_json]
        self.time = int(json_content["time"])
        self.lang = json_content["lang"]
        self.timestamp = dateutil.parser.parse(json_content["timestamp"])
        self.original_json = json_content

    def get_annotations(self, min_rho=None):
        '''
        Get the list of annotations found.
        :param min_rho: if set, only get entities with a rho-score (confidence) higher than this.
        '''
        return (a for a in self.annotations if min_rho is None or a.score > min_rho)

    def __str__(self):
        return "{}msec, {} annotations".format(self.time, len(self.annotations))


class Mention:
    '''
    A mention, i.e. a part of text that may mention an entity.
    '''
    def __init__(self, mention_json):
        self.begin = int(mention_json.get("start"))
        self.end = int(mention_json.get("end"))
        self.linkprob = float(mention_json.get("lp"))
        self.mention = mention_json.get("spot")

    def __str__(self):
        return "{} [{},{}] lp={}".format(self.mention, self.begin, self.end, self.linkprob)


class MentionsResponse:
    '''
    A response to a call to the mention finding (/spot) service. It contains the list of mentions
    found.
    '''
    def __init__(self, json_content):
        self.mentions = [Mention(mention_json) for mention_json in json_content["spots"]]
        self.time = int(json_content["time"])
        self.lang = json_content["lang"]
        self.timestamp = dateutil.parser.parse(json_content["timestamp"])

    def get_mentions(self, min_lp=None):
        '''
        Get the list of mentions found.
        :param min_lp: if set, only get mentions with a link probability higher than this.
        '''
        return (m for m in self.mentions if min_lp is None or m.linkprob > min_lp)

    def __str__(self):
        return "{}msec, {} mentions".format(self.time, len(self.mentions))


class Relatedness:
    '''
    A relatedness, i.e. a real value between 0 and 1 indicating how semantically close two entities
    are.
    '''
    def __init__(self, rel_json):
        self.title1, self.title2 = (wiki_title(t) for t in rel_json["couple"].split(" "))
        self.rel = float(rel_json["rel"]) if "rel" in rel_json else None

    def as_pair(self):
        '''
        Get this relatedness value as a pair (titles, rel), where rel is the relatedness value and
        titles is the pair of the two titles/Wikipedia IDs.
        '''
        return ((self.title1, self.title2), self.rel)

    def __str__(self):
        return "{}, {} rel={}".format(self.title1, self.title2, self.rel)


class RelatednessResponse:
    '''
    A response to a call to the relatedness (/rel) service. It contains the list of relatedness for
    each pair.
    '''
    def __init__(self, json_contents):
        self.relatedness = [Relatedness(rel_json)
                            for json_content in json_contents
                            for rel_json in json_content["result"]]
        self.lang = json_contents[0]["lang"]
        self.timestamp = dateutil.parser.parse(json_contents[0]["timestamp"])
        self.calls = len(json_contents)

    def __iter__(self):
        for rel in self.relatedness:
            yield rel.as_pair()

    def get_relatedness(self, i=0):
        '''
        Get the relatedness of a pairs of entities.
        :param i: the index of an entity pair. The order is the same as the request.
        '''
        return self.relatedness[i].rel

    def __str__(self):
        return "{} relatedness pairs, {} calls".format(len(self.relatedness), self.calls)


def normalize_title(title):
    '''
    Normalize a title to Wikipedia format. E.g. "barack Obama" becomes "Barack_Obama"
    :param title: a title to normalize.
    '''
    title = title.strip().replace(" ", "_")
    return title[0].upper() + title[1:]


def wiki_title(title):
    '''
    Given a normalized title, get the page title. E.g. "Barack_Obama" becomes "Barack Obama"
    :param title: a wikipedia title.
    '''
    return unescape(title.strip(" _").replace("_", " "))


def title_to_uri(entity_title, lang=DEFAULT_LANG):
    '''
    Get the URI of the page describing a Wikipedia entity.
    :param entity_title: an entity title.
    :param lang: the Wikipedia language.
    '''
    return WIKIPEDIA_URI_BASE.format(lang, normalize_title(entity_title))


def annotate(text, gcube_token=None, lang=DEFAULT_LANG, api=DEFAULT_TAG_API,
             long_text=DEFAULT_LONG_TEXT):
    '''
    Annotate a text, linking it to Wikipedia entities.
    :param text: the text to annotate.
    :param gcube_token: the authentication token provided by the D4Science infrastructure.
    :param lang: the Wikipedia language.
    :param api: the API endpoint.
    :param long_text: long_text parameter (see TagMe documentation).
    '''
    payload = [("text", text.encode("utf-8")),
            #    ("long_text", long_text),
               ("lang", lang)]
    print(gcube_token,"gcubbee")
    json_response = _issue_request(api, payload, gcube_token)
    return AnnotateResponse(json_response) if json_response else None


def mentions(text, gcube_token=None, lang=DEFAULT_LANG, api=DEFAULT_SPOT_API):
    '''
    Find possible mentions in a text, do not link them to any entity.
    :param text: the text where to find mentions.
    :param gcube_token: the authentication token provided by the D4Science infrastructure.
    :param lang: the Wikipedia language.
    :param api: the API endpoint.
    '''
    payload = [("text", text.encode("utf-8")),
               ("lang", lang.encode("utf-8"))]
    json_response = _issue_request(api, payload, gcube_token)
    return MentionsResponse(json_response) if json_response else None


def relatedness_wid(wid_pairs, gcube_token=None, lang=DEFAULT_LANG, api=DEFAULT_REL_API):
    '''
    Get the semantic relatedness among pairs of entities. Entities are indicated by their
    Wikipedia ID (an integer).
    :param wid_pairs: either one pair or a list of pairs of Wikipedia IDs.
    :param gcube_token: the authentication token provided by the D4Science infrastructure.
    :param lang: the Wikipedia language.
    :param api: the API endpoint.
    '''
    return _relatedness("id", wid_pairs, gcube_token, lang, api)


def relatedness_title(tt_pairs, gcube_token=None, lang=DEFAULT_LANG, api=DEFAULT_REL_API):
    '''
    Get the semantic relatedness among pairs of entities. Entities are indicated by their
    Wikipedia ID (an integer).
    :param tt_pairs: either one pair or a list of pairs of entity titles.
    :param gcube_token: the authentication token provided by the D4Science infrastructure.
    :param lang: the Wikipedia language.
    :param api: the API endpoint.
    '''
    return _relatedness("tt", tt_pairs, gcube_token, lang, api)


def _relatedness(pairs_type, pairs, gcube_token, lang, api):
    if not isinstance(pairs[0], (list, tuple)):
        pairs = [pairs]

    # In Python 3, all strings are Unicode by default
    # Check if we have bytes and decode them
    if isinstance(pairs[0][0], bytes):
        pairs = [(p[0].decode("utf-8"), p[1].decode("utf-8")) for p in pairs]

    # Normalize titles if they are strings
    if isinstance(pairs[0][0], str):
        pairs = [(normalize_title(p[0]), normalize_title(p[1])) for p in pairs]

    json_responses = []
    for chunk in range(0, len(pairs), MAX_RELATEDNESS_PAIRS_PER_REQUEST):
        payload = [("lang", lang)]
        payload += ((pairs_type, "{} {}".format(p[0], p[1]))
                    for p in pairs[chunk:chunk + MAX_RELATEDNESS_PAIRS_PER_REQUEST])
        json_responses.append(_issue_request(api, payload, gcube_token))
    return RelatednessResponse(json_responses) if json_responses and json_responses[0] else None


def get_entity_image(entity_title, lang=DEFAULT_LANG, image_size='medium'):
    '''
    Get the main image for a Wikipedia entity.
    :param entity_title: the entity title (e.g., "Barack_Obama")
    :param lang: the Wikipedia language code
    :param image_size: 'small' (150px), 'medium' (300px), 'large' (500px), or 'original'
    :return: WikipediaImage object or None
    '''
    try:
        # Clean the title
        clean_title = wiki_title(entity_title)
        
        # Wikipedia API endpoint
        api_url = WIKIPEDIA_API_BASE.format(lang)
        
        # Parameters for getting page info with main image
        params = {
            'action': 'query',
            'format': 'json',
            'titles': clean_title,
            'prop': 'pageimages|pageterms',
            'pithumbsize': _get_image_size_pixels(image_size),
            'pilimit': 1,
            'wbptterms': 'description'
        }
        print(api_url,"api_url")
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        
        for page_id, page_data in pages.items():
            if page_id == '-1':  # Page not found
                continue
                
            # Get thumbnail info
            thumbnail = page_data.get('thumbnail')
            if not thumbnail:
                continue
            
            # Get original image info
            pageimage = page_data.get('pageimage')
            original_url = None
            
            if pageimage:
                # Get original image URL
                image_params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': f'File:{pageimage}',
                    'prop': 'imageinfo',
                    'iiprop': 'url|size|extmetadata'
                }
                
                img_response = requests.get(api_url, params=image_params, timeout=10)
                if img_response.status_code == 200:
                    img_data = img_response.json()
                    img_pages = img_data.get('query', {}).get('pages', {})
                    
                    for img_page_id, img_page_data in img_pages.items():
                        imageinfo = img_page_data.get('imageinfo', [])
                        if imageinfo:
                            original_url = imageinfo[0].get('url')
                            break
            
            # Extract description
            description = ""
            terms = page_data.get('terms', {})
            if 'description' in terms:
                description = terms['description'][0]
            
            # Create WikipediaImage object
            return WikipediaImage(
                url=original_url or thumbnail['source'],
                title=pageimage or 'Unknown',
                description=description,
                width=thumbnail.get('width'),
                height=thumbnail.get('height'),
                thumbnail_url=thumbnail['source'],
                page_url=title_to_uri(entity_title, lang)
            )
            
    except Exception as e:
        logging.warning(f"Error fetching image for {entity_title}: {e}")
        
    return None


def get_entity_images(entity_title, lang=DEFAULT_LANG, limit=5):
    '''
    Get multiple images for a Wikipedia entity.
    :param entity_title: the entity title (e.g., "Barack_Obama")
    :param lang: the Wikipedia language code
    :param limit: maximum number of images to return
    :return: list of WikipediaImage objects
    '''
    try:
        # Clean the title
        clean_title = wiki_title(entity_title)
        
        # Wikipedia API endpoint
        api_url = WIKIPEDIA_API_BASE.format(lang)
        
        # First, get the page ID
        page_params = {
            'action': 'query',
            'format': 'json',
            'titles': clean_title,
            'prop': 'info'
        }
        print("api_url",api_url)
        response = requests.get(api_url, params=page_params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        
        page_id = None
        for pid, page_data in pages.items():
            if pid != '-1':
                page_id = pid
                break
        
        if not page_id:
            return []
        
        # Get images from the page
        img_params = {
            'action': 'query',
            'format': 'json',
            'pageids': page_id,
            'prop': 'images',
            'imlimit': limit * 2  # Get more to filter out non-photos
        }
        
        img_response = requests.get(api_url, params=img_params, timeout=10)
        img_response.raise_for_status()
        
        img_data = img_response.json()
        page_images = img_data.get('query', {}).get('pages', {}).get(page_id, {}).get('images', [])
        
        if not page_images:
            return []
        
        # Get detailed info for each image
        image_titles = [img['title'] for img in page_images]
        
        # Filter out common non-photo files
        filtered_titles = []
        for title in image_titles:
            lower_title = title.lower()
            if any(ext in lower_title for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                if not any(skip in lower_title for skip in ['commons-logo', 'edit-icon', 'wikimedia']):
                    filtered_titles.append(title)
        
        if not filtered_titles:
            return []
        
        # Get image info
        info_params = {
            'action': 'query',
            'format': 'json',
            'titles': '|'.join(filtered_titles[:limit]),
            'prop': 'imageinfo',
            'iiprop': 'url|size|extmetadata',
            'iiurlwidth': 300
        }
        
        info_response = requests.get(api_url, params=info_params, timeout=10)
        info_response.raise_for_status()
        
        info_data = info_response.json()
        info_pages = info_data.get('query', {}).get('pages', {})
        
        images = []
        for img_page_id, img_page_data in info_pages.items():
            if img_page_id == '-1':
                continue
                
            imageinfo = img_page_data.get('imageinfo', [])
            if not imageinfo:
                continue
            
            info = imageinfo[0]
            
            # Extract metadata
            extmetadata = info.get('extmetadata', {})
            description = ""
            
            if 'ImageDescription' in extmetadata:
                desc_data = extmetadata['ImageDescription']
                if isinstance(desc_data, dict) and 'value' in desc_data:
                    description = _clean_html(desc_data['value'])
            
            # Create WikipediaImage object
            image = WikipediaImage(
                url=info.get('url'),
                title=img_page_data.get('title', '').replace('File:', ''),
                description=description,
                width=info.get('width'),
                height=info.get('height'),
                thumbnail_url=info.get('thumburl'),
                page_url=title_to_uri(entity_title, lang)
            )
            
            images.append(image)
            
            if len(images) >= limit:
                break
        
        return images
        
    except Exception as e:
        logging.warning(f"Error fetching images for {entity_title}: {e}")
        return []


def _get_image_size_pixels(size):
    '''Convert size string to pixel value for Wikipedia API'''
    size_map = {
        'small': 150,
        'medium': 300,
        'large': 500,
        'original': 1000
    }
    return size_map.get(size, 300)


def _clean_html(html_text):
    '''Remove HTML tags from text'''
    if not html_text:
        return ""
    
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', html_text)
    # Decode HTML entities
    clean_text = unescape(clean_text)
    # Clean up whitespace
    clean_text = ' '.join(clean_text.split())
    
    return clean_text


def _issue_request(api, payload, gcube_token):
    if not gcube_token:
        print(GCUBE_TOKEN)
        gcube_token = GCUBE_TOKEN
    if not GCUBE_TOKEN:
        raise RuntimeError("You must define GCUBE_TOKEN before calling this function or pass the "
                           "gcube_token parameter.")

    payload.append(("gcube-token", gcube_token))
    logging.debug("Calling %s", api)
    res = requests.post(api, data=payload)
    if res.status_code != 200:
        logging.warning("Tagme returned status code %d message:\n%s", res.status_code, res.content)
        return None
    
    # In Python 3, response.content is always bytes, response.text is the decoded string
    res_content = res.text
    return json.loads(res_content)