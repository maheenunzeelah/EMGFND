# %%
import re
from IPython.display import Markdown, display, update_display, Image
import base64
import requests
import os
import io
from PIL import Image as PILImage
from functools import lru_cache
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import glob

def get_image_filename(url):
    # Use regular expression to find the part after '.com/' and before '.jpg'
    match = re.search(r'\.com/(.*?)(?=\.[a-zA-Z]{3,4})', url)  # Match part between '.com/' and file extension
    if match:
        return match.group(1).replace('/', '_')  # Replace '/' with '_' for safe filenames
    return None  # Return None if no match is found


def encode_image(image_source):
    print(image_source)
    if image_source.startswith("http://") or image_source.startswith("https://"):
        # Handle URL
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(image_source,headers=headers)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
        else:
            raise Exception(f"Failed to fetch image from URL: {image_source}")
    elif os.path.exists(image_source):
        # Handle local file
        with open(image_source, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        raise ValueError("Invalid image source. Provide a valid file path or image URL.")
    

def load_and_display_image(img_input, folder='allData_images'):
    # Case 1: If input is a PIL image
    if isinstance(img_input, PILImage.Image):
        display(img_input)
        return

    # Case 2: If input is a URL
    if isinstance(img_input, str):
        # Try to extract filename
        filename = get_image_filename(img_input)
        if filename:
            image_path = os.path.join(folder, filename + '.jpg')
            if os.path.exists(image_path):
                display(PILImage.open(image_path))
                return
            else:
                # Try to load from URL directly
                try:
                    response = requests.get(img_input)
                    if response.status_code == 200:
                        img = PILImage.open(io.BytesIO(response.content))
                        display(img)
                        return
                    else:
                        print(f"Failed to load image from URL: {img_input}")
                except Exception as e:
                    print(f"Error fetching image from URL: {e}")
        else:
            print("Invalid URL format.")
    else:
        print("Unsupported input type. Provide a URL, PIL.Image, or a known filename.")

    return None


def pil_image_to_bytes(img: PILImage):
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()

# Create a session with connection pooling and retry strategy
def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(
        pool_connections=20,
        pool_maxsize=100,
        max_retries=retry_strategy
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    return session

# Global session for connection reuse
session = create_session()

@lru_cache(maxsize=1000)
def get_pil_image_cached(input_data):
    """Cached version of get_pil_image for repeated URLs/paths"""
    return get_pil_image_fast(input_data)

def get_pil_image_fast(input_data):
    print("org fileeee",input_data)
    """Optimized version using global session"""
    if not isinstance(input_data, str):
        raise TypeError("Input must be a URL, base64 string, or file path string.")
    
    if input_data.startswith(("http://", "https://")):
        try:
            response = session.get(input_data, timeout=30)
            response.raise_for_status()
            return PILImage.open(io.BytesIO(response.content))
        except Exception as e:
            raise RuntimeError(f"Error downloading image from URL: {e}") from e
    
    if input_data.startswith("data:image"):
        try:
            _, base64_data = input_data.split(",", 1)
            image_data = base64.b64decode(base64_data)
            return PILImage.open(io.BytesIO(image_data))
        except Exception as e:
            raise RuntimeError(f"Error decoding base64 data URI: {e}") from e
    
    if "/" in input_data or "\\" in input_data or "." in input_data:
        if os.path.isfile(input_data):

            try:
                return PILImage.open(input_data)
            except Exception as e:
                raise RuntimeError(f"Error opening image file: {e}") from e
    
    if len(input_data) > 100:
        try:
            image_data = base64.b64decode(input_data, validate=True)
            return PILImage.open(io.BytesIO(image_data))
        except Exception as e:
            raise RuntimeError(f"Error decoding base64 image: {e}") from e
    
    raise ValueError("Input string is neither a valid URL, file path, nor a base64 image string.")

def bytes_to_pil_image(byte_data: bytes):
    return PILImage.open(io.BytesIO(byte_data))

def find_image_file(base_filename, image_dir):
    """Find an image file with any extension for the given base filename."""
    pattern = os.path.join(image_dir, base_filename + ".*")
    matches = glob.glob(pattern)
    return matches[0] if matches else None
