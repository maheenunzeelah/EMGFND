import re
from IPython.display import Markdown, display, update_display, Image
import base64
import requests
import os

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
    

def load_and_display_image(url, folder='allData_images'):
    filename = get_image_filename(url)
    if filename:
        image_path = os.path.join(folder, filename + '.jpg')
        if os.path.exists(image_path):
            display(Image(image_path))
        else:
            print(f"Image not found at {image_path}")
    else:
        print("Invalid URL format.")
    return None