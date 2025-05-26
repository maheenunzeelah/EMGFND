import re
import unicodedata
from bs4 import BeautifulSoup
import datetime

# Map of common Unicode code points to their ASCII equivalents
unicode_replacements = {
    '2019': "'",  # right single quote
    '2018': "'",  # left single quote
    '201C': '"',  # left double quote
    '201D': '"',  # right double quote
    '2013': '-',  # en dash
    '2014': '-',  # em dash
}

def replace_unicode_placeholders(text):
    def replacer(match):
        code = match.group(1).upper()
        return unicode_replacements.get(code, '')  # Replace known codes or remove
    return re.sub(r'<U\+([0-9A-Fa-f]{4})>', replacer, text)

def clean_text(text, lowercase=True, remove_punct=True, remove_html=True):
    """
    Cleans text by removing HTML, punctuation (including apostrophes and dashes), and normalizing.
    """
    if not isinstance(text, str):
        return ''
    
    # Replace unicode placeholders
    text = replace_unicode_placeholders(text)
    
    # Remove HTML
    if remove_html:
        text = BeautifulSoup(text, "html.parser").get_text()
    
    # Normalize Unicode
    text = unicodedata.normalize("NFKD", text)
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove punctuation including apostrophes and dashes
    if remove_punct:
        text = re.sub(r"[^\w\s]", '', text)
    
    # Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_na_rows(df,col):
   return df.dropna(subset=[col])


def subset_df(df,cols):
   return df[cols]


def clean_ocr_text(text):
    # Replace multiple spaces/newlines/tabs with a single space
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text) 
    # Strip leading/trailing spaces
    text = text.strip()
    
    # Optionally, remove unwanted characters (if any)
    # For example, remove non-ASCII or non-printable chars:
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text

def extract_year(date_val):
    if isinstance(date_val, (int, float)):
        dt = datetime.datetime.utcfromtimestamp(date_val)
    elif isinstance(date_val, str):
        try:
            dt = datetime.datetime.strptime(date_val, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return ""  # Fallback if format is unexpected
    else:
        return ""  # Fallback for unknown types

    year = dt.year
    return year

def is_number_like(text):
    """Return True if the text is mostly numeric (e.g., '25', '2023', '3rd')"""
    text = text.strip().lower()
    return bool(re.fullmatch(r'[\d,.\-–/]+(st|nd|rd|th)?', text)) or text.isdigit()

def normalize_name_part(part):
    # Lowercase, remove trailing apostrophe s, and remove non-alpha chars
    part = part.lower()
    part = re.sub(r"'s$", "", part)          # Remove trailing 's (apostrophe s)
    part = re.sub(r"[^a-z]", "", part)       # Remove all non-alpha chars
    return part