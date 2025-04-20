import re
import unicodedata
from bs4 import BeautifulSoup

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