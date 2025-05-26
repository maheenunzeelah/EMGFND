# %%
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import List, Tuple
import pandas as pd
# Global variables for model initialization
tokenizer = None
model = None
nlp = None
device = None

def initialize_models():
    """Initialize BERT model, tokenizer, and SpaCy"""
    global tokenizer, model, nlp, device
    
    # Initialize BERT
    model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    model.eval()
    
    # Initialize SpaCy
    try:
        nlp = spacy.load("en_core_web_lg")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found. Please install it with:")
        print("python -m spacy download en_core_web_lg")
        raise

def preprocess_text(text: str) -> List[str]:
    """
    Clean and split text into sentences using SpaCy
    
    Args:
        text: Input text to preprocess
        
    Returns:
        List of cleaned sentences
    """
    if nlp is None:
        initialize_models()
    
    # Process text with SpaCy
    doc = nlp(text)
    
    # Extract sentences
    sentences = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        # Filter out very short sentences (less than 10 characters)
        if len(sentence_text) > 10:
            sentences.append(sentence_text)
    
    return sentences

def count_tokens(text: str) -> int:
    """
    Count tokens in text using BERT tokenizer
    
    Args:
        text: Input text
        
    Returns:
        Number of tokens
    """
    if tokenizer is None:
        initialize_models()
    
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def get_sentence_embeddings(sentences: List[str]):
    """
    Get BERT embeddings for sentences
    
    Args:
        sentences: List of sentences
        
    Returns:
        Tensor of sentence embeddings
    """
    if model is None or tokenizer is None:
        initialize_models()
    
    embeddings = []
    max_length = 512
    
    with torch.no_grad():
        for sentence in sentences:
            # Tokenize and encode
            inputs = tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get BERT outputs
            outputs = model(**inputs)
            
            # Use [CLS] token embedding as sentence representation
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(cls_embedding)
    
    return torch.cat(embeddings, dim=0)

def calculate_sentence_scores(sentence_embeddings: torch.Tensor) -> np.ndarray:
    """
    Calculate sentence importance scores using cosine similarity
    
    Args:
        sentence_embeddings: Tensor of sentence embeddings
        
    Returns:
        Array of sentence scores
    """
    # Convert to numpy for sklearn
    embeddings_np = sentence_embeddings.numpy()
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_np)
    
    # Calculate scores as sum of similarities with all other sentences
    scores = np.sum(similarity_matrix, axis=1)
    
    # Normalize scores
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    
    return scores

def select_sentences_by_tokens(sentences: List[str], scores: np.ndarray, 
                              max_tokens: int = 512) -> List[Tuple[str, float, int]]:
    """
    Select top sentences based on scores with 512 token limit constraint
    
    Args:
        sentences: List of original sentences
        scores: Array of sentence scores
        max_tokens: Maximum number of tokens in summary (default: 512)
        
    Returns:
        List of tuples (sentence, score, original_index)
    """
    # Get indices sorted by score (highest first)
    sorted_indices = np.argsort(scores)[::-1]
    
    selected = []
    current_tokens = 0
    
    for idx in sorted_indices:
        sentence = sentences[idx]
        sentence_tokens = count_tokens(sentence)
        
        # Check if adding this sentence would exceed token limit
        if current_tokens + sentence_tokens > max_tokens:
            # If this is the first sentence and it exceeds limit, truncate it
            if len(selected) == 0 and sentence_tokens > max_tokens:
                # Truncate the sentence to fit within token limit
                tokens = tokenizer.tokenize(sentence)
                truncated_tokens = tokens[:max_tokens-2]  # Leave space for [CLS] and [SEP]
                truncated_sentence = tokenizer.convert_tokens_to_string(truncated_tokens)
                selected.append((truncated_sentence, scores[idx], idx))
                current_tokens = max_tokens
            break
        
        selected.append((sentence, scores[idx], idx))
        current_tokens += sentence_tokens
    
    # If no sentences selected due to token constraints, select at least one (truncated if necessary)
    if not selected and sentences:
        best_idx = sorted_indices[0]
        sentence = sentences[best_idx]
        sentence_tokens = count_tokens(sentence)
        
        if sentence_tokens > max_tokens:
            # Truncate to fit
            tokens = tokenizer.tokenize(sentence)
            truncated_tokens = tokens[:max_tokens-2]
            truncated_sentence = tokenizer.convert_tokens_to_string(truncated_tokens)
            selected.append((truncated_sentence, scores[best_idx], best_idx))
        else:
            selected.append((sentence, scores[best_idx], best_idx))
    
    # Sort by original order to maintain text flow
    selected.sort(key=lambda x: x[2])
    
    return selected

def bert_extractive_summarize(text: str, max_tokens: int = 512) -> dict:
    """
    Generate extractive summary of the input text with 512 token limit
    
    Args:
        text: Input text to summarize
        max_tokens: Maximum number of tokens in summary (default: 512)
        
    Returns:
        Dictionary containing summary and metadata
    """
    # Preprocess text using SpaCy
    sentences = preprocess_text(text)
    
    if len(sentences) == 0:
        return {
            'summary': '',
            'original_sentences': 0,
            'summary_sentences': 0,
            'selected_sentences': [],
            'total_tokens': 0
        }
    
    if len(sentences) == 1:
        sentence = sentences[0]
        sentence_tokens = count_tokens(sentence)
        
        # If single sentence exceeds token limit, truncate it
        if sentence_tokens > max_tokens:
            tokens = tokenizer.tokenize(sentence)
            truncated_tokens = tokens[:max_tokens-2]
            sentence = tokenizer.convert_tokens_to_string(truncated_tokens)
            sentence_tokens = max_tokens
        
        return {
            'summary': sentence,
            'original_sentences': 1,
            'summary_sentences': 1,
            'selected_sentences': [(sentence, 1.0, 0)],
            'total_tokens': sentence_tokens
        }
    
    # Get sentence embeddings
    embeddings = get_sentence_embeddings(sentences)
    
    # Calculate sentence scores
    scores = calculate_sentence_scores(embeddings)
    
    # Select top sentences with token limit
    selected = select_sentences_by_tokens(sentences, scores, max_tokens)
    
    # Create summary
    summary = ' '.join([sent for sent, _, _ in selected])
    total_tokens = count_tokens(summary)
    
    return {
        'summary': summary,
        'original_sentences': len(sentences),
        'summary_sentences': len(selected),
        'selected_sentences': selected,
        'total_tokens': total_tokens,
        'all_scores': list(zip(sentences, scores.tolist()))
    }

# Example usage and demonstration
def summarize_news_text(sample_text):
    
    
    # Generate summary with 512 token limit
    result = bert_extractive_summarize(sample_text, max_tokens=512)

    return result['summary']
    
# %%

df=pd.read_csv("../../datasets/processed/all_data_df_resolved.csv")


# %%
# Initialize models
initialize_models()

# %%
df['summarized_text']= df["clean_text"].apply(summarize_news_text)
# %%

# %%
df.to_csv('../../datasets/processed/all_data_df_summarized.csv',index=False)
# %%
df.info()
# %%
