# %%
from transformers import BertTokenizer, BertModel, pipeline
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import spacy
import pandas as pd
from PIL import Image as PILImage
# %%
# Load spaCy English model    

spacy.require_gpu()
nlp = spacy.load("en_core_web_trf")
# %%
device = 0 if torch.cuda.is_available() else -1
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')
# device = 0 if torch.cuda.is_available() else -1
def get_text_embeddings(text_list):
    # text_list: list of strings
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_size)
    sentence_embeddings = embeddings.mean(dim=1)  # shape: (batch_size, hidden_size)

    return sentence_embeddings  # shape: (N, 768)


# %%
def token_count():
    all_data_df=pd.read_csv("../../datasets/raw/all_data.csv")
    all_data_df = all_data_df.loc[:, ~all_data_df.columns.str.contains('^Unnamed')]
    tqdm.pandas()
    all_data_df['token_count'] = all_data_df['clean_text'].progress_apply(lambda x: len(tokenizer.tokenize(str(x))))

    # Step 2: Plot token distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(all_data_df['token_count'], bins=50, kde=True, color='skyblue')
    plt.axvline(512, color='red', linestyle='--', label='BERT Token Limit (512)')
    plt.title('Token Count Distribution per Row')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Number of Texts')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
# %%
def analyze_named_entities(df, text_column):
    """
    Counts named entities in each row of a text column and shows a bar plot.

    Args:
        df (pd.DataFrame): DataFrame containing the text column.
        text_column (str): Name of the column containing text.
    
    Returns:
        pd.DataFrame: DataFrame with an added 'num_entities' column.
    """

    # Use nlp.pipe for efficient processing
    texts = df[text_column].fillna("").astype(str).tolist()
    tqdm.pandas()
    entity_counts = [len(doc.ents) for doc in nlp.pipe(texts)]

    # Add result to DataFrame
    df['num_entities'] = entity_counts

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['num_entities'])
    plt.xlabel("Row Index")
    plt.ylabel("Number of Named Entities")
    plt.title("Named Entities per Text Row")
    plt.tight_layout()
    plt.show()
# %%
df=pd.read_csv("../../datasets/processed/all_data_df_resolved.csv")
analyze_named_entities(df,"resolved_text")
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
embedding_df = pd.read_pickle("../embeddings/image_embeddings_400.pkl")
embedding_df["index"][399]
# %%
embedding_df["embedding_count"] = embedding_df["referenced_image_embeddings"].apply(len)


plt.figure(figsize=(10, 6))
sns.countplot(x="embedding_count", data=embedding_df)
plt.title("Number of Image Embeddings per Row")
plt.xlabel("Number of Embeddings")
plt.ylabel("Row Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# %%
embedding_df["embedding_count"] = embedding_df["referenced_image_embeddings"].apply(len)
image_counts = embedding_df["embedding_count"].value_counts().sort_index()

# Step 3: Plot with marker='o' and linestyle='-'
plt.figure(figsize=(10, 6))
plt.plot(image_counts.index, image_counts.values, marker='o', linestyle='-')
plt.title("Number of Rows vs Number of Image Embeddings")
plt.xlabel("Number of Embeddings per Row")
plt.ylabel("Number of Rows")
plt.grid(True)
plt.xticks(image_counts.index)  # optional: ensures x-axis ticks are aligned
plt.tight_layout()
plt.show()
# %%
