import pandas as pd
from pipelines.utils import get_batch, get_text_embedding, process_text_embeddings_batch
import json

include_resolved_text = True

df=pd.read_csv("datasets/processed/all_data_df_resolved.csv")
embedding_df = pd.read_pickle("src/bert-text-embeddings/text_embeddings_1500.pkl")

batch_df = get_batch(0,1500,df)

with open("../bert-text-embeddings/all_texts_1500.json", "r", encoding="utf-8") as f:
    all_texts = json.load(f)

text_embeddings = process_text_embeddings_batch(all_texts, df , get_text_embedding, include_resolved_text=include_resolved_text)


batch_embedding_df = pd.DataFrame({
    "index": batch_df.index,  # preserves mapping to original df
    "text_embeddings": text_embeddings
})

embedding_df = pd.concat([embedding_df, batch_embedding_df], ignore_index=True)
embedding_df.to_pickle("src/bert-text-embeddings/text_embeddings_3950.pkl")
