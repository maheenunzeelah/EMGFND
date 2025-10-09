import pandas as pd
from utils.pipeline_utils import extract_text_arrays_from_column, get_batch, load_images_for_batch, process_img_embeddings_batch, process_text_embeddings_batch
import json
import config
import pickle

include_resolved_text = False # Whether to include resolved text in embeddings for long articles
embedding_type = "text"

df=pd.read_csv(config.all_data_text_entities_df)

if embedding_type == 'image':
    image_dir = config.image_dir
    batch_df = get_batch(0,1,df) # Use get_batch(0,1,df) for first row, get_batch(1,5,df) for rows 1 to 5 etc. This will help in batching the processing.

    all_imgs = load_images_for_batch(batch_df, image_dir)

    all_embeddings = process_img_embeddings_batch(all_imgs, df, batch_df, model=config.image_embed_model)

    # Create batch embedding dataframe
    batch_embedding_df = pd.DataFrame({
        "index": batch_df.index,  # preserves mapping to original df
        "referenced_image_embeddings": all_embeddings
    })


    # # # Append to the master embedding DataFrame
    # batch_embedding_df = pd.concat([embedding_df, batch_embedding_df], ignore_index=True)
  
else:
    # embedding_df = pd.read_pickle(config.media_eval_clip_title_embedding_path.split('.')[0]+"_train_3000.pkl") # Load existing embeddings if continuing from a previous batch
    batch_df = get_batch(1, 5, df)
    all_texts = extract_text_arrays_from_column(batch_df, column_name="entity_titles")

    text_embeddings = process_text_embeddings_batch(all_texts, batch_df, include_resolved_text=include_resolved_text)

    batch_embedding_df = pd.DataFrame({
        "index": batch_df.index,
        "text_embeddings": text_embeddings
    })

    # embedding_df = pd.concat([embedding_df, batch_embedding_df], ignore_index=True) # Append to the master embedding DataFrame if continuing from a previous batch
    batch_embedding_df.to_pickle(config.media_eval_clip_title_embedding_path) # Save after processing each batch to avoid data loss. Use embedding_df if continuing from a previous batch.

