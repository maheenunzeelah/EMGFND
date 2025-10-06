import pandas as pd
from utils.pipeline_utils import extract_text_arrays_from_column, get_batch, load_images_for_batch, media_eval_load_images_for_batch, process_img_embeddings_batch, process_text_embeddings_batch
import json
import config
import pickle

include_resolved_text = False
embedding_type = "text"
dataset = config.dataset
df=pd.read_csv(config.media_eval_train_path)
df = df.rename(columns={
    'tweetText': 'resolved_title',
})
embedding_df = pd.read_pickle(config.clip_text_embeddings_path)

if embedding_type == 'image':
    embedding_df = pd.read_pickle(config.media_eval_clip_img_title_embeddings_path.split('.')[0]+"_test_1000.pkl")
    image_dir = config.image_dir
    batch_df = get_batch(1000,len(df),df)

    all_imgs = media_eval_load_images_for_batch(batch_df, image_dir)

    
    all_embeddings = process_img_embeddings_batch(all_imgs, df, batch_df, model=config.image_embed_model)

    # Create batch embedding dataframe
    batch_embedding_df = pd.DataFrame({
        "index": batch_df.index,  # preserves mapping to original df
        "referenced_image_embeddings": all_embeddings
    })


    # # # Append to the master embedding DataFrame
    embedding_df = pd.concat([embedding_df, batch_embedding_df], ignore_index=True)
    embedding_df.to_pickle(config.media_eval_clip_img_title_embeddings_path.split('.')[0]+"_test.pkl")

else:
    embedding_df = pd.read_pickle(config.media_eval_clip_title_embedding_path.split('.')[0]+"_train_3000.pkl")

    batch_df = get_batch(3000, len(df), df)
    all_texts = extract_text_arrays_from_column(batch_df, column_name="entity_titles")
    # print(all_texts,"printt")

    text_embeddings = process_text_embeddings_batch(all_texts, batch_df, include_resolved_text=include_resolved_text)

    batch_embedding_df = pd.DataFrame({
        "index": batch_df.index,
        "text_embeddings": text_embeddings
    })

    embedding_df = pd.concat([embedding_df, batch_embedding_df], ignore_index=True) 
    embedding_df.to_pickle(config.media_eval_clip_title_embedding_path.split('.')[0]+"_train.pkl")
# else:
#     if embedding_type == 'text':
        # batch_df = get_batch(0,800,df)

        # with open(config.text_json_path_1,"r",encoding="utf-8") as f1, open(config.text_json_path_2,"r",encoding="utf-8") as f2:
        #     text1 = json.load(f1)
        #     text2 = json.load(f2)

        # # Merge lists
        # all_texts = text1 + text2

        # text_embeddings = process_text_embeddings_batch(all_texts, batch_df, include_resolved_text=include_resolved_text)


        # batch_embedding_df = pd.DataFrame({
        #     "index": batch_df.index,  # preserves mapping to original df
        #     "text_embeddings": text_embeddings
        # })

        # # embedding_df = pd.concat([embedding_df, batch_embedding_df], ignore_index=True)
        # batch_embedding_df.to_pickle(config.clip_text_embeddings_path)

#     else:
#         embedding_df = pd.read_pickle(config.resnet_img_text_embeddings_path.split('.')[0]+"_3500.pkl")
        
#         image_dir = config.image_dir
#         batch_df = get_batch(3500,len(df),df)
    
#         all_imgs = load_images_for_batch(batch_df, image_dir)

        
#         all_embeddings = process_img_embeddings_batch(all_imgs, df, batch_df, model=config.image_embed_model)
#         # Create batch embedding dataframe
#         batch_embedding_df = pd.DataFrame({
#             "index": batch_df.index,  # preserves mapping to original df
#             "referenced_image_embeddings": all_embeddings
#         })


#     # # # Append to the master embedding DataFrame
#     embedding_df = pd.concat([embedding_df, batch_embedding_df], ignore_index=True)
#     embedding_df.to_pickle(config.resnet_img_text_embeddings_path)


with open(config.media_eval_bert_title_train_embeddings_path, "rb") as f:
    df1 = pickle.load(f)

# Load the second pickle file
with open(config.media_eval_bert_title_test_embeddings_path, "rb") as f:
    df2 = pickle.load(f)

# Concatenate them
final_df = pd.concat([df1, df2], ignore_index=True)

# Save back to pickle if needed
final_df.to_pickle("src/embeddings/media_eval_bert_title_embeddings/title_embeddings.pkl")

print("Final shape:", final_df.shape)

print(final_df.head())
# media_eval_train_path
# media_eval_test_path
