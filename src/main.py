import pandas as pd
from utils.data_cleaning import remove_na_rows, subset_df

train_df=pd.read_table("datasets/raw/multimodal_train.tsv")
# val_df=pd.read_table("multimodal_validate.tsv")
# test_df=pd.read_table("multimodal_test_public.tsv")

train_df_cleaned=subset_df(remove_na_rows(train_df,'image_url'),['id','clean_title', 'created_utc', 'image_url','2_way_label'])
# val_df_cleaned=subset_df(remove_na_rows(val_df,'image_url'),['id','clean_title', 'created_utc', 'image_url','2_way_label'])
# test_df_cleaned=subset_df(remove_na_rows(test_df,'image_url'),['id','clean_title', 'created_utc', 'image_url','2_way_label'])
print(train_df_cleaned.head())