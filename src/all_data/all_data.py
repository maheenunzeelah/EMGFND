# %%
import pandas as pd
from utils.data_cleaning import remove_na_rows, subset_df, clean_text, replace_unicode_placeholders 
from utils.image_utils import get_image_filename
import os

# List of filenames in the directory (without extension)


def process_dataset():
    all_data_df=pd.read_csv("../../datasets/raw/all_data.csv")
    all_data_df = all_data_df.loc[:, ~all_data_df.columns.str.contains('^Unnamed')]
    saved_images = {os.path.splitext(f)[0] for f in os.listdir('../../allData_images') if f.endswith('.jpg')}
    all_data_df['image_filename'] = all_data_df['main_img_url'].apply(get_image_filename)
    df_filtered = all_data_df[all_data_df['image_filename'].isin(saved_images)].copy()

    df_filtered['clean_title'] = all_data_df['title'].fillna('').apply(replace_unicode_placeholders)
    df_filtered['clean_text'] = all_data_df['text'].fillna('').apply(replace_unicode_placeholders)
    all_data_df_cleaned=subset_df(df_filtered,['title', 'text', 'clean_title', 'clean_text', 'main_img_url','published','type','image_filename'])
    all_data_df_cleaned.reset_index(drop=True, inplace=True)
    return all_data_df_cleaned
    
processed_df=process_dataset()
processed_df.to_csv('../../datasets/processed/all_data_df_processed.csv',index=False)
processed_df.tail()

# %%
