


import datetime
import os
import re
import textwrap

import openai

from constants.paths import IMG_PATH
from utils.image_utils import encode_image, get_image_filename
from utils.img_web_scraper import google_image_search
from utils.validators import is_url


def match_image_title(df,index,id_col,title_col,scraped_imgs_url):

   # Construct the image path dynamically
   if is_url(df[id_col].iloc[index]):
       image_path_url = get_image_filename(df[id_col].iloc[index])
       image_path = os.path.join(IMG_PATH, f"{image_path_url}.jpg")
   else:    
       image_path = os.path.join(IMG_PATH, f"{df[id_col].iloc[index]}.jpg")
   encoded_image = encode_image(image_path)
   print(encoded_image)
#    display(Image(encoded_image))

   result = openai.chat.completions.create(
     model="gpt-4-turbo",
     messages = [{
      "role": "user",
      "content": [
        {"type": "text", "text": f"This is title {df[title_col].iloc[index]}"},
        {"type": "text", "text": "Do you think title correctly describes the first image? If you are unsure about it check reference images and match first image with reference images.If it is a person in first image, match first image with reference images and see if they are same person.Respond with yes or no and briefly explain"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
       ] + [
        {"type": "image_url", "image_url": {"url": ref_img}}
        for ref_img in scraped_imgs_url 
      ]
   }],
    max_tokens=200
   )
   return textwrap.fill(result.choices[0].message.content, width=70)


def generate_query(df, title_col, date_col, index):
    date_val = df[date_col].iloc[index]
    print(date_val)
    # Check if the value is numeric (UNIX timestamp)
    if isinstance(date_val, (int, float)):
        dt = datetime.datetime.utcfromtimestamp(date_val)
    elif isinstance(date_val, str):
        try:
            dt = datetime.datetime.strptime(date_val, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return f"{df[title_col].iloc[index]}"  # Fallback if format is unexpected
    else:
        return f"{df[title_col].iloc[index]}"  # Fallback for unknown types

    year = dt.year
    return f"{df[title_col].iloc[index]} {year}"


# Example usage:
def predict_news_using_similarity(df,index,id_col,title_col,date_col):
   query=generate_query(df,title_col,date_col,index)
   print(query)
   images = google_image_search(f"{query}", num_images=5)
   scraped_imgs_url=[]
   for i, url in enumerate(images, 1):
     scraped_imgs_url.append(url[0])
     # print(f"Image {i}: {url[0]}")
    
   content=match_image_title(df,index,id_col,title_col,scraped_imgs_url)
    
   match = re.match(r"^(Yes|No)[,:\s-]*", content, re.IGNORECASE)
   if match:
        answer = match.group(1).lower()
        binary_label = 1 if answer == "yes" else 0
        reasoning = content[len(match.group(0)):].strip()
   else:
        binary_label = None
        reasoning = content

    # Save to the dataframe
   df.at[index, 'match_label'] = binary_label
   df.at[index, 'match_reasoning'] = reasoning
   df.at[index, 'scraped_reference_imgs']=scraped_imgs_url