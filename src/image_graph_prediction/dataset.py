
#%%
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from image_graph_prediction.graph import create_fully_connected_graph
import glob
#%%
def create_graph_dataset(all_img_embeddings, all_text_embeddings, labels):


    graphs = [create_fully_connected_graph(embed, label) for embed, label in zip(all_embeddings, labels)]
    print(graphs)


    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, stratify=labels)
    train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.1, stratify=[g.y.item() for g in train_graphs])

    print(f"Number of training graphs: {len(train_graphs)}")
    print(f"Number of validation graphs: {len(val_graphs)}")
    print(f"Number of test graphs: {len(test_graphs)}")

    torch.save(train_graphs, 'resnet-graphs/train_graphs.pt')
    torch.save(val_graphs, 'resnet-graphs/val_graphs.pt')
    torch.save(test_graphs, 'resnet-graphs/test_graphs.pt')

# pickle_files = sorted(glob.glob("src/embeddings/image_embeddings_*.pkl"))
# embedding_dfs = [pd.read_pickle(f) for f in pickle_files]
# all_embeddings = merge_embeddings(embedding_dfs)
# print(type(all_embeddings))
# print(all_embeddings.info())
#%%

df=pd.read_csv("../../datasets/processed/all_data_df_resolved.csv")
img_embeddings_df = pd.read_pickle("../resnet-embeddings/image_embeddings_3954.pkl")
text_embeddings_df = pd.read_pickle("../resnet-embeddings/image_embeddings_3950.pkl")
valid_df = df[df['resolved_text'].notnull()].reset_index(drop=True)

# Step 2: Filter embeddings_df to match the valid rows
# Assuming both dataframes are aligned row-wise (i.e., same original index)
valid_img_embeddings_df = img_embeddings_df.loc[valid_df.index].reset_index(drop=True)
valid_text_embeddings_df = text_embeddings_df.loc[valid_df.index].reset_index(drop=True)

all_img_embeddings = valid_img_embeddings_df['referenced_image_embeddings'].values
all_text_embeddings = valid_text_embeddings_df['referenced_image_embeddings'].values

valid_df['label'] = valid_df['type'].map({'real': 1, 'fake': 0})
# %%
create_graph_dataset(all_img_embeddings, all_text_embeddings, valid_df['label'].values)
# %%
all_img_embeddings
# %%
df.info()
# %%
df=pd.read_csv("../../datasets/processed/all_data_df_resolved.csv")
# %%
print(df['type'].value_counts())
# %%
