# %%
import torch
from torchvision import models, transforms
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# %%
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# We'll use features + avgpool + first FC layer as embeddings (4096 dim)
# So we keep the features and classifier[0] layer only
class VGGEmbeddingExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super(VGGEmbeddingExtractor, self).__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        # Extract only first FC layer (4096-dim)
        self.fc = original_model.classifier[0]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

# Embedding extractor for 2048-dim output (default ResNet-50 output)
class ResNetEmbeddingExtractor(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # All layers except the final FC
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])  # up to avgpool

    def forward(self, x):
        x = self.features(x)         # [B, 2048, 1, 1]
        x = torch.flatten(x, 1)      # [B, 2048]
        return x

# Load VGG16
vgg = models.vgg16(pretrained=True)
vgg_model = VGGEmbeddingExtractor(vgg).to(device)
vgg_model.eval()

 # Load ResNet-50
resnet = models.resnet50(pretrained=True)
resnet_model = ResNetEmbeddingExtractor(resnet).to(device)
resnet_model.eval()

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

# Get image embeddings
def transform_embedding(image, model, transform, device):
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        embedding = model(image)
    return embedding.squeeze(0).cpu()   # Remove batch dim

def get_embeddings(imgs, model = "clip"):
    
    if model != "clip":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225])
        ])
        if model == "vgg":   
            embeddings = torch.stack([
            transform_embedding(p, vgg_model, transform, device) for p in imgs
            ])
        else:
            embeddings = torch.stack([
            transform_embedding(p, resnet_model, transform, device) for p in imgs
            ])    
        return embeddings      
      
    else:                                                                                                                                   
        # Ensure CLIP model is provided
        assert model is not None, "Please pass a model name"

        # Use CLIP-specific preprocessing
        clip_transform = transforms.Compose([
            transforms.Resize(size=224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(size=(224, 224)),
            lambda img: img.convert('RGB'),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711))
        ])

        # Preprocess and stack all images
        preprocessed_imgs = [clip_transform(img) for img in imgs]
        batch = torch.stack(preprocessed_imgs).to(device)

        # Run through CLIP vision model
        with torch.no_grad():
            inputs = dict(pixel_values=batch)
            vision_outputs = clip_model.vision_model(**inputs)
            image_embeds = vision_outputs[1]
            image_embeds = clip_model.visual_projection(image_embeds)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)


        return image_embeds.cpu()
        




# # Build graph edges based on threshold
# threshold = 0.7
# sources, targets, weights = [], [], []

# num_images = image_emb.size(0)
# for i, j in itertools.product(range(num_images), repeat=2):
#     if i != j and cos_sim[i, j] >= threshold:
#         sources.append(i)
#         targets.append(j)
#         weights.append(cos_sim[i, j].item())

# edge_index = torch.tensor([sources, targets], dtype=torch.long)
# edge_weight = torch.tensor(weights, dtype=torch.float)

# # PyG Data object
# graph_data = Data(
#     x=image_emb,  # [N, 4096]
#     edge_index=edge_index,  # [2, num_edges]
#     edge_attr=edge_weight   # [num_edges]
# )

# print(graph_data)

# %%
# img1='data:image/jpeg;base64,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'

# img2='data:image/jpeg;base64,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'

# img3 = encode_image("https://static01.nyt.com/images/2016/10/18/us/18fd-trumpfoundation/18fd-trumpfoundation-master675.jpg")

# image_paths = [img1,img2,img3]
# embeddings=get_embeddings(image_paths)
# # %%
# cosine_sim_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
# # %%
# cosine_sim_matrix
# %%
