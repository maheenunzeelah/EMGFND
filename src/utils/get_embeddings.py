# %%
import config
import torch
from torchvision import models, transforms
from transformers import CLIPModel
from PIL import Image

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

def get_embeddings(imgs, model = config.image_embed_model):
    
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
