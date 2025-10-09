# External Multimodal Graph Based Fake News Detection

A comprehensive pipeline for fake news detection using multimodal graph neural networks with entity-aware image and text embeddings.

---

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Pipelines](#pipelines)
  - [Text Summarization](#text-summarization)
  - [Entity & Image Extraction](#entity--image-extraction)
  - [Embedding Processing](#embedding-processing)
- [Model Training & Evaluation](#model-training--evaluation)
- [Project Structure](#project-structure)

---

## 🚀 Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
TAG_ME_TOKEN=your_tagme_api_token_here
```

> **⚠️ Important:** 
> - Never commit your `.env` file to version control
> - Add `.env` to your `.gitignore`
> - The TAG_ME_TOKEN is required for entity extraction

### Configuration File (`config.py`)

Centralizes all project settings including:

- **Dataset Paths:** Processed data, entities, and reference images
- **Embedding Paths:** CLIP, BERT, ResNet, and MediaEval embeddings
- **Model Settings:** Text/image embedding models and dimensions
- **Image Directories:** Reference image storage locations

**Usage:**
1. Update paths in `config.py` to match your project structure
2. Ensure all directories exist before running pipelines
3. Switch datasets by modifying the `dataset` variable

---

## 🔄 Pipelines

### Text Summarization

**Script:** `src/pipelines/text_summarization.py`

Performs extractive summarization using BERT embeddings and SpaCy.

**Features:**
- Sentence splitting with SpaCy
- BERT-based sentence embeddings
- Cosine similarity ranking
- 512-token limit summaries

**Usage:**
```bash
python src/pipelines/text_summarization.py
```

**Output:** Adds `summarized_text` column to your DataFrame

---

### Entity & Image Extraction

#### Text Entity Extraction

**Script:** `src/pipelines/text_reference_images.py`

Extracts entities from article text using TagMe API and fetches Wikipedia images.

**Features:**
- Wikipedia entity annotation
- Async batch processing
- Image retrieval and caching
- Metadata management (JSON)

**Usage:**
```bash
python src/pipelines/text_reference_images.py
```

**Output:**
- CSV file with entities (`all_data_text_entities_df`)
- Reference images in `all_data_reference_images_dir`
- Metadata in `text_entity_metadata.json`

#### Title Entity Extraction

**Script:** `src/pipelines/title_reference_images.py`

Extracts entities from article titles.

**Features:**
- Title-based entity annotation
- Wikipedia image retrieval
- Efficient async processing

**Usage:**
```bash
python src/pipelines/title_reference_images.py
```

**Output:**
- CSV file with title entities (`all_data_title_entities_df`)
- Images in `all_data_reference_images_dir`
- Metadata in `entity_metadata.json`

---

### Embedding Processing

**Script:** `src/pipelines/processing_embeddings.py`

Processes image and text embeddings for ML models.

**Features:**
- Supports image and text embeddings
- Batch processing for efficiency
- Saves embeddings as pickle files

**Usage:**
```bash
python src/pipelines/processing_embeddings.py
```

**Configuration:**
- Set `embedding_type` to `"image"` or `"text"`
- Configure paths in `config.py`

---

### Embedding Extraction Utility

**Script:** `src/utils/get_embeddings.py`

Generates embeddings using pre-trained models.

**Supported Models:**
- **CLIP** (default)
- **ResNet-50** (2048-dim)
- **VGG16** (4096-dim)

**Usage:**
```python
from src.utils.get_embeddings import get_embeddings
from PIL import Image

images = [Image.open("path/to/image.jpg")]

# CLIP embeddings
embeddings = get_embeddings(images)

# ResNet embeddings
embeddings_resnet = get_embeddings(images, model="resnet")

# VGG embeddings
embeddings_vgg = get_embeddings(images, model="vgg")
```

**Adding New Models:**
1. Load pre-trained model in PyTorch
2. Define custom embedding extractor class
3. Add to `get_embeddings` function

---

## 🤖 Model Training & Evaluation

### EMGFND Module (`src/emgfnd`)

#### Dataset Setup (`src/emgfnd/utils.py`)

**Functions:**

**`set_up_all_data_dataset()`**
- Prepares multimodal dataset for training
- Filters invalid data
- Splits: 70% train, 20% val, 10% test
- Returns `MultimodalGraphDataset` objects

**`set_up_media_eval_dataset()`**
- MediaEval dataset preparation
- Splits: 70% train, 15% val, 15% test

**Usage:**
```python
from emgfnd.utils import set_up_all_data_dataset

train_dataset, val_dataset, test_dataset = set_up_all_data_dataset()
```

---

#### Model Configuration (`src/emgfnd/model_config.py`)

Contains the `Config` class with all hyperparameters:
- Batch size
- Learning rate
- Model dimensions
- Dataset paths

---

#### PGAT Model (`src/emgfnd/pgat_model.py`)

**PGATClassifier** - Position-aware Graph Attention Network
- Graph-based neural network (PyTorch Geometric)
- Forward propagation methods
- Classification for fake news detection

---

#### Training Pipeline (`src/emgfnd/modal_training.py`)

**Features:**
- ✅ Handles class imbalance (weighted loss, focal loss, weighted sampling)
- ✅ Early stopping
- ✅ Optimal threshold selection
- ✅ WandB logging (metrics, ROC/PR curves)
- ✅ Reproducible training (fixed seeds)

**Key Components:**
- `AUCMetrics`: ROC AUC and PR AUC computation
- `calculate_class_weights`: Dynamic weight calculation
- `find_optimal_threshold`: F1-maximizing threshold
- `train_func_epoch`: Training loop

**Usage:**
```bash
python src/emgfnd/modal_training.py
```

**Configuration:**
- Set `HANDLE_IMBALANCE = True` for imbalanced datasets
- Set `dataset_name` to `"all_data"` or `"media_eval"`

**Output:**
- Best model checkpoint saved to `config.best_model_path`
- Training logs in WandB

---

#### Model Inference (`src/emgfnd/model_inference.py`)

Evaluates trained model on test data.

**Features:**
- Supports both datasets (`all_data`, `media_eval`)
- Reproducible evaluation
- Comprehensive metrics reporting
- Custom threshold support

**Usage:**
```bash
python src/emgfnd/model_inference.py
```

**Metrics Reported:**
- Test Loss & Accuracy
- Precision, Recall, F1-score (per class)
- ROC AUC & PR AUC

**Dataset Options:**
```python
# Option 1: Setup function
train_dataset, val_dataset, test_dataset = set_up_all_data_dataset()

# Option 2: Load saved dataset
dataset_test = torch.load("path_to_saved_test_dataset.pt")
```

---

#### Evaluation Utilities (`src/emgfnd/evaluation_utils.py`)

Helper functions for model evaluation:
- Accuracy, precision, recall, F1-score
- AUC metrics
- Performance report generation

---

## 📁 Project Structure

```
.
├── src/
│   ├── pipelines/
│   │   ├── text_summarization.py
│   │   ├── text_reference_images.py
│   │   ├── title_reference_images.py
│   │   └── processing_embeddings.py
│   ├── emgfnd/
│   │   ├── utils.py
│   │   ├── model_config.py
│   │   ├── pgat_model.py
│   │   ├── modal_training.py
│   │   ├── model_inference.py
│   │   └── evaluation_utils.py
│   └── utils/
│       ├── get_embeddings.py
│       ├── pipeline_utils.py
│       └── tagMe.py
├── config.py
├── requirements.txt
├── .env
└── README.md
```

---

## 📦 Dependencies

- pandas
- transformers
- torch
- torchmetrics
- torch-geometric
- spacy
- scikit-learn
- aiohttp
- tqdm
- python-dotenv
- Pillow
- nest_asyncio
- wandb

---

## 📝 Notes

- **Reproducibility:** Random seeds are set across all scripts
- **Class Imbalance:** Training pipeline handles imbalanced datasets automatically
- **WandB Logging:** Requires WandB account for training visualization
- **Embeddings:** Use title-only or title+text embeddings by changing column references

---

## 🤝 Contributing

Ensure all paths in `config.py` match your environment before running any pipeline.
