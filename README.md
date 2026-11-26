# Brain Tumor Classification using Deep Learning

## Project Overview

This project focuses on classifying brain MRI images into four categories using Deep Learning and Transfer Learning.

A custom CNN and a fine-tuned ResNet50 model were developed and evaluated. The final model achieved **92.97% test accuracy**.

## Dataset

**Dataset:** Brain Tumors MRI – 256×256

The dataset contains four classes:

| Class | Description |
|---|---|
| Normal | No brain tumor |
| Glioma | Glioma tumor |
| Meningioma | Meningioma tumor |
| Pituitary | Pituitary tumor |

After removing exact duplicate images, the dataset contained **2,845 images**.

### Dataset Split

| Split | Images |
|---|---:|
| Training | 1,991 |
| Validation | 427 |
| Testing | 427 |
| **Total** | **2,845** |

## Exploratory Data Analysis & Preprocessing

The project included:

- Dataset inspection and class distribution analysis.
- Exact duplicate image removal.
- Image resizing to 256×256.
- Training, validation, and test splitting.
- Data augmentation applied to training images only.
- Image normalization and preprocessing.

## Models

Two main approaches were explored:

### 1. Custom CNN
A CNN architecture was developed and trained from scratch to establish a deep learning classification model.

### 2. ResNet50 – Transfer Learning
A pretrained ResNet50 model was fine-tuned for the four-class brain tumor classification task.

**ResNet50 Fine-Tuned** achieved the best performance and was selected as the final model.

## Model Performance

### Test Set Results

<ins>Best Model: ResNet50 Fine-Tuned</ins>

| Metric | Score |
|---|---:|
| Test Accuracy | **92.97%** |
| Macro F1-Score | **93.26%** |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Normal | 0.90 | 0.98 | 0.95 |
| Glioma | 0.95 | 0.89 | 0.92 |
| Meningioma | 0.87 | 0.92 | 0.90 |
| Pituitary | 0.96 | 0.97 | 0.97 |

## Model Deployment

The trained model is integrated into:

- **FastAPI:** REST API for brain MRI image predictions.
- **Streamlit:** Interactive interface for uploading MRI images and viewing predictions.

## Tech Stack

- Python
- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- FastAPI
- Streamlit
- MLflow

## Project Structure

```text
Brain-Tumor-Classification/
│
├── app/
│   └── streamlit_app.py
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_resnet50_training.ipynb
│
├── src/
│   ├── api.py
│   ├── data.py
│   ├── model.py
│   ├── predict.py
│   ├── resnet_data.py
│   └── resnet_model.py
│
├── requirements.txt
└── .gitignore
```

## Disclaimer

This project is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis or clinical evaluation.
