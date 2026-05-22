# Drone Image Quality Assessment & Self-Learning Pipeline

## Overview

Drone-based photogrammetry and 3D reconstruction pipelines require **high-quality images**. Poor-quality images such as blurred, overexposed, noisy, or textureless images can significantly degrade the quality of generated **3D models and maps**.

This project develops an **intelligent multi-stage pipeline** that automatically evaluates drone images and classifies them into three categories:

- **GOOD** → Suitable for direct use in 3D reconstruction  
- **RECOVERABLE** → Can potentially be enhanced and reused  
- **BAD** → Should be discarded  

Instead of performing only image classification, the system implements a **complete adaptive workflow** that:

1. Extracts image quality metrics
2. Trains a CNN classifier (EfficientNet-B0)
3. Enhances recoverable images
4. Re-evaluates image quality
5. Logs results for monitoring
6. Supports a **self-learning mechanism**

The objective is to improve the quality of input images before they are used in **3D mapping systems such as COLMAP or OpenSfM**.

---

# Key Contributions

• Automated drone image quality classification  
• Feature-driven dataset labeling approach  
• Multi-stage enhancement and reclassification pipeline  
• EfficientNet-B0 based deep learning model  
• Self-learning mechanism for dataset expansion  

---

# Dataset

Due to GitHub storage limitations, datasets are hosted on **Google Drive**.

### Raw Drone Image Dataset

Initial mixed dataset used for labeling and feature extraction:

Dataset Folder 1  
https://drive.google.com/drive/folders/1WxPWBNJL9MHO-BT16NifqncJGqisxGRE?usp=drive_link

Dataset Folder 2  
https://drive.google.com/drive/folders/1FO_usOJZJzLATcPgPTbnAxgQRZmT6r6j?usp=drive_link

These datasets contain raw drone images that were manually inspected and categorized during the early stages of the project.

---

# Feature Extraction

Before training the deep learning model, several **image quality metrics** were extracted from the dataset.

These metrics quantify image quality properties such as blur, brightness, texture richness, and atmospheric noise.

### Extracted Metrics

- Laplacian Variance (Blur Detection)
- ORB Keypoints (Texture Richness)
- Brightness
- Saturation
- Haze Estimation

Feature extraction results were stored in CSV format.

### Feature Datasets

Good Images Feature Dataset  
https://drive.google.com/file/d/1Sti3eoVucljoF2zp3rV7jf4uOI4tPySq/view?usp=sharing

Bad Images Feature Dataset  
https://drive.google.com/file/d/1WcS9483Trvn5NIMq9utH3RMRbbUZyWPm/view?usp=sharing

Using these extracted features, the images were labeled into:

- GOOD
- BAD
- RECOVERABLE

The labeled images were then placed into their respective folders.

---

# Data Augmentation

To increase dataset size and improve model robustness, several augmentation techniques were applied.

### Augmentation Techniques

- Horizontal flipping
- Brightness variation
- Contrast variation
- Minor rotations

### Augmented Datasets

Augmented BAD Images  
https://drive.google.com/drive/folders/12YavNN2qXg_eIDFUCdLprkzWKApGThTb?usp=sharing

Augmented GOOD Images  
https://drive.google.com/drive/folders/1jRV76EtcYttc3d-ozkxyWH3B8f2S3R2o?usp=sharing

This augmentation process helped balance the dataset and improved CNN training performance.

---

# Training Dataset Split

After augmentation, the dataset was divided into three sets:

- **Training Set**
- **Validation Set**
- **Testing Set**

### Dataset Structure


dataset
├── train
│ ├── GOOD
│ ├── BAD
│ └── RECOVERABLE
│
├── val
│ ├── GOOD
│ ├── BAD
│ └── RECOVERABLE
│
└── test
├── GOOD
├── BAD
└── RECOVERABLE



This structured dataset was used to train and evaluate the classification model.

---

# Model Training

The classification model used in this project is **EfficientNet-B0**.

EfficientNet was selected because it provides:

- High accuracy
- Efficient parameter usage
- Strong generalization performance for image classification tasks

### Training Procedure

1. Load EfficientNet-B0 pretrained weights
2. Replace the final classification layer
3. Train the model on the augmented dataset
4. Validate using the validation dataset
5. Evaluate final performance on the test dataset

---

# Main Project Pipeline

The core contribution of this project is a **multi-stage image processing pipeline** that evaluates drone images before they enter the photogrammetry workflow.

<img width="520" height="538" alt="Pipeline Architecture" src="https://github.com/user-attachments/assets/8b233cf7-2de8-4b9c-98cc-a647ca67eb02">

---

# Image Enhancement Stage

If an image is classified as **RECOVERABLE**, the system attempts to improve it using image enhancement techniques such as:

- Contrast enhancement
- Sharpening
- Noise reduction

The enhanced image is then **re-classified by the CNN model**.

If the enhanced image is predicted as **GOOD**, it is accepted into the usable dataset.

---

# Self-Learning Mechanism

The pipeline supports **continual learning**.

If the classifier predicts images with **high confidence**, those images can automatically be added to the training dataset.

This allows the system to gradually improve as more drone images are processed.

<img width="357" height="586" alt="Self Learning Pipeline" src="https://github.com/user-attachments/assets/b15997a3-2d15-431c-b35e-526697e3fccc">

---

# Running the Pipeline

### Step 1 — Clone the repository

This structured dataset was used to train and evaluate the classification model.

---

# Model Training

The classification model used in this project is **EfficientNet-B0**.

EfficientNet was selected because it provides:

- High accuracy
- Efficient parameter usage
- Strong generalization performance for image classification tasks

### Training Procedure

1. Load EfficientNet-B0 pretrained weights
2. Replace the final classification layer
3. Train the model on the augmented dataset
4. Validate using the validation dataset
5. Evaluate final performance on the test dataset

---

# Main Project Pipeline

The core contribution of this project is a **multi-stage image processing pipeline** that evaluates drone images before they enter the photogrammetry workflow.

<img width="520" height="538" alt="Pipeline Architecture" src="https://github.com/user-attachments/assets/8b233cf7-2de8-4b9c-98cc-a647ca67eb02">

---

# Image Enhancement Stage

If an image is classified as **RECOVERABLE**, the system attempts to improve it using image enhancement techniques such as:

- Contrast enhancement
- Sharpening
- Noise reduction

The enhanced image is then **re-classified by the CNN model**.

If the enhanced image is predicted as **GOOD**, it is accepted into the usable dataset.

---

# Self-Learning Mechanism

The pipeline supports **continual learning**.

If the classifier predicts images with **high confidence**, those images can automatically be added to the training dataset.

This allows the system to gradually improve as more drone images are processed.

<img width="357" height="586" alt="Self Learning Pipeline" src="https://github.com/user-attachments/assets/b15997a3-2d15-431c-b35e-526697e3fccc">

---

# Running the Pipeline

### Step 1 — Clone the repository
git clone https://github.com/Pratheeeksha/Image-quality-pipeline.git


### Step 2 — Install dependencies
pip install torch torchvision opencv-python pandas scikit-learn pillow

### Step 3 — Add new drone images

Place drone images inside:
data/incoming_drone_images

### Step 4 — Run the pipeline
python pipeline/run_full_pipeline.py

The system will:

1. Extract image quality metrics
2. Classify images
3. Enhance recoverable images
4. Re-classify enhanced images
5. Save results to a CSV file

---

# Example Output
Processing: image1.jpg
Processing: image2.jpg
Processing: image3.jpg

Pipeline finished.
Results saved to data/processed/pipeline_results.csv


---

# Technologies Used

- Python
- PyTorch
- EfficientNet-B0
- OpenCV
- Pandas
- Scikit-learn

---

# Future Improvements

Possible future improvements include:

- GAN-based image restoration models
- Integration with COLMAP for reconstruction quality evaluation
- Automated threshold tuning
- Improved continual learning strategies

---

# Author

Drone Image Quality Assessment & Self-Learning Pipeline

Developed for improving **drone photogrammetry input quality and automated dataset curation**.

