# Skin Cancer Detection with 3D-TBP

Deep learning project for early detection of skin cancer using Total Body Photography (3D-TBP) images combined with patient metadata.

## My Contributions
- Preprocessed dermoscopic and 3D-TBP images (resize, normalization, augmentation).
- Trained CNN models (ResNet, EfficientNet) for skin lesion classification.
- Combined CNN logits with patient metadata (age, gender, lesion site) using Gradient Boosting (XGBoost).
- Designed ensemble models that improved accuracy over standalone CNNs.
- Documented results, accuracy metrics, and evaluation process.

## Tech Stack
- Python
- Libraries: TensorFlow/Keras, PyTorch, scikit-learn, XGBoost, OpenCV
- Jupyter Notebook for model training and evaluation

## Files
- test.py → testing script
- train_cnn.ipynb (if included) → CNN training and evaluation (ResNet, EfficientNet)
- preprocess.py (if included) → image preprocessing and augmentation
- metadata_boost.py (if included) → combining CNN features with metadata using XGBoost
- requirements.txt (if included) → required libraries

## Note
- Datasets (train.csv, test.csv) are *not included* due to size and privacy restrictions.  
- Trained model weights (best_efficientnetv2_model.keras) are *excluded* due to GitHub size limits.  
- This repo contains code and scripts to demonstrate the complete pipeline.
