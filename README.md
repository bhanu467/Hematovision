# Project Title: Hematovision – A Deep Learning-Based Blood Cell Classifier

## 🔍 Problem Statement
Early and accurate detection of blood cell types is critical for diagnosing various diseases like leukemia or anemia. Manual inspection under a microscope is time-consuming and error-prone. This project automates the classification of blood cells into 4 categories: Neutrophil, Eosinophil, Lymphocyte, and Monocyte, using a deep learning model.

## 🧠 Technologies Used
- Python
- TensorFlow / Keras
- NumPy, OpenCV
- Flask (for web app)
- Railway (deployment)
- GitHub (version control)

## 📁 Project Structure
- `model/` → Trained TensorFlow SavedModel
- `app.py` → Flask backend for image upload & prediction
- `templates/` → HTML templates for frontend
- `static/` → CSS, JS, and image assets
- `convert_model.py` → Script to export the trained model
- `requirements.txt` → Dependencies

## 📊 Dataset
Kaggle Blood Cell Dataset: [https://www.kaggle.com/datasets/paultimothymooney/blood-cells](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)

## ✅ Results
The MobileNetV2-based model achieved:
- ✅ **Train Accuracy**: 95.81%
- ✅ **Validation Accuracy**: 91.63%
- ✅ **Test Accuracy**: 93.24%

## 🎥 Demo Video
https://drive.google.com/file/d/1-BfA6s-T34cuVJ6nRSYgHYeesudv8YlN/view?usp=sharing

## 👩‍💻 Contributors
- G. Bhanu Mayukha

