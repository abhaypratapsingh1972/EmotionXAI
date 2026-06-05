# EmotionXAI

# 😊 EmotionXAI

### Explainable Facial Emotion Recognition using EfficientNet-B0 and Grad-CAM

EmotionXAI is an end-to-end Explainable AI (XAI) application for facial emotion recognition. The system predicts human emotions from facial images and provides visual explanations using Grad-CAM to highlight the facial regions that influenced the prediction.

🌐 **Live Demo:**[https://emotionxai-99hzggsyvkze63lthle5hv.streamlit.app/](https://emotionxai-99hzggsyvkze63lthle5hv.streamlit.app/)

---

## 🚀 Features

* Facial Emotion Recognition from images
* EfficientNet-B0 Transfer Learning
* Explainable AI using Grad-CAM
* Top-3 Emotion Predictions
* Confidence Scores
* Probability Distribution Visualization
* Downloadable Prediction Reports
* Interactive Streamlit Web Application
* Public Cloud Deployment

---

## 🎯 Supported Emotions


| Class    |
| -------- |
| Angry    |
| Disgust  |
| Fear     |
| Happy    |
| Neutral  |
| Sad      |
| Surprise |

---

## 🧠 Model Architecture

The model uses **EfficientNet-B0** as the backbone network and fine-tunes the classification head for 7-class facial emotion recognition.

Input Image (224×224 RGB)
↓
EfficientNet-B0 Backbone
↓
Global Feature Extraction
↓
Fully Connected Layer
↓
7 Emotion Classes

---

## 📊 Model Performance

### FER2013 Test Results


| Metric            | Score |
| ----------------- | ----- |
| Accuracy          | 69.5% |
| Macro Precision   | 0.69  |
| Macro Recall      | 0.68  |
| Macro F1 Score    | 0.69  |
| Weighted F1 Score | 0.69  |

### Per-Class Performance


| Emotion  | Precision | Recall | F1 Score |
| -------- | --------- | ------ | -------- |
| Angry    | 0.60      | 0.61   | 0.61     |
| Disgust  | 0.76      | 0.68   | 0.72     |
| Fear     | 0.58      | 0.55   | 0.56     |
| Happy    | 0.87      | 0.87   | 0.87     |
| Neutral  | 0.64      | 0.64   | 0.64     |
| Sad      | 0.57      | 0.58   | 0.57     |
| Surprise | 0.82      | 0.84   | 0.83     |

---

## 🔍 Explainable AI with Grad-CAM

EmotionXAI integrates Grad-CAM to provide transparency and interpretability.

The heatmap highlights facial regions that contribute most strongly to the model's prediction.

Examples:

* Happy → Mouth and cheek regions
* Angry → Eyebrows and eyes
* Sad → Eyes and mouth
* Surprise → Eyes and mouth
* Fear → Eyes and facial tension regions

This allows users to understand why the model predicted a particular emotion.

---

## 🖥️ Application Workflow

Upload Image
↓
Face Emotion Prediction
↓
Confidence Score
↓
Top-3 Predictions
↓
Grad-CAM Heatmap
↓
Natural Language Explanation
↓
Download Prediction Report

---

## 🛠️ Tech Stack

### Machine Learning

* PyTorch
* TorchVision
* TIMM
* EfficientNet-B0

### Explainable AI

* Grad-CAM

### Deployment

* Streamlit
* Streamlit Cloud

### Data Processing

* NumPy
* Pandas
* Pillow
* OpenCV

---

## 📂 Project Structure

```text
EmotionXAI/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── best_b0_finetuned.pth
│
└── utils/
    ├── explain.py
    ├── gradcam_utils.py
    └── predict.py
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/abhaypratapsingh1972/EmotionXAI.git
cd EmotionXAI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📸 Demo

The application provides:

* Emotion Prediction
* Confidence Score
* Top-3 Predictions
* Probability Distribution
* Grad-CAM Visualization
* Downloadable Report

---

## 🔮 Future Improvements

* Multi-Face Emotion Detection
* Real-Time Webcam Emotion Recognition
* ConvNeXt and Vision Transformer Models
* FER+ Dataset Support
* Emotion Tracking over Time
* Mobile Deployment

---

## 👨‍💻 Author

**Abhay Pratap Singh**

* GitHub: [https://github.com/abhaypratapsingh1972](https://github.com/abhaypratapsingh1972)
* LinkedIn: [https://linkedin.com/in/abhay-pratap-singh-400860290](https://linkedin.com/in/abhay-pratap-singh-400860290)

---

## ⭐ Acknowledgements

* FER2013 Dataset
* PyTorch
* TIMM
* Streamlit
* Grad-CAM

If you found this project useful, please consider giving it a ⭐ on GitHub.

Explainable Facial Emotion Recognition using EfficientNet-B0 and Grad-CAM.

## Features

* Facial Emotion Recognition
* EfficientNet-B0 Transfer Learning
* Grad-CAM Explainability
* Top-3 Emotion Predictions
* Confidence Scores
* Interactive Streamlit Interface
* Downloadable Prediction Reports

## Dataset

FER2013 Dataset

Classes:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

## Model Performance


| Metric      | Score |
| ----------- | ----- |
| Accuracy    | 69.5% |
| Macro F1    | 0.69  |
| Weighted F1 | 0.69  |

## Explainable AI

EmotionXAI integrates Grad-CAM visualizations to highlight facial regions influencing model predictions.

## Tech Stack

* Python
* PyTorch
* EfficientNet-B0
* Grad-CAM
* Streamlit
* OpenCV

## Demo

Upload a facial image and receive:

* Predicted Emotion
* Confidence Score
* Top-3 Predictions
* Attention Heatmap
* Natural Language Explanation

## Future Improvements

* Multi-face Detection
* Video Emotion Analysis
* Vision Transformer Integration
* FER+ Dataset Support
