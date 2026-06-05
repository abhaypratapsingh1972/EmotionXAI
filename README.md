# EmotionXAI

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
