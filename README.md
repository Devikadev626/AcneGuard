# AcneGuard - AI Acne Severity Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-ff4b4b.svg)](https://streamlit.io/)

**AcneGuard** is an AI-powered Acne Severity Detection System that automatically classifies facial acne images into different severity levels using Deep Learning and Computer Vision. 

The system analyzes acne images and predicts the corresponding severity grade alongside a confidence score. It features an end-to-end ML pipeline encompassing automated training, evaluation metrics, a scalable FastAPI REST API backend, and an easy-to-use Streamlit web interface.

---

## 📌 Problem Statement

Dermatologists and skin care professionals often assess acne severity manually, which is highly subjective, time-consuming, and prone to intra-observer variability. 

The goal of **AcneGuard** is to provide an objective, automated AI system capable of:
* Extracting diagnostic features from skin and facial images.
* Classifying acne into predefined clinical severity grades.
* Outputting precise confidence scores for medical triage visibility.
* Exposing microservice APIs for clinical system integration.
* Presenting an intuitive frontend interface tailored for non-technical users.

---

## 📊 Dataset Information

* **Dataset Type:** Acne Severity Classification Dataset
* **Total Samples:** 1457 Images
* **Data Root Directory:** `data/raw/severity_dataset/`

### Classes & Distribution

| Grade | Clinical Description | Image Count |
| :--- | :--- | :--- |
| **Grade 0** | Clear / Very Mild Acne | 410 |
| **Grade 1** | Mild Acne | 506 |
| **Grade 2** | Moderate Acne | 146 |
| **Grade 3** | Severe Acne | 103 |

### Dataset Storage Structure

```text
data/raw/severity_dataset/
├── JPEGImages/               # Source image repository (.jpg/.jpeg)
├── NNEW_trainval_0.txt        # Cross-validation splits per grade index
├── NNEW_test_0.txt
├── NNEW_trainval_1.txt
├── NNEW_test_1.txt
├── NNEW_trainval_2.txt
├── NNEW_test_2.txt
├── NNEW_trainval_3.txt
├── NNEW_test_3.txt
├── NNEW_trainval_4.txt
└── NNEW_test_4.txt