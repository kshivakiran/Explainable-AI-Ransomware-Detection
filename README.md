# RansomShield AI: Explainable Ransomware Detection Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art cybersecurity project using **Explainable AI (XAI)** to detect ransomware through behavioral analysis. This framework combines generative-style anomaly detection with discriminative classification to provide highly accurate and transparent security auditing.
LIVE DEMO :https://explainable-ai-ransomware-detection-1.onrender.com

## 🚀 Key Features

- **Dual-Model Architecture**: Uses `IsolationForest` for unsupervised anomaly detection and `RandomForest` for supervised classification.
- **LIME Explainability**: Provides human-readable explanations for every detection, showing exactly which system behaviors (CPU usage, encryption rate, etc.) triggered the alert.
- **Modern Dashboard**: A premium, responsive web interface built with Vanilla CSS and Chart.js for real-time visualization.
- **Production Ready**: Optimized for deployment on platforms like Render or Heroku with minimal resource footprint.

## 🧠 How It Works

1. **Anomaly Detection**: An `IsolationForest` model learns the "manifold" of normal (benign) system behavior. Ransomware activity typically deviates significantly, producing a high anomaly score.
2. **Classification**: A `RandomForestClassifier` processes 30+ behavioral features plus the anomaly score to make a final diagnosis.
3. **Interpretability**: The LIME algorithm generates local explanations, highlighting feature contributions toward the "Ransomware" or "Benign" label.

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Gunicorn
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Explainability**: LIME (Local Interpretable Model-agnostic Explanations)
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript (ES6+), Chart.js
- **Dataset**: CIC-Ransomware-2019

## 📋 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ransomshield-ai.git
   cd ransomshield-ai
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models**:
   ```bash
   python train_model.py
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```
   The app will be available at `http://localhost:5000`.

## 📂 Project Structure

- `app.py`: Flask backend with prediction logic.
- `train_model.py`: ML pipeline for training and saving models.
- `mycode.html`: The interactive frontend dashboard.
- `model.pkl`: Trained classifier.
- `iso_forest.pkl`: Trained anomaly detector.
- `test_*.csv`: Sample files for testing the demo.

## 📊 Evaluation Results

The model achieves near-perfect performance on the CIC-Ransomware-2019 dataset:
- **Accuracy**: 100%
- **Precision/Recall**: 1.00
- **ROC-AUC**: 1.00

---

Built for academic excellence and professional portfolio development.
