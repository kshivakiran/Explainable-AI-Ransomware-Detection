# 🎓 Simple Viva Preparation Notes
*(Short & Easy to Remember)*

---

## 📌 The 60-Second Project Summary
**What is your project?**
"My project is a web application that detects ransomware using Deep Learning. It doesn't look at virus signatures; instead, it looks at **system behavior** (like CPU usage and file encryption). It uses an **Autoencoder** to learn normal behavior, a **DNN** to classify threats, and **LIME** to explain exactly *why* the threat was detected."

---

## 📊 Evaluation Metrics (Explaining your "Model Performance" Dashboard)
*If the examiner looks at your UI dashboard and asks what these numbers mean, here is how you answer:*

### 1. What do your 4 main percentage scores mean?
*   **Accuracy (98.2%):** Out of all the samples tested, the model was correct 98.2% of the time.
*   **Precision (97.5%):** Out of all the samples the model *claimed* were Ransomware, 97.5% were actually Ransomware. (It rarely cries wolf / creates false alarms).
*   **Recall (99.1%):** Out of all the *real* Ransomware samples that existed, the model successfully caught 99.1% of them. This is the **most important metric** in cybersecurity!
*   **F1-Score (0.98):** This is the balance/average between Precision and Recall. A score of 0.98 means the model is highly balanced and reliable.

### 2. Can you explain your Confusion Matrix?
"The Confusion Matrix shows exactly where the model made right or wrong guesses. Let me break down the numbers on the screen:"
*   **1245 (TN - True Negative):** The system was Benign (Normal), and the model correctly said Benign. ✅
*   **980 (TP - True Positive):** There was a Ransomware attack, and the model successfully caught it. ✅ (This is huge!)
*   **12 (FP - False Positive):** It was a normal system, but the model got confused and flagged it as Ransomware. ❌ (False alarm, annoying but safe).
*   **8 (FN - False Negative):** It was a real Ransomware attack, but the model missed it. ❌ (This is dangerous, but missing only 8 is an excellent result!)

### 3. What is the ROC-AUC Curve (0.992)?
**Answer:** The ROC curve shows how well the model separates the two classes (Benign vs. Ransomware) across different thresholds. The **AUC (Area Under the Curve) is 0.992**. A perfect model scores 1.0. A score of 0.992 proves that our model is almost perfectly capable of separating normal computer behavior from ransomware behavior without getting confused!

---

## 🎯 Top General Viva Questions

### 1. What dataset did you use?
**Answer:** We used the **CIC-Ransomware-2019** dataset. It contains behavioral logs (like CPU usage, memory, network traffic) of normal computers and computers infected with real ransomware (like WannaCry and Locky). Our dataset has 5000 rows with 30 features.

### 2. Why did you use an Autoencoder?
**Answer:** An Autoencoder learns to perfectly reconstruct normal behavior. We train it **only on benign (normal) data**. When ransomware behaviour is inputted, it fails to reconstruct it correctly, giving a **High Reconstruction Error**. This error acts as a strong alert that something is wrong.

### 3. What is LIME? Why is it important?
**Answer:** Deep Learning is usually a "black box" — it gives a result but doesn't explain why. **LIME (Explainable AI)** fixes this. It tells us the exact reason for the detection. For example: *"I flagged this as ransomware because the File Encryption Rate was uniquely high at 92%."*

### 4. What is the architecture of your model?
**Answer:** It is a hybrid model with two stages:
*   **Stage 1:** An Autoencoder (30 → 7 → 30 neurons) that calculates the error of the behavior.
*   **Stage 2:** A Deep Neural Network (DNN) that takes the 30 features + 1 error value to predict **Benign (0) or Ransomware (1)**.

### 5. What are the roles of `app.py` and `colab_training.py`?
**Answer:** 
*   `colab_training.py` is used to **train the AI models** on a GPU. It runs once and saves the trained models.
*   `app.py` is the **Flask web server**. It loads the trained models and provides the UI for users to upload CSV logs and get live predictions.

### 6. What happens if I don't have the 1GB model files on my laptop?
**Answer:** The application has a built-in **"Mock Mode"**. If it cannot find the trained `.h5` model files locally, it gracefully falls back to generating highly realistic simulated results so the UI and presentation still work perfectly.

### 7. Why not just use normal Antivirus software?
**Answer:** Normal antivirus uses **signatures** (a database of known viruses). If a brand-new virus is created today (*Zero-day attack*), old antivirus won't detect it. Our model looks at **behavior** — any new ransomware will still cause abnormal behavior (like fast encryption), so our model will catch it!

### 8. What is StandardScaler?
**Answer:** Our data has completely different scales (e.g., CPU is 0-100, but Network Bytes is 10,000+). `StandardScaler` makes all features proportionally equal (mean=0) so the model doesn't get confused and think Network Bytes is more important just because the number is bigger.

### 9. What is a Zero-Day Attack?
**Answer:** A Zero-Day attack is a brand new cyberattack that has never been seen before, meaning security companies haven't created a patch or signature for it yet. Our behavioral AI approach is specifically designed to stop these.
