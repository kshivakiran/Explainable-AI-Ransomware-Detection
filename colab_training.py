# =============================================
# COLAB TRAINING CODE - Ransomware Detection
# Copy each CELL into a separate Colab cell
# Run one by one (Shift + Enter)
# =============================================


# ──── CELL 1: Install & Import ────
# !pip install lime

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

print("✅ All libraries loaded!")


# ──── CELL 2: Load Dataset ────
# Upload the CIC_Ransomware_2019_Full_Dataset.csv file to Google Colab first!

print("Loading dataset...")
df = pd.read_csv('CIC_Ransomware_2019_Full_Dataset.csv')

# Feature names - these are system behaviors we monitor
features = [
    'cpu_usage', 'cpu_variance', 'num_processes', 'process_creation_rate',
    'memory_usage', 'page_faults', 'virtual_memory',
    'file_reads', 'file_writes', 'file_deletes', 'file_renames',
    'file_encryption_rate', 'unique_extensions', 'file_entropy',
    'net_bytes_sent', 'net_bytes_recv', 'net_connections', 'dns_queries', 'external_ips',
    'registry_mods', 'service_changes', 'privilege_escalations', 'shadow_copy_deletes',
    'io_bytes', 'thread_count', 'handle_count',
    'api_crypto', 'api_filesystem', 'api_network', 'api_registry'
]

X = df[features].values
y = (df['label'] == 'Ransomware').astype(int).values  # 0=Benign, 1=Ransomware

print(f"✅ Dataset loaded: {len(y)} samples, {len(features)} features")
print(f"   Benign: {sum(y==0)}, Ransomware: {sum(y==1)}")


# ──── CELL 3: Preprocess Data ────
# Split into train/test, scale features, separate benign-only data

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale all features to same range (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Benign-only data for Autoencoder (it must learn ONLY normal behavior)
X_train_benign = X_train_scaled[y_train == 0]

print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
print(f"   Benign-only for Autoencoder: {len(X_train_benign)}")


# ──── CELL 4: Train Autoencoder ────
# Autoencoder = learns to compress & reconstruct normal data
# When ransomware data comes in, it can't reconstruct it → HIGH error

input_dim = X_train_scaled.shape[1]  # 30

# Architecture: 30 → 20 → 14 → 7 → 14 → 20 → 30
inp = Input(shape=(input_dim,))
enc = Dense(20, activation='relu')(inp)
enc = BatchNormalization()(enc)
enc = Dense(14, activation='relu')(enc)
bottleneck = Dense(7, activation='relu')(enc)       # Compressed to 7
dec = Dense(14, activation='relu')(bottleneck)
dec = Dense(20, activation='relu')(dec)
out = Dense(input_dim, activation='linear')(dec)    # Reconstruct back to 30

autoencoder = Model(inp, out)
autoencoder.compile(optimizer='adam', loss='mse')

# Train on benign data ONLY — input = output (reconstruct itself)
autoencoder.fit(
    X_train_benign, X_train_benign,
    epochs=50, batch_size=32, validation_split=0.15,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)
print("✅ Autoencoder trained!")


# ──── CELL 5: Compute Reconstruction Error ────
# Low error = normal, High error = ransomware!

train_recon = autoencoder.predict(X_train_scaled, verbose=0)
train_error = np.mean(np.square(X_train_scaled - train_recon), axis=1)

test_recon = autoencoder.predict(X_test_scaled, verbose=0)
test_error = np.mean(np.square(X_test_scaled - test_recon), axis=1)

print(f"✅ Benign avg error:     {train_error[y_train==0].mean():.4f} (LOW)")
print(f"   Ransomware avg error: {train_error[y_train==1].mean():.4f} (HIGH)")

# Add error as new feature (30 features + 1 error = 31 features)
X_train_aug = np.hstack((X_train_scaled, train_error.reshape(-1, 1)))
X_test_aug = np.hstack((X_test_scaled, test_error.reshape(-1, 1)))


# ──── CELL 6: Train DNN Classifier ────
# Takes 31 features → predicts Benign(0) or Ransomware(1)

classifier = Sequential([
    Dense(64, activation='relu', input_dim=31),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output: 0-1 probability
])

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(
    X_train_aug, y_train,
    epochs=30, batch_size=32, validation_split=0.2,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)],
    verbose=1
)
print("✅ Classifier trained!")


# ──── CELL 7: Evaluate Model ────

y_pred_prob = classifier.predict(X_test_aug, verbose=0).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print("\n📊 RESULTS:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Ransomware']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")


# ──── CELL 8: LIME Explainability ────
# Shows WHY the AI made its decision

def predict_fn(x):
    p = classifier.predict(x, verbose=0)
    return np.hstack((1 - p, p))

explainer = LimeTabularExplainer(
    X_train_aug,
    feature_names=features + ['recon_error'],
    class_names=['Benign', 'Ransomware'],
    mode='classification'
)

# Explain one ransomware sample
sample = X_test_aug[y_test == 1][0]
exp = explainer.explain_instance(sample, predict_fn, num_features=8)

print("\n🔍 LIME Explanation (Why is this Ransomware?):")
for feat, weight in exp.as_list():
    print(f"   {feat:40s} → {weight:+.4f}")

exp.save_to_file('lime_explanation.html')
print("✅ Saved: lime_explanation.html")


# ──── CELL 9: Save Models ────

autoencoder.save("autoencoder_model.h5")
classifier.save("classifier_model.h5")
with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)
with open("features.pkl", "wb") as f: pickle.dump(features, f)

# Save test CSV files for the web app
pd.DataFrame([X_test[y_test==1][0]], columns=features).to_csv('test_ransomware.csv', index=False)
pd.DataFrame([X_test[y_test==0][0]], columns=features).to_csv('test_benign.csv', index=False)

print("✅ All files saved!")
print("""
📥 Download these and put in your app/ folder:
   - autoencoder_model.h5
   - classifier_model.h5
   - scaler.pkl
   - features.pkl
Then run: python app.py → Real AI mode activated!
""")



