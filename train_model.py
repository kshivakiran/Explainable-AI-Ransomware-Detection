"""
train_model.py - Train all ML models. Run ONCE before starting app.py
Uses: IsolationForest (anomaly) + RandomForest (classifier) + LIME (explainability)
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

print("[1/5] Loading dataset...")
df = pd.read_csv('CIC_Ransomware_2019_Full_Dataset.csv')

FEATURES = [
    'cpu_usage', 'cpu_variance', 'num_processes', 'process_creation_rate',
    'memory_usage', 'page_faults', 'virtual_memory',
    'file_reads', 'file_writes', 'file_deletes', 'file_renames',
    'file_encryption_rate', 'unique_extensions', 'file_entropy',
    'net_bytes_sent', 'net_bytes_recv', 'net_connections', 'dns_queries', 'external_ips',
    'registry_mods', 'service_changes', 'privilege_escalations', 'shadow_copy_deletes',
    'io_bytes', 'thread_count', 'handle_count',
    'api_crypto', 'api_filesystem', 'api_network', 'api_registry'
]

X = df[FEATURES].fillna(0).values
y = (df['label'] == 'Ransomware').astype(int).values
print(f"     Loaded {len(y)} samples | Benign: {sum(y==0)} | Ransomware: {sum(y==1)}")

print("[2/5] Splitting and scaling features...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("[3/5] Training IsolationForest on benign data only (anomaly detector)...")
X_train_benign = X_train_scaled[y_train == 0]
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
iso.fit(X_train_benign)

train_anomaly = (-iso.score_samples(X_train_scaled)).reshape(-1, 1)
test_anomaly  = (-iso.score_samples(X_test_scaled)).reshape(-1, 1)
print(f"     Benign avg anomaly score:     {train_anomaly[y_train==0].mean():.4f} (LOW)")
print(f"     Ransomware avg anomaly score: {train_anomaly[y_train==1].mean():.4f} (HIGH)")

X_train_aug = np.hstack((X_train_scaled, train_anomaly))
X_test_aug  = np.hstack((X_test_scaled,  test_anomaly))

print("[4/5] Training RandomForest classifier (200 trees)...")
clf = RandomForestClassifier(
    n_estimators=200, max_depth=20, min_samples_leaf=2,
    random_state=42, n_jobs=-1, class_weight='balanced'
)
clf.fit(X_train_aug, y_train)

y_pred      = clf.predict(X_test_aug)
y_pred_prob = clf.predict_proba(X_test_aug)[:, 1]

print("\n--- EVALUATION RESULTS ---")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Ransomware']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

print("[5/5] Saving all model files...")
np.random.seed(42)
bg_idx = np.random.choice(len(X_train_aug), min(200, len(X_train_aug)), replace=False)
np.save('lime_background.npy', X_train_aug[bg_idx])

with open('model.pkl',      'wb') as f: pickle.dump(clf,     f)
with open('iso_forest.pkl', 'wb') as f: pickle.dump(iso,     f)
with open('scaler.pkl',     'wb') as f: pickle.dump(scaler,  f)
with open('features.pkl',   'wb') as f: pickle.dump(FEATURES, f)

pd.DataFrame([X_test[y_test==1][0]], columns=FEATURES).to_csv('test_ransomware.csv', index=False)
pd.DataFrame([X_test[y_test==0][0]], columns=FEATURES).to_csv('test_benign.csv',     index=False)

print("\n[DONE] All files saved successfully!")
print("  model.pkl, iso_forest.pkl, scaler.pkl, features.pkl")
print("  lime_background.npy")
print("  test_ransomware.csv  <-- upload this to see RANSOMWARE DETECTED")
print("  test_benign.csv      <-- upload this to see SYSTEM BENIGN")
print("\nNow run: python app.py")
