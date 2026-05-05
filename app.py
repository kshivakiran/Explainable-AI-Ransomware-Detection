"""
app.py — Explainable AI Ransomware Detection Server
Real ML using scikit-learn (IsolationForest + RandomForest + LIME)
No TensorFlow required — works on Render free tier!
"""

import os
import time
import traceback
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Model Loading ─────────────────────────────────────────────────────────────
MODEL_STATUS   = "offline"
model          = None
iso_forest     = None
scaler         = None
feature_columns= None
lime_background= None

try:
    import pickle
    from lime.lime_tabular import LimeTabularExplainer

    required = ['model.pkl', 'iso_forest.pkl', 'scaler.pkl', 'features.pkl']
    if all(os.path.exists(f) for f in required):
        with open('model.pkl',      'rb') as f: model           = pickle.load(f)
        with open('iso_forest.pkl', 'rb') as f: iso_forest      = pickle.load(f)
        with open('scaler.pkl',     'rb') as f: scaler          = pickle.load(f)
        with open('features.pkl',   'rb') as f: feature_columns = pickle.load(f)

        if os.path.exists('lime_background.npy'):
            lime_background = np.load('lime_background.npy')

        MODEL_STATUS = "live"
        print("[OK] LIVE MODE - All models loaded successfully!")
    else:
        print("[WARN] OFFLINE - Model files not found. Run: python train_model.py")

except Exception as e:
    print(f"[ERROR] Failed to load models: {e}")
    traceback.print_exc()


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return send_from_directory('.', 'mycode.html')

@app.route('/status')
def status():
    return jsonify({
        'mode':     MODEL_STATUS,
        'features': len(feature_columns) if feature_columns else 0
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'mode': MODEL_STATUS})


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # ── Offline Guard ──────────────────────────────────────────────────────────
    if MODEL_STATUS != "live":
        return jsonify({
            'error': 'AI models not loaded. Run train_model.py first to generate model files.',
            'mode': 'offline'
        }), 503

    try:
        input_df = pd.read_csv(file)

        # Validate columns
        missing = [c for c in feature_columns if c not in input_df.columns]
        if missing:
            return jsonify({
                'error': f'Missing {len(missing)} required columns (e.g. {missing[:3]}). '
                         f'Please upload test_ransomware.csv or test_benign.csv.'
            }), 400

        # ── Feature Extraction ─────────────────────────────────────────────────
        raw      = input_df[feature_columns].fillna(0).iloc[0].values.reshape(1, -1)
        scaled   = scaler.transform(raw)

        # IsolationForest anomaly score (higher = more suspicious)
        anomaly  = float(-iso_forest.score_samples(scaled)[0])
        aug      = np.hstack((scaled, [[anomaly]]))   # 31 features

        # ── Prediction ─────────────────────────────────────────────────────────
        probs        = model.predict_proba(aug)[0]
        is_ransomware= probs[1] > 0.5
        risk_score   = int(round(probs[1] * 100))

        diagnosis  = "⚠️ RANSOMWARE DETECTED" if is_ransomware else "✅ SYSTEM BENIGN"
        risk_label = (
            "CRITICAL" if risk_score >= 80 else
            "HIGH"     if risk_score >= 60 else
            "MEDIUM"   if risk_score >= 40 else
            "LOW"      if risk_score >= 20 else
            "CLEAN"
        )

        # ── LIME Explainability ────────────────────────────────────────────────
        all_features = feature_columns + ['anomaly_score']
        bg_data      = lime_background if lime_background is not None else np.zeros((10, len(all_features)))

        from lime.lime_tabular import LimeTabularExplainer
        explainer = LimeTabularExplainer(
            bg_data,
            feature_names=all_features,
            class_names=['Benign', 'Ransomware'],
            mode='classification',
            random_state=42
        )

        def predict_fn(x):
            return model.predict_proba(x)

        exp        = explainer.explain_instance(aug.reshape(-1), predict_fn, num_features=12)
        lime_list  = exp.as_list()

        # LIME features for Chart.js (all 12 features)
        lime_features = []
        for feat_str, weight in lime_list:
            clean = feat_str.split(' ')[0]
            fidx  = all_features.index(clean) if clean in all_features else -1
            fval  = float(aug.reshape(-1)[fidx]) if fidx >= 0 else 0.0
            lime_features.append({
                'feature':    feat_str,
                'clean_name': clean,
                'weight':     round(float(weight), 5),
                'value':      round(fval, 4),
                'direction':  'ransomware' if weight > 0 else 'benign'
            })

        # Top 5 for breakdown table (use original unscaled values)
        raw_vals    = input_df[feature_columns].fillna(0).iloc[0]
        top_features= []
        for feat_str, weight in lime_list[:5]:
            clean   = feat_str.split(' ')[0]
            orig    = float(raw_vals[clean]) if clean in raw_vals.index else 0.0
            top_features.append({
                'name':      clean,
                'value':     f"{orig:.4f}",
                'weight':    round(float(weight), 5),
                'impact':    'Critical' if abs(weight) > 0.15 else 'High' if abs(weight) > 0.08 else 'Medium',
                'direction': 'ransomware' if weight > 0 else 'benign'
            })

        return jsonify({
            'diagnosis':     diagnosis,
            'risk_score':    risk_score,
            'risk_label':    risk_label,
            'anomaly_score': round(anomaly, 5),
            'probabilities': {
                'Benign':     round(float(probs[0]), 4),
                'Ransomware': round(float(probs[1]), 4)
            },
            'lime_features': lime_features,
            'top_features':  top_features,
            'timestamp':     time.strftime("%Y-%m-%d %H:%M:%S"),
            'mode':          'live',
            'filename':      file.filename
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"[START] Ransomware Detection Server (Mode: {MODEL_STATUS})")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
