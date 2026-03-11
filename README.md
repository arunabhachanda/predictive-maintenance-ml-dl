# Predictive Maintenance for Sensor-Driven Systems
### Machine Failure Classification + Remaining Useful Life Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.x-red?logo=keras)](https://keras.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-latest-blue?logo=scikit-learn)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🚨 The Problem This Project Solves

Every year, **unplanned industrial equipment failures cost global manufacturers over $50 billion** in downtime, emergency repairs and lost productivity. Companies like Amazon, Mercedes-Benz and Siemens face this challenge daily across their warehouse robots, assembly line machinery and fleet vehicles.

Traditional maintenance approaches are broken:

| Approach | Problem | Cost |
|----------|---------|------|
| **Reactive** | Wait for breakdown, then fix | Highest — production halts, emergency repairs |
| **Scheduled** | Service every X days regardless | Wasteful — often unnecessary, misses unpredictable failures |
| **Predictive ✅** | Predict failure before it happens | Lowest — targeted, timely, data-driven |

**This project builds a complete, production-grade predictive maintenance system** — going beyond simply predicting *whether* a machine will fail, to identifying *what type* of failure is coming and *exactly how many operational cycles remain* before it occurs.

---

## 🎯 Project Overview

This is a **two-phase end-to-end ML/DL project** that mirrors how real predictive maintenance systems are architected at companies like Amazon and Siemens:

```
All Machines/Engines
        │
        ▼
┌─────────────────────┐
│   PHASE 1a          │  ── Will it fail?
│   Binary            │     LightGBM Classifier
│   Classification    │     F1: 0.79 | ROC-AUC: 0.96
└────────┬────────────┘
         │ YES
         ▼
┌─────────────────────┐
│   PHASE 1b          │  ── What type of failure?
│   Multi-Class       │     Cascaded LightGBM Classifier
│   Classification    │     Overall F1: 0.95 | Accuracy: 0.95
└────────┬────────────┘
         │ Schedule targeted maintenance
         ▼
┌─────────────────────┐
│   PHASE 2           │  ── How many cycles remain?
│   RUL Prediction    │     Tuned GRU Deep Learning Model
│   (Regression)      │     RMSE: 13.80 | Beats literature baseline
└─────────────────────┘
```

---

## 📊 Results at a Glance

### Phase 1 — Machine Failure Classification (AI4I 2020)

| Model | F1 Score | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| Random Forest | 0.60 | 0.49 | 0.78 | 0.9658 |
| XGBoost | 0.67 | 0.57 | 0.81 | 0.9641 |
| LightGBM (default) | 0.71 | 0.62 | 0.84 | 0.9778 |
| **LightGBM (tuned) ⭐** | **0.79** | **0.79** | **0.79** | **0.9604** |

**Failure Type Classification (Phase 1b):**

| Failure Type | F1 Score | Support |
|-------------|----------|---------|
| HDF (Heat Dissipation) | 1.00 | 23 |
| PWF (Power Failure) | 0.97 | 18 |
| OSF (Overstrain) | 0.94 | 16 |
| TWF (Tool Wear) | 0.82 | 9 |
| **Overall** | **0.95** | **66** |

### Phase 2 — Remaining Useful Life Prediction (NASA CMAPSS)

| Model | RMSE (cycles) | MAE (cycles) | vs Literature |
|-------|--------------|-------------|---------------|
| LSTM (default) | 15.59 | 11.27 | Beats baseline |
| GRU (default) | 14.44 | 10.13 | Better |
| **GRU (tuned) ⭐** | **13.80** | **10.09** | **~25% better than baseline** |
| Literature baseline | ~17–20 | ~13–15 | Reference |
| State of the art | ~12–13 | ~9–10 | Transformers + Attention |

> **Our tuned GRU achieves ~25% improvement over the published literature baseline and approaches state-of-the-art performance without attention mechanisms.**

---

## 🏭 Real-World Industry Impact

### 📦 Amazon — Warehouse & Logistics
Amazon operates **40,000+ delivery vans**, thousands of **Kiva warehouse robots** and hundreds of **fulfilment centre conveyor systems**. A single conveyor breakdown during peak season (Prime Day, Black Friday) can halt an entire fulfilment centre processing 1M+ packages per day.

**With this system:**
- Phase 1 flags equipment at risk before failure occurs
- Phase 2 tells maintenance teams exactly how many operational cycles remain
- Maintenance can be scheduled during off-peak hours, preventing costly surprise shutdowns

### 🚗 Automotive — Mercedes-Benz R&D
In automotive R&D, **milling machines and CNC equipment** used in vehicle component testing must operate with zero unplanned downtime. Each test cycle costs thousands of euros.

**With this system:**
- Failure type classification (Phase 1b) tells engineers whether to replace a tool, check power systems or inspect for overstrain
- RUL prediction (Phase 2) allows test schedules to be planned around predicted maintenance windows
- *(Note: This directly connects to my professional experience building vehicle testing data infrastructure at Mercedes-Benz R&D via Capgemini)*

### 🏭 Manufacturing — Siemens / Bosch / GE
Industrial milling machines, assembly line robots and CNC equipment across global manufacturing plants run 24/7. Unplanned downtime on a single production line costs **$260,000+ per hour** (automotive industry average).

**With this system:**
- Cascaded classifier architecture (Phase 1a → 1b) mirrors production ML system design
- RUL prediction enables **proactive, just-in-time maintenance** — neither too early (wasteful) nor too late (breakdown)

### ✈️ Aviation — Lufthansa / Airbus
Jet engine maintenance is one of the most regulated and expensive aspects of aviation. A single unscheduled engine removal costs $1M+.

**With this system:**
- Phase 2 directly addresses turbofan engine RUL prediction (NASA CMAPSS is an aerospace dataset)
- Airlines can predict remaining engine component life and schedule maintenance at the optimal time

### 💻 Data Centres — Google / AWS
Google and AWS operate millions of servers and cooling units. Server hardware degradation directly impacts uptime SLAs.

**With this system:**
- Sensor readings from cooling systems and hardware can be fed into the Phase 2 pipeline
- Predicted RUL allows proactive hardware replacement before failures impact customer services

---

## 🗂️ Repository Structure

```
predictive-maintenance-ml/
│
├── Phase1_AI4I2020/
│   └── Predictive_Maintenance_Phase1_AI4I2020.ipynb
│       ├── EDA & Visualisations
│       ├── Feature Engineering (3 domain-driven features)
│       ├── SMOTE + Preprocessing Pipeline
│       ├── Model Comparison (RF, XGBoost, LightGBM)
│       ├── Hyperparameter Tuning (RandomizedSearchCV)
│       ├── Feature Importance Analysis
│       └── Multi-Class Failure Type Classifier (Phase 1b)
│
├── Phase2_NASA_CMAPSS/
│   └── Predictive_Maintenance_Phase2_NASA_CMAPSS_RUL.ipynb
│       ├── Dataset Loading & EDA
│       ├── Sensor Variance Analysis (7 sensors dropped)
│       ├── RUL Label Engineering + Piecewise Linear Cap
│       ├── Sensor Degradation Visualisation
│       ├── MinMaxScaler + Sliding Window Sequences
│       ├── LSTM Architecture + Training
│       ├── GRU Architecture + Training
│       ├── Hyperparameter Tuning (L2, Dropout, LR, Gradient Clipping)
│       └── Final Evaluation & Model Comparison
│
└── README.md
```

---

## 🧠 Technical Highlights

### Phase 1 — What Makes It Stand Out

**Domain-Driven Feature Engineering**
Three physics-based features were engineered from first principles of rotating machinery — and both top engineered features outperformed ALL raw sensor readings in the final model:

| Feature | Formula | Why |
|---------|---------|-----|
| `Temp_Difference` | Process Temp − Air Temp | Captures thermal stress beyond environment — raw temps move together (r=0.88) |
| `Power` | Torque × Rotational Speed | Fundamental physics equation for rotating systems — resolves -0.88 multicollinearity |
| `Torque_per_Wear` | Torque ÷ (Tool Wear + 1) | Captures stress relative to tool age — the real failure condition |

**Cascaded Classifier Architecture**
Phase 1b (failure type) only runs on machines already flagged by Phase 1a (failure detection). This mirrors real production ML system design — lightweight binary check first, detailed diagnosis only when needed. Avoids unnecessary computation and false type classifications on healthy machines.

**Correct SMOTE Application**
SMOTE applied ONLY on training data after the train-test split. Never on test data. Prevents synthetic samples from leaking into evaluation, ensuring test metrics reflect true real-world performance on the original 3.39% failure rate.

### Phase 2 — What Makes It Stand Out

**Piecewise Linear RUL Assumption**
RUL capped at 125 cycles — solves the contradictory labelling problem where two engines with identical sensor readings would otherwise receive very different RUL labels. Focuses model learning on the informative degradation phase.

**Two-Stage LSTM/GRU Architecture**
```
Input (50 timesteps × 14 sensors)
    ↓
GRU(128, return_sequences=True)   ← learns local patterns
    ↓
BatchNormalization + Dropout(0.2)
    ↓
GRU(64, return_sequences=False)   ← summarises to single vector
    ↓
BatchNormalization + Dropout(0.2)
    ↓
Dense(32, relu) + Dropout(0.15)
    ↓
Dense(1, linear)                  ← continuous RUL prediction
```

**Systematic Tuning Approach**
Each tuning decision was driven by a specific hypothesis:
- `learning_rate 0.001 → 0.0005` — smoother convergence, prevents overshooting
- `Dropout 0.3 → 0.2` — model was over-regularised, freed to learn more
- `L2 regularisation (0.001)` — penalises large weights directly, complementary to dropout
- `Gradient clipping (clipnorm=1.0)` — prevents exploding gradients in BPTT
- `he_uniform initialiser` — correct choice for ReLU activations in Dense layer
- `Batch size 256 → 128` — better gradient estimates per update

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Deep Learning** | TensorFlow 2.19, Keras |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **Data Processing** | Pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Environment** | Google Colab |
| **Version Control** | Git, GitHub |

---

## 📁 Datasets

| Phase | Dataset | Source | Size |
|-------|---------|--------|------|
| Phase 1 | AI4I 2020 Predictive Maintenance | [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | 10,000 rows × 14 cols |
| Phase 2 | NASA CMAPSS Turbofan Engine Degradation | [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) | 20,631 rows × 26 cols |

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)
1. Open the notebook directly in Google Colab
2. Run all cells sequentially
3. For Phase 2: download `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` from [Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and upload when prompted

### Option 2 — Local Environment
```bash
# Clone the repository
git clone https://github.com/arunabhachanda/predictive-maintenance-ml
cd predictive-maintenance-ml

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn tensorflow keras

# Launch Jupyter
jupyter notebook
```

---

## 📈 Key Learnings & Design Decisions

1. **Feature engineering beats raw sensors** — Engineered features (Temp_Difference, Power) ranked #1 and #2 in feature importance, outperforming all 9 raw sensor readings
2. **GRU > LSTM for moderate-length sequences** — 50-timestep sequences are well within GRU's capability; fewer parameters = better generalisation
3. **MinMaxScaler over StandardScaler for neural networks** — aligned with LSTM/GRU's internal sigmoid activations
4. **Recall over Precision in maintenance** — missing a real failure is far costlier than a false alarm
5. **Cascaded classifiers mirror production architecture** — two-stage detection + classification is how real ML systems are built at scale
6. **Piecewise linear RUL cap is essential** — without it, identical sensor readings at different lifecycle stages receive contradictory labels, confusing the model

---

## 🔮 Future Improvements

- [ ] Add **Attention Mechanism** to GRU — the key architectural difference between our model and SOTA (RMSE ~12–13)
- [ ] Implement **Transformer-based architecture** for RUL prediction
- [ ] Extend Phase 2 to **FD002, FD003, FD004** — multiple operating conditions and fault modes
- [ ] Build a **real-time inference API** (Flask/FastAPI) to serve predictions from live sensor streams
- [ ] Add **SHAP values** for model explainability — critical for production deployment in regulated industries
- [ ] **Ensemble** LSTM and GRU predictions for further RMSE reduction

---

## 👨‍💻 Author

**Arunabha Kumar Chanda**
Data Scientist | ML Engineer | Backend Developer

- 🎓 M.Sc. Business Intelligence & Data Science — ISM Munich
- 💼 3+ years backend engineering — Capgemini (Mercedes-Benz R&D) & TCS (Air India)
- 📍 Dachau, Munich, Germany

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/arunabha-kumar-chanda-42401a1b9)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/arunabhachanda)
[![Email](https://img.shields.io/badge/Email-Contact-red?logo=gmail)](mailto:arunabhachanda1998@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*If this project was useful or interesting, consider giving it a ⭐ — it helps others discover it!*
