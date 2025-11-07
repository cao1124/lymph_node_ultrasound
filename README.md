# Lymph Node Ultrasound — Deep Learning–Based Intelligent Classification System

### 基于超声和深度学习的颈部淋巴结智能分类系统

**Affiliation:** 中山医院 / 肿瘤医院 联合项目
**Author(s):** 张伟，徐辉雄 指导
**Code Maintainer:** [@cao1124](https://github.com/cao1124)

---

## 🧠 Project Overview

This repository contains the full codebase and experiment configurations for **deep learning–based ultrasound diagnosis of cervical lymph nodes**, covering a **three-part hierarchical classification framework**:

1. **Benign vs. Malignant Classification（良恶性分类）**
2. **Malignant: Metastatic vs. Non-Metastatic Classification（恶性淋巴转移与非转移分类）**
3. **Lymph Node Subtype Classification（淋巴结亚型分类）**

The system leverages *multi-task learning*, *dual-branch feature fusion*, and *ResNet/Swin Transformer–based architectures* to achieve clinically interpretable and generalizable models for ultrasound image classification.
It serves as an **intelligent assistant** for radiologists to reduce unnecessary biopsies and improve diagnostic accuracy.

---

## 📂 Repository Structure

lymph_node_ultrasound/
│
├── classify_txt.py                  # Base training script (single-image input)
├── classify_util.py                 # Core dataset, model, and training utilities
├── classify_txt_fusion.py           # Two-branch fusion training (big + ROI crop)
├── classify_util_fusion.py          # Fusion model + dual-input dataset definition

├── multitask_one_cls_aux.py         # Multi-task model (main + auxiliary task)
├── multitask_cls.py                 # Generalized multi-task classification framework
├── models/                          # Saved model checkpoints (*.pt)
├── datasets/                        # Dataset list files and preprocessed data
│   └── 中山淋巴良恶性2分类.txt
│   └── 中山淋巴恶性瘤淋巴瘤2分类.txt
│   └── 中山淋巴第三部分细分-转移6分类.txt
│   └── ... ...
└── README.md

---

## ⚙️ Experimental Framework

The project is divided into **three major experimental stages**, corresponding to clinically meaningful diagnostic levels:

### **1️⃣ Benign vs. Malignant Classification（良恶性分类）**

- **Objective:** Differentiate benign from malignant lymph nodes using pathological confirmation.
- **Model:** Auxiliary multi-task learning network — main task: benign/malignant, auxiliary task: suspected enlargement.
- **Performance:**
  - Recall (malignant): **92.2%**
  - F1-score: > 0.86
  - External test accuracy: **80.2%**
- **Clinical Value:** Reduces false negatives for malignant lesions and helps avoid unnecessary biopsies.

### **2️⃣ Malignant Lymph Node: Metastatic vs. Non-Metastatic Classification（恶性淋巴转移与非转移分类）**

- **Objective:** Further distinguish metastatic lymph nodes from lymphomas.
- **Model:** Swin Transformer / ResNet backbone with internal and external testing.
- **Results:**
  - Internal accuracy: **87.9%**, F1 (metastatic) = **0.925**
  - External accuracy: **86.7%**
  - Recall (metastatic): **88.4%**, Precision: **97.4%**
- **Observation:** Model demonstrates strong generalization across medical centers.

### **3️⃣ Lymph Node Subtype Classification（淋巴结亚型分类）**

- **Objective:** Predict the **primary tumor origin** for metastatic lymph nodes.
- **Classes:** Lung, Breast, Esophagus, Nasopharyngeal, Abdominal, Others.
- **Method:** Dual-input feature fusion using **original image + ROI crop**, via a twin-branch ResNet fusion network.
- **Performance:**
  - Accuracy (original): 0.7146
  - Accuracy (crop only): 0.6956[README_final.md](assets/README_final.md)
  - Ongoing experiments using feature fusion show improved discriminability.
- **Significance:** Enables fine-grained tracing of primary malignancy sources and enhances decision support for oncologists.

---

## 🧮 Multi-Task Learning Modules

### **multitask_one_cls_aux.py — Single Main Task + Auxiliary Task**

This script implements a **dual-head multi-task learning framework** for *benign vs. malignant* classification.
It introduces an **auxiliary branch** to jointly predict whether a lymph node is *suspiciously enlarged*, which regularizes the main task and improves the model’s generalization.

**Key Features:**

- Shared convolutional backbone (ResNet/Swin Transformer)
- Two output heads:
  - **Head 1:** Main task — benign vs. malignant classification
  - **Head 2:** Auxiliary task — suspected enlargement classification
- Weighted multi-task loss:
  L_total = α * L_main + β * L_aux
  where typical setting is α = 1.0, β = 0.5
- Joint backpropagation ensures that both lesion-level and morphological information are captured.

**Purpose:**
Improve benign recognition and reduce false positives for malignant lesions by encouraging the model to learn structural cues related to lymph node enlargement.

---

### **multitask_cls.py — General Multi-Task Classification Framework**

This file generalizes the above design to support **multiple simultaneous classification objectives**, allowing flexible definition of auxiliary tasks beyond enlargement detection.

**Key Features:**

- Modular task head registration (each head can correspond to a different diagnostic objective).
- Independent loss weighting for each task via configurable dictionary (`task_weights`).
- Can be extended for:
  - Lesion subtype + malignancy joint learning
  - Origin prediction + clinical indicator estimation
  - Semi-supervised auxiliary tasks (e.g., uncertainty or difficulty prediction)
- Supports early stopping and cross-validation consistent with the main training pipeline.

**Use Case:**
Serves as the unified framework for **Stage Ⅰ (benign–malignant)** and **Stage Ⅱ (metastasis vs. lymphoma)** classification, integrating auxiliary cues to stabilize training on small and imbalanced datasets.

## 🧩 Model Architecture Highlights

- **Feature Extractor:** ResNet-50 / Swin Transformer backbone
- **Fusion Strategy:** Dual-branch concatenation (big image + ROI crop)
- **Optimization:** SGD + Cosine Annealing LR + Warmup Scheduler
- **Augmentation:** Random rotation, flip, color jitter, Gaussian blur
- **Loss:** Weighted Cross-Entropy (class-balanced)
- **Evaluation:** 5-fold cross-validation, internal & external datasets

---

## 🧪 Dataset Summary

All samples are **ultrasound images of pathologically confirmed abnormal lymph nodes**:


| Experiment           | Description                                                | Samples                                                |
| -------------------- | ---------------------------------------------------------- | ------------------------------------------------------ |
| **良恶性分类**       | Benign vs. Malignant                                       | –                                                     |
| **恶性淋巴转移分类** | Metastatic vs. Lymphoma                                    | 1,440 cases (Metastatic = 2,247, Non-metastatic = 894) |
| **亚型6分类**        | Lung, Breast, Esophagus, Nasopharyngeal, Abdominal, Others | 2,247 metastatic samples                               |

External validation includes **multi-center ultrasound data** from Zhongshan and Oncology hospitals, used for human–AI comparative evaluation (20251105 dataset).

---

## 📈 Experimental Results Summary


| Stage | Task             | Internal Acc | External Acc | Notes                         |
| ----- | ---------------- | ------------ | ------------ | ----------------------------- |
| Ⅰ    | 良恶性分类       | 0.8791       | 0.8571       | Multi-task auxiliary learning |
| Ⅱ    | 恶性淋巴转移分类 | 0.8571       | 0.7500       | High recall for metastasis    |
| Ⅲ    | 亚型6分类        | 0.7146       | 0.6956       | Dual-input fusion ongoing     |

---

## 🔬 Clinical Significance

- **Reduces unnecessary biopsies** by improving benign case recognition.
- **Provides origin prediction** for metastatic lymph nodes, aiding tumor tracing.
- **Achieves parity with senior radiologists** in diagnostic performance.
- **Demonstrates potential for cross-institutional deployment**.

---

## 🚀 Future Work

1. **Model optimization:** Enhance benign recall, integrate balanced supervision.
2. **Fusion generalization:** Explore attention-based feature weighting.
3. **Modality completion:** Incorporate GAN-based missing modality synthesis.
4. **Collective intelligence framework:** Ensemble multiple backbones to address long-tail imbalance.

---

## 🧩 Installation & Usage

```bash
git clone https://github.com/cao1124/lymph_node_ultrasound.git
cd lymph_node_ultrasound
pip install -r requirements.txt
```

### Single-image training (Stages 1 & 2)

```bash
python classify_txt.py
```

### Dual-image fusion training (Stage 3: Lymph Node Subtype Classification)

```bash
python classify_txt_fusion.py
```

Training data list format:

```
path/to/image.jpg,肺癌
path/to/image.jpg,乳腺癌
...
```

ROI crop images are automatically located via path substitution:

```
.replace('20251105-最后测试内外部人机对比', '20251105-最后测试内外部人机对比-crop')
```

---

## 📜 Citation

If you use this code or dataset, please cite:

> Zhang W., Xu H., *Deep Learning–Based Intelligent Classification of Cervical Lymph Nodes in Ultrasound Imaging*, Zhongshan Hospital, 2025.
> Code available at: [https://github.com/cao1124/lymph_node_ultrasound](https://github.com/cao1124/lymph_node_ultrasound)

---

## 🧾 License

This repository is released for academic and research purposes only.
Commercial use requires explicit authorization from the project team.
