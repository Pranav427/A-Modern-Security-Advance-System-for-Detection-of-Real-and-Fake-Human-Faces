# Face Anti-Spoof Detection
### A Modern Security Advance System for Detection of Real and Fake Human Faces

[![Springer Paper](https://img.shields.io/badge/Springer-Published-orange.svg)](https://link.springer.com/chapter/10.1007/978-3-031-92854-3_16)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Demo-red.svg)](https://real-vs-fake-face.streamlit.app/)
[![Keras / TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-blue.svg)](https://tensorflow.org/)

To address the security risks posed by hyper-realistic AI-generated faces, this repository implements a deployment-ready deep learning system for real-time face anti-spoof detection. Utilizing a customized lightweight EfficientNetV2-B0 architecture, this work contains the official implementation of our peer-reviewed research published in **Springer Nature (ICETCI 2025)**, packaged with a modular backend and deployed as an interactive Streamlit application suitable for demonstration and evaluation.

🔗 **Read the Research Paper:** [Springer Link](https://link.springer.com/chapter/10.1007/978-3-031-92854-3_16)  
🚀 **Live Application Demo:** [real-vs-fake-face.streamlit.app](https://real-vs-fake-face.streamlit.app/)  
💼 **Developer Portfolio:** [portfolio-self-one-10.vercel.app](https://portfolio-self-one-10.vercel.app/)

---

## 📖 Research Abstract & Problem Statement

As generative models (such as StyleGAN) produce hyper-realistic human faces indistinguishable from real photos, they pose serious security and verification challenges. This project implements a modern detection pipeline:

1. **Model Architecture:** We employ an optimized **EfficientNetV2-B0** classifier, customized and fine-tuned for high-accuracy binary classification (Real vs. Fake).
2. **Dataset:** Trained on the **140k Real and Fake Faces** dataset, mapping complex local features and high-frequency generator artifacts.
3. **Key Findings:** Achieved a peak test accuracy of **93.91%**, demonstrating robust performance in digital forensics scenarios.

![Dataset Specifications](assets/dataset_specifications.png)

---

## ✨ Key Features
* **Production-Ready Dashboard:** Interactive Streamlit interface styled with custom CSS layout overrides.
* **Highly Optimized Extractor:** Leverages a modified EfficientNetV2-B0 backbone with Fused-MBConv layers, achieving high accuracy with only 5.9M parameters.
* **Dual-Framework Baseline Pipeline:** Supports TensorFlow/Keras for live dashboard serving and PyTorch for offline baseline model evaluations.
* **Springer-Published Research:** Displays peer-reviewed credentials and integrated academic similarity indices directly in the UI.

---

## 🛠️ Engineering Highlights
* **Dynamic Model Deserialization:** The model loader reconstructs the network dynamically in Python code before injecting weights, resolving deserialization compatibility conflicts between Keras 2.x and Keras 3.x.
* **Magic-Byte Header Sanitization:** Uploaded file streams are validated at the raw binary header level for JPEG/PNG signatures before instantiating image buffers to protect server instances against arbitrary file injection.
* **Optimized Aspect Constraints:** Constrains vertical preview bounds via thumbnail scaling, aligning the visual pipeline side-by-side across all viewport ratios.

---

## 📊 Model Performance
The classifier was trained over 10 epochs using an Adam optimizer and Binary Cross-Entropy loss. Below is the final evaluation summary matching the published metrics:

| Phase | Accuracy | Loss | Training Duration |
| :--- | :--- | :--- | :--- |
| **Training Set** | 97.33% | 0.0729 | 6,264.28 seconds |
| **Test Set** | **93.91%** | 0.1883 | (Inference: ~0.8s) |

---

## 🖥️ Dashboard Preview
![System Dashboard Preview](assets/dashboard_preview.png)

## ⚙️ System Architecture

```mermaid
graph TD
    A[User Image Input] --> B[Image Preprocessing RGB Conversion & 256x256 Resizing]
    B --> C[EfficientNetV2-B0 Neural Network]
    C --> D{Sigmoid Output Probability}
    D -- Probability >= 0.5 --> E[Predicted class: REAL]
    D -- Probability < 0.5 --> F[Predicted class: FAKE]
    E --> G[Streamlit Premium Visualizer]
    F --> G
```

---

## 📁 Repository Structure

```
├── app.py                     # Main Streamlit dashboard entrypoint
├── src/                       # Production application source code
│   ├── styles.py              # Premium CSS layout injections
│   ├── model_loader.py        # Safe model loading & error handling
│   └── inference.py           # Preprocessing & model inference
├── notebooks/                 # Model training and notebook development
│   └── model_training.ipynb   # Original training workflow & evaluation logs
├── research/                  # PyTorch model architecture definitions
│   └── model_pytorch.py       # ResNet50 baseline research code
└── requirements.txt           # Python environment dependencies
```

---

## 🚀 Setup & Execution

### 1. Prerequisites
Ensure you have Python 3.9 or higher installed.

### 2. Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Pranav427/A-Modern-Security-Advance-System-for-Detection-of-Real-and-Fake-Human-Faces.git
cd A-Modern-Security-Advance-System-for-Detection-of-Real-and-Fake-Human-Faces
pip install -r requirements.txt
```

### 3. Model Weights
Place your trained Keras/TensorFlow weight file `dffnetv2B0.h5` in the root directory. 

*(Note: `dffnetv2B0.json` is no longer required as model structure deserialization is now constructed dynamically via code for cross-version compatibility).*

### 4. Multi-Framework Comparative Research
This project evaluates both TensorFlow and PyTorch baselines for experimental validation:
* **Production Model (TensorFlow/Keras):** Deployed live in the Streamlit application for high-performance CPU inference.
* **Baseline Research Baselines (PyTorch):** Scripts located in [research/](research/) (e.g. `model_pytorch.py`) are kept for training baseline comparisons against ResNet architectures.

### 5. Running the Local Demo
Launch the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

---

## 🔒 Responsible AI & Forensic Boundaries
This model was trained on the `140k-real-and-fake-faces` dataset (CelebA-HQ vs. StyleGAN). Because of this:
* **Scope:** The model is optimized to identify GAN-specific (StyleGAN) structural fingerprints and high-frequency noise artifacts.
* **Boundaries:** It is not calibrated to detect modern Latent Diffusion outputs (e.g. Midjourney, Stable Diffusion, Flux) or active deepfake video face-swapping algorithms. For production-grade security, we recommend retraining the base extractor on a multi-generator hybrid dataset.

---

## 🎓 Academic Integrity & Plagiarism Check
In compliance with research integrity standards (crucial for MSc admissions screening):
* **Similarity Index:** 21% (Well within standard acceptable academic thresholds)
* **Primary Sources:** 12% Internet, 10% Publications, 11% Student papers.
This confirms the authenticity of the codebase and thesis implementation.

![Academic Credentials](assets/academic_credentials.png)

---

## 📑 How to Cite

If you use this codebase or research in your studies, please cite our Springer publication:

```bibtex
@InProceedings{obili2025modern,
  author    = {Obili, Pranav},
  title     = {A Modern Security Advance System for Detection of Real and Fake Human Faces},
  booktitle = {International Conference on Advanced Security Systems},
  year      = {2025},
  publisher = {Springer},
  doi       = {10.1007/978-3-031-92854-3_16}
}
```
