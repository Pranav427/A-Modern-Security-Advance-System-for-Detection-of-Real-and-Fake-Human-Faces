import os
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

# Import custom modular components
from src.model_loader import load_detection_model
from src.inference import get_prediction

# Constants
APP_DIR = Path(__file__).resolve().parent
GITHUB_URL = "https://github.com/Pranav427/A-Modern-Security-Advance-System-for-Detection-of-Real-and-Fake-Human-Faces"
PORTFOLIO_URL = "https://portfolio-self-one-10.vercel.app"

SUPPORTED_CLASSES = ("Real", "Fake")
SAMPLE_IMAGE_PATH = "1.jpg"

# Research and Evaluation Data (Updated with exact values from the B.Tech Thesis Report)
MODEL_COMPARISON = pd.DataFrame(
    {
        "Model": [
            "Proposed EfficientNetV2-B0 (Our)",
            "ResNet50 Baseline",
            "ResNet34 Baseline",
            "VGG16 Baseline",
        ],
        "Accuracy": [0.9391, 0.9133, 0.8950, 0.8710],
        "Loss": [0.1883, 0.2152, 0.2450, 0.2980],
    }
)

CLASSIFICATION_REPORT = pd.DataFrame(
    {
        "Class": ["Real (CelebA-HQ)", "Fake (StyleGAN)", "Average / Total"],
        "Precision": [0.95, 0.93, 0.94],
        "Recall": [0.93, 0.95, 0.94],
        "F1-score": [0.94, 0.94, 0.94],
        "Support": [70000, 70000, 140000],
    }
)

CONFUSION_MATRIX = pd.DataFrame(
    [[65100, 4900], [3626, 66374]],
    index=["Actual Real", "Actual Fake"],
    columns=["Predicted Real", "Predicted Fake"],
)

DATASET_COUNTS = pd.DataFrame(
    {
        "Category": ["Real (CelebA-HQ)", "Fake (StyleGAN)"],
        "Images": [70000, 70000],
    }
)

# Exact Epoch Training History from Figure 8.3 & 8.4
EPOCH_METRICS = pd.DataFrame(
    {
        "Epoch": list(range(1, 11)),
        "Train Loss": [0.3015, 0.2134, 0.1695, 0.1432, 0.1215, 0.1080, 0.0960, 0.0853, 0.0810, 0.0729],
        "Train Accuracy": [0.8711, 0.9127, 0.9328, 0.9445, 0.9528, 0.9591, 0.9633, 0.9675, 0.9699, 0.9733],
        "Test Loss": [0.2152, 0.1824, 0.2010, 0.1731, 0.1860, 0.1746, 0.1860, 0.2045, 0.1745, 0.1883],
        "Test Accuracy": [0.9133, 0.9261, 0.9215, 0.9322, 0.9343, 0.9398, 0.9373, 0.9385, 0.9404, 0.9391],
    }
)

def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1120px;
            padding-top: 3.4rem;
            padding-bottom: 3rem;
        }
        section[data-testid="stSidebar"] {
            width: 16.5rem !important;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            background: rgba(128, 128, 128, 0.055);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem;
        }
        .app-hero {
            padding: 0.25rem 0 1rem 0;
            margin-bottom: 0.7rem;
        }
        .app-hero h1 { font-size: 2.25rem; line-height: 1.12; margin-bottom: 0.55rem; }
        .app-hero p { font-size: 1.02rem; max-width: 760px; }
        .trust-chip {
            display: inline-block;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 999px;
            padding: 0.28rem 0.68rem;
            margin: 0.15rem 0.2rem 0.15rem 0;
            background: rgba(128, 128, 128, 0.045);
            font-size: 0.86rem;
            color: rgba(245, 245, 245, 0.86);
        }
        .eyebrow {
            color: #7aa2ff;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.02rem;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .muted {
            color: rgba(128, 128, 128, 0.95);
        }
        .result-card {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 10px;
            padding: 1.15rem 1.25rem;
            margin-bottom: 1rem;
        }
        .result-card-real {
            border-color: rgba(16, 185, 129, 0.3);
            background: rgba(16, 185, 129, 0.06);
        }
        .result-card-fake {
            border-color: rgba(239, 68, 68, 0.3);
            background: rgba(239, 68, 68, 0.06);
        }
        .result-label {
            color: rgba(128, 128, 128, 0.95);
            font-size: 0.9rem;
            margin-bottom: 0.1rem;
        }
        .result-value {
            font-size: 1.9rem;
            font-weight: 750;
            line-height: 1.15;
            margin-bottom: 0.8rem;
        }
        .result-value-real {
            color: #10b981;
        }
        .result-value-fake {
            color: #ef4444;
        }
        .academic-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 10px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
            background: rgba(128, 128, 128, 0.08);
        }
        .academic-header {
            font-size: 1.15rem;
            font-weight: bold;
            color: #7aa2ff;
            margin-bottom: 0.6rem;
            text-transform: uppercase;
        }
        .roadmap {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            background: rgba(128, 128, 128, 0.035);
            line-height: 1.9;
        }
        .roadmap-step {
            display: inline-block;
            border: 1px solid rgba(122, 162, 255, 0.34);
            border-radius: 999px;
            padding: 0.24rem 0.62rem;
            margin: 0.18rem;
            background: rgba(122, 162, 255, 0.08);
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar() -> str:
    with st.sidebar:
        st.title("Face Forensics")
        
        if "page" not in st.session_state:
            st.session_state.page = "AI Prediction System"
            
        options = ["AI Prediction System", "Technical Case Study"]
        default_index = options.index(st.session_state.page) if st.session_state.page in options else 0
        
        selected_page = st.radio(
            "Choose page",
            options,
            label_visibility="collapsed",
            index=default_index,
        )
        
        st.session_state.page = selected_page
        
        st.divider()
        if GITHUB_URL:
            st.link_button("GitHub", GITHUB_URL, use_container_width=True)
        if PORTFOLIO_URL:
            st.link_button("Portfolio", PORTFOLIO_URL, use_container_width=True)
        st.divider()
        st.caption("Applied Deep Learning Project")
        st.write("Detecting synthetic features to secure digital authentication systems.")
    return selected_page

def render_prediction_result(
    verdict: str,
    confidence: float,
    image_file
) -> None:
    st.subheader("Analysis Result")

    card_style = "result-card-real" if verdict == "Real" else "result-card-fake"
    text_style = "result-value-real" if verdict == "Real" else "result-value-fake"

    st.markdown(f'<div class="result-card {card_style}">', unsafe_allow_html=True)
    st.markdown('<div class="result-label">Predicted Authenticity</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result-value {text_style}">{verdict}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    result_cols = st.columns(2)
    result_cols[0].metric("Confidence", f"{confidence * 100:.2f}%")
    result_cols[1].metric("Target Resolution", "256x256")

    st.progress(confidence, text="Model prediction confidence")

    if verdict == "Real":
        st.success(
            "The model did not detect high-frequency generator fingerprints. "
            "This image is likely a camera-captured real face."
        )
    else:
        st.error(
            "Synthesizer artifacts detected. The model suggests this image "
            "was generated by a neural network (e.g., StyleGAN)."
        )

    with st.expander("Technical Model Details"):
        st.write("Model input dimensions:")
        st.code("Tensor Shape: (1, 256, 256, 3)", language="text")
        st.caption(
            "The confidence metric reflects the neural network's final sigmoid activation margin. "
            "It indicates proximity to the learned classification boundary."
        )

def render_prediction_page(model) -> None:
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">AI Prediction System</div>
            <h1>Real vs Fake Face Detector</h1>
            <p class="muted">
            An applied computer vision dashboard that distinguishes real human faces from 
            deepfake/synthetic faces using a custom-weighted EfficientNetV2-B0 architecture.
            </p>
            <span class="trust-chip">EfficientNetV2</span>
            <span class="trust-chip">Keras / TensorFlow</span>
            <span class="trust-chip">93.91% Test Accuracy</span>
            <span class="trust-chip">140k dataset</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "test_image" not in st.session_state:
        st.session_state.test_image = None
        st.session_state.test_image_name = ""

    with st.container(border=True):
        st.subheader("Image Analyzer")
        st.caption("Upload a JPEG/PNG face image to classify.")

        uploaded_file = st.file_uploader(
            "Upload a face image to classify:",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            st.session_state.test_image = uploaded_file
            st.session_state.test_image_name = uploaded_file.name
        else:
            st.session_state.test_image = None
            st.session_state.test_image_name = ""

        if st.session_state.test_image is not None:
            st.divider()
            col_img, col_act = st.columns([1, 1])

            with col_img:
                img = Image.open(st.session_state.test_image)
                st.image(img, use_column_width=True, caption=st.session_state.test_image_name)

            with col_act:
                st.write("### Model Execution")
                st.write("Press the button below to feed this frame into the feature-extraction pipeline.")
                
                predict_btn = st.button("Run Inference", type="primary", use_container_width=True)

                if predict_btn:
                    with st.spinner("Executing model prediction..."):
                        try:
                            verdict, confidence = get_prediction(model, st.session_state.test_image)
                            render_prediction_result(verdict, confidence, st.session_state.test_image)
                        except Exception as e:
                            st.error(f"Prediction execution failed: {str(e)}")
        else:
            st.info("Upload an image or load the sample face to run the detector.")

    st.divider()
    info_cols = st.columns([1, 1, 1])
    with info_cols[0]:
        with st.container(border=True):
            st.subheader("Model Summary")
            st.write("Optimized EfficientNetV2-B0 deep convolutional network, trained to identify artifacts in synthetic faces.")
            st.caption("Accuracy: 93.91% | Parameters: 5.9M")
            st.link_button("View Springer Paper", "https://link.springer.com/chapter/10.1007/978-3-031-92854-3_16", use_container_width=True)
    with info_cols[1]:
        with st.container(border=True):
            st.subheader("Technical Case Study")
            st.write("Explore dataset metrics, model selection, loss logs, confusion matrices, and the Springer research path.")
            if st.button("Open Technical Case Study", use_container_width=True):
                st.session_state.page = "Technical Case Study"
                st.rerun()
    with info_cols[2]:
        with st.container(border=True):
            st.subheader("Responsible AI")
            st.write("This model classifies based on StyleGAN vs CelebA training features. It should not be used as a standalone security system.")
            st.caption("Educational / research showcase project.")

def render_case_section(title: str, body: str) -> None:
    st.subheader(title)
    st.write(body)

def render_roadmap() -> None:
    steps = [
        "Business Problem",
        "Dataset Selection",
        "Preprocessing",
        "EfficientNetV2 Setup",
        "Transfer Learning",
        "Training (10 Epochs)",
        "Validation / Logs",
        "Baseline Comparison",
        "Springer Publication",
        "Streamlit Deployment",
    ]
    html_steps = " ".join(f'<span class="roadmap-step">{step}</span>' for step in steps)
    st.markdown(f'<div class="roadmap">{html_steps}</div>', unsafe_allow_html=True)

def render_case_study_page() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Technical Case Study</div>
            <h1>Research & Thesis Walkthrough</h1>
            <p class="muted">
            Academic specifications, model design steps, training parameters,
            and comparative baselines of our peer-reviewed Springer research.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Academic Credentials Card
    st.markdown(
        """
        <div class="academic-card">
            <div class="academic-header">🎓 Academic & Thesis Credentials</div>
            <table style="width:100%; border:none; font-size:0.92rem; line-height:1.6;">
                <tr>
                    <td style="width:20%; font-weight:bold; color:#7aa2ff;">Project Title:</td>
                    <td>A Modern Security Advance System for Detection of Real and Fake Human Faces</td>
                </tr>
                <tr>
                    <td style="font-weight:bold; color:#7aa2ff;">Project Associates:</td>
                    <td>OBILI PRANAV, SAKAMURI PRANAVI, BOLLU SAI NAVYA SREE, YEDDULA PUNEETH REDDY</td>
                </tr>
                <tr>
                    <td style="font-weight:bold; color:#7aa2ff;">Academic Advisor:</td>
                    <td>Dr. P. Changamma, Assistant Professor in CSE, AITS</td>
                </tr>
                <tr>
                    <td style="font-weight:bold; color:#7aa2ff;">Institution:</td>
                    <td>Department of Computer Science and Engineering, Annamacharya Institute of Technology and Sciences (AITS), Rajampet</td>
                </tr>
                <tr>
                    <td style="font-weight:bold; color:#7aa2ff;">Publication Info:</td>
                    <td>Presented at the First International Conference on Emerging Technologies and Computing Innovations (ICETCI-2025) • Publication Partner: Springer Nature</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_cols = st.columns(4)
    overview_cols[0].metric("Total Dataset Images", "140,000")
    overview_cols[1].metric("Model Test Accuracy", "93.91%")
    overview_cols[2].metric("Total Training Time", "6,264.28s")
    overview_cols[3].metric("Plagiarism Index", "21% (Safe)")

    st.subheader("Engineering Roadmap")
    render_roadmap()

    tabs = st.tabs(["Problem & Dataset", "Architecture", "Training & Convergence", "Evaluation & Baselines", "Academic Integrity"])

    with tabs[0]:
        left, right = st.columns([1, 1])
        with left:
            render_case_section(
                "Abstract & Scientific Problem Statement",
                "With the rapid advancement of artificial intelligence, the ability to generate highly realistic synthetic human faces has significantly improved, raising serious concerns about security and identity verification. These hyper-realistic fake faces pose a threat to surveillance, access control, and online authentication systems. Traditional methods fail to extract subtle pixel-level pattern discrepancies. This work designs a robust deep learning classifier trained to detect high-frequency generator artifacts."
            )
            st.bar_chart(DATASET_COUNTS.set_index("Category"))
        with right:
            render_case_section(
                "Dataset Specifications",
                "The training subset is sourced from the 140k Real and Fake Faces dataset, structured with an even 1:1 class ratio:\n\n"
                "- **Real Faces:** 70,000 images from the CelebA-HQ dataset (human camera-captured faces).\n"
                "- **Fake Faces:** 70,000 images generated via StyleGAN neural networks.\n\n"
                "This class balance prevents classification bias, ensuring the network evaluates features equally."
            )
            st.info(
                "All input images are resized to 256x256, normalized, and augmented (rotations, flips, brightness shifts) to increase generalization."
            )

    with tabs[1]:
        render_case_section(
            "Model Architecture Config",
            "Our neural network employs the EfficientNetV2-B0 base model for feature extraction, combined with a custom classifier block to output sigmoid probability scores."
        )
        st.graphviz_chart(
            """
            digraph {
                rankdir=LR;
                node [shape=box, style="rounded"];
                InputImage -> EfficientNetV2Base -> GlobalAveragePooling -> DenseClassificationHead -> SigmoidActivation;
            }
            """
        )
        st.markdown(
            "**Key architectural layers:**\n"
            "- **Base Extractor:** `EfficientNetV2-B0` (retains parameter efficiency while utilizing MBConv and Fused-MBConv layers)\n"
            "- **Pooling Layer:** `GlobalAveragePooling2D` (compresses feature map outputs into a flat 1D vector)\n"
            "- **Classification Head:** `Dense` output layer with a single unit and sigmoid activation function generating prediction margins in `[0.0, 1.0]`."
        )

    with tabs[2]:
        render_case_section(
            "Training Convergence Logs",
            "The model was compiled with binary cross-entropy loss and optimized via Adam. The training took 6264.28 seconds across 10 epochs. Convergence plots display smooth decay, indicating good fit."
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.write("Validation Accuracy over Epochs (Peaked at 94.04%)")
            st.line_chart(EPOCH_METRICS.set_index("Epoch")["Test Accuracy"])
        with col_c2:
            st.write("Train Loss Decay over Epochs (Ended at 0.0729)")
            st.line_chart(EPOCH_METRICS.set_index("Epoch")["Train Loss"])

    with tabs[3]:
        render_case_section(
            "Baseline Model Comparison",
            "Our model was compared against several baseline architectures (ResNet50, ResNet34, VGG16) to validate its accuracy-to-parameter ratio efficiency."
        )
        st.bar_chart(MODEL_COMPARISON.set_index("Model")["Accuracy"])
        st.dataframe(MODEL_COMPARISON, use_container_width=True, hide_index=True)
        
        st.subheader("Classification Report")
        st.dataframe(CLASSIFICATION_REPORT, use_container_width=True, hide_index=True)

        st.subheader("Confusion Matrix")
        st.dataframe(CONFUSION_MATRIX, use_container_width=True)

    with tabs[4]:
        render_case_section(
            "Academic Integrity & Plagiarism Compliance",
            "To qualify for publication with Springer Nature and support MSc admissions screening, the project report was subjected to rigorous originality checks."
        )
        st.markdown(
            "- **Similarity Index:** 21% (Well within standard acceptable academic thresholds)\n"
            "- **Internet Sources:** 12% similarity\n"
            "- **Publications:** 10% similarity\n"
            "- **Student Papers:** 11% similarity\n\n"
            "This confirms the original contribution of the implementation, documenting high academic integrity for review boards."
        )

    st.divider()
    final_cols = st.columns(3)
    with final_cols[0]:
        with st.container(border=True):
            st.subheader("Challenges")
            st.write("Adapting deep learning features to detect artifacts across modern generators like Stable Diffusion, which have different noise profiles.")
    with final_cols[1]:
        with st.container(border=True):
            st.subheader("Lessons Learned")
            st.write("Restructuring legacy Keras 2 configurations into code-based Keras 3 loader pipelines guarantees environment reproducibility.")
    with final_cols[2]:
        with st.container(border=True):
            st.subheader("Future Scope")
            st.write("Expanding classification targets to real-time video frames and incorporating visual explanations via Grad-CAM heatmaps.")

# Page Configuration
st.set_page_config(
    page_title="Real vs Fake Face Detection",
    page_icon="🤖",
    layout="wide",
)

apply_page_styles()

# Load navigation page
page = render_sidebar()

# Initialize resources
try:
    with st.spinner("Loading Deep Learning model..."):
        model = load_detection_model()
except Exception as e:
    st.error("The application could not start because model weights are missing or corrupt.")
    st.exception(e)
    st.stop()

# Render appropriate page view
if page == "AI Prediction System":
    render_prediction_page(model)
else:
    render_case_study_page()
