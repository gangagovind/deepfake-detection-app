"""
Signal Forensics Lab — Deepfake Detection (Streamlit deployment)
Dual-stream EfficientNet-B0 (RGB + FFT) — built on Person 2's model_utils.py
and Person 1's MTCNN config, re-skinned with a frequency-domain-forensics theme.
"""

import os
import glob
import json
import shutil
import subprocess
import tempfile
import uuid

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN
from PIL import Image

st.set_page_config(page_title="DeepFake Detection With Compression Resilience", page_icon="◉", layout="wide")

# ─────────────────────────────────────────────────────────────────────────
# 1. THEME ENGINE
# ─────────────────────────────────────────────────────────────────────────

BG = "#0B0E14"
PANEL = "#131822"
TEXT = "#E4E9F0"
MUTED = "#7C8798"
MINT = "#4CFFB3"
CORAL = "#FF5D5D"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] {{
    background-color: {BG} !important;
    color: {TEXT} !important;
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
}}
[data-testid="stSidebar"] {{
    background-color: {PANEL} !important;
    border-right: 1px solid #232B3A;
}}
.stButton>button {{
    background-color: {MINT} !important;
    color: {BG} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600;
    border: none !important;
    border-radius: 6px !important;
}}
.stButton>button:disabled {{
    background-color: #2A3140 !important;
    color: {MUTED} !important;
}}
[data-testid="stFileUploader"] {{
    background-color: {PANEL} !important;
    border: 1px dashed #2A3140 !important;
    border-radius: 10px !important;
    padding: 1rem;
}}
.verdict-real {{
    color: {MINT}; font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem; font-weight: 700;
}}
.verdict-fake {{
    color: {CORAL}; font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem; font-weight: 700;
}}
.mono {{ font-family: 'JetBrains Mono', monospace; color: {MUTED}; font-size: 0.85rem; }}
.waveform {{
    height: 3px; margin: 1.5rem 0;
    background: linear-gradient(90deg, transparent, {MINT}, transparent);
    background-size: 200% 100%;
    animation: scan 2.5s linear infinite;
}}
@keyframes scan {{ 0% {{background-position: 200% 0;}} 100% {{background-position: -200% 0;}} }}
.caption-note {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    color: {MUTED}; border-left: 2px solid #2A3140; padding-left: 0.6rem;
    margin-top: 0.4rem;
}}
</style>
""", unsafe_allow_html=True)


def waveform_divider():
    st.markdown('<div class="waveform"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# 2. MODEL DEFINITIONS (unchanged logic from model_utils.py)
# ─────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def fft_preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    f_shifted = np.fft.fftshift(f)
    magnitude = np.abs(f_shifted)
    log_magnitude = np.log1p(magnitude)
    norm = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min())
    norm_3ch = np.stack([norm, norm, norm], axis=-1)
    normalized = (norm_3ch - IMAGENET_MEAN) / IMAGENET_STD
    return normalized, norm


class DualStreamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_branch = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.fft_branch = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(2560, 1)

    def forward(self, rgb_input, fft_input):
        rgb_feat = self.rgb_branch(rgb_input)
        fft_feat = self.fft_branch(fft_input)
        fused = torch.cat([rgb_feat, fft_feat], dim=1)
        return self.classifier(self.dropout(fused))

class SingleStreamModel(nn.Module):
    def __init__(self, branch_name="rgb_branch"):
        super().__init__()
        setattr(self, branch_name, timm.create_model("efficientnet_b0", pretrained=False, num_classes=0))
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(1280, 1)
        self.branch_name = branch_name

    def forward(self, x):
        branch = getattr(self, self.branch_name)
        return self.classifier(self.dropout(branch(x)))

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.dirname(os.path.abspath(__file__)))

CHECKPOINTS = {
    "compression_aware": "model_compression_aware_epoch1.pth",
    "baseline": "model_baseline_epoch7.pth",
    "rgb_only": "model_rgb_only_epoch5.pth",
    "fft_only": "model_fft_only_epoch8.pth",
}
DEFAULT_LABEL = "our robustness-trained model"


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_filename):
    model_path = os.path.join(MODELS_DIR, checkpoint_filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "rgb_only" in checkpoint_filename:
        model = SingleStreamModel(branch_name="rgb_branch").to(device)
    elif "fft_only" in checkpoint_filename:
        model = SingleStreamModel(branch_name="fft_branch").to(device)
    else:
        model = DualStreamModel().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

@st.cache_resource(show_spinner=False)
def load_mtcnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MTCNN(image_size=224, margin=20, keep_all=False, device=device)

def predict_face_probability(model, device, face_rgb_uint8, is_single_stream=False, stream_type=None):
    img = cv2.resize(face_rgb_uint8, (224, 224))

    if is_single_stream:
        if stream_type == "rgb_branch":
            rgb_norm = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
            tensor = torch.tensor(rgb_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            _, fft_vis = fft_preprocess(img)
            with torch.no_grad():
                logit = model(tensor)
            return torch.sigmoid(logit).item(), fft_vis
        else:
            fft_img, fft_vis = fft_preprocess(img)
            tensor = torch.tensor(fft_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(tensor)
            return torch.sigmoid(logit).item(), fft_vis

    rgb_norm = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    rgb_tensor = torch.tensor(rgb_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    fft_img, fft_vis = fft_preprocess(img)
    fft_tensor = torch.tensor(fft_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(rgb_tensor, fft_tensor)
        prob_real = torch.sigmoid(logit).item()
    return prob_real, fft_vis

# ─────────────────────────────────────────────────────────────────────────
# 3. UPLOAD SAFETY: ffprobe validation, UUID paths, scoped cleanup
# ─────────────────────────────────────────────────────────────────────────

def is_valid_video(path):
    """Reject anything that isn't a real, readable video before FFmpeg/MTCNN touch it."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=15,
    )
    return result.returncode == 0 and "video" in result.stdout


def save_upload_safely(uploaded_file):
    """Never trust the user's filename — generate a UUID working path."""
    session_dir = tempfile.mkdtemp(prefix=f"sfl_{uuid.uuid4().hex[:8]}_")
    dest = os.path.join(session_dir, "input.mp4")
    with open(dest, "wb") as f:
        f.write(uploaded_file.read())
    return dest, session_dir


def extract_faces_from_video(video_path, mtcnn, frame_interval=10, max_faces=40):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0, 0
    faces, frame_idx, total_read = [], 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            face_tensor = mtcnn(pil_img)
            if face_tensor is not None:
                face_np = (
                    (face_tensor.permute(1, 2, 0).numpy() * 128 + 127.5)
                    .clip(0, 255).astype("uint8")
                )
                faces.append(face_np)
            if len(faces) >= max_faces:
                break
        frame_idx += 1
    cap.release()
    return faces, frame_idx, total_read


def score_video(video_path, model, device, mtcnn, frame_interval, max_faces, checkpoint_filename=""):
    faces, sampled_frames, total_read = extract_faces_from_video(video_path, mtcnn, frame_interval, max_faces)
    if faces is None:
        return {"error": "Could not read this as a video file."}
    if len(faces) == 0:
        return {"error": f"No face detected in {sampled_frames} sampled frames."}

    is_single = "rgb_only" in checkpoint_filename or "fft_only" in checkpoint_filename
    stream_type = "rgb_branch" if "rgb_only" in checkpoint_filename else ("fft_branch" if "fft_only" in checkpoint_filename else None)

    probs, fft_samples = [], []
    for i, face in enumerate(faces):
        p, fft_vis = predict_face_probability(model, device, face, is_single_stream=is_single, stream_type=stream_type)
        probs.append(p)
        if i < 3:
            fft_samples.append((face, fft_vis))
    return {
        "score": float(np.mean(probs)), "probs": probs, "fft_samples": fft_samples,
        "n_faces": len(faces), "sampled_frames": sampled_frames, "total_frames": total_read,
    }

# ─────────────────────────────────────────────────────────────────────────
# 4. COMPRESSION-COMPARISON HELPERS
# ─────────────────────────────────────────────────────────────────────────

CRF_PRESETS = {"crf23": (23, "1280:720"), "crf28": (28, "854:480"), "crf35": (35, "640:360")}


def reencode_video(input_path, crf, scale, session_dir):
    output_path = os.path.join(session_dir, f"reencode_{uuid.uuid4().hex[:6]}.mp4")
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", f"scale={scale}",
           "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", output_path]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    return output_path


# ─────────────────────────────────────────────────────────────────────────
# 5. EVALUATION RESULTS DATA (loaded from your results/ JSONs)
# ─────────────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


@st.cache_data(show_spinner=False)
def load_results():
    def _load(name):
        path = os.path.join(RESULTS_DIR, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None
    return {
        "ablation": _load("all_ablation_results.json"),
        "df40_family": _load("df40_family_level_results.json"),
        "error_analysis": _load("error_analysis.json"),
    }


# ─────────────────────────────────────────────────────────────────────────
# 6. UI — SIDEBAR
# ─────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## DeepFake Detection With Compression Resilience")
    
    st.markdown('<span class="mono">dual-stream RGB+FFT detector</span>', unsafe_allow_html=True)
    waveform_divider()
    research_mode = st.toggle("Research mode", value=False)
    if research_mode:
        checkpoint_choice = st.selectbox("Checkpoint", list(CHECKPOINTS.keys()))
        frame_interval = st.slider("Sample every Nth frame", 5, 30, 10)
        max_faces = st.slider("Max frames analyzed", 10, 60, 24)
    else:
        checkpoint_choice = "compression_aware"
        frame_interval, max_faces = 10, 24

tab_detect, tab_eval = st.tabs(["◉ Detect", "▤ Evaluation Results"])

# ─────────────────────────────────────────────────────────────────────────
# 7. TAB: DETECT
# ─────────────────────────────────────────────────────────────────────────

with tab_detect:
    st.markdown("## DeepFake Detection With Compression Resilience")
    st.markdown(
        '<span class="mono">Upload a video. We analyze faces in both the pixel domain '
        'and the frequency domain to flag synthetic content.</span>', unsafe_allow_html=True
    )
    waveform_divider()

    uploaded = st.file_uploader("Drop a video for analysis", type=["mp4", "mov", "avi", "mkv", "webm"])

    if uploaded is not None:
        if uploaded.size > 100 * 1024 * 1024:
            st.error("File exceeds the 100MB limit for this demo.")
            st.stop()

        video_path, session_dir = save_upload_safely(uploaded)

        try:
            if not is_valid_video(video_path):
                st.error("This doesn't look like a valid video file.")
                st.stop()

            checkpoint_path = CHECKPOINTS[checkpoint_choice]
            if not os.path.exists(os.path.join(MODELS_DIR, checkpoint_path)):
                st.error(f"Checkpoint file missing: `{checkpoint_path}`. Place it next to app.py.")
                st.stop()

            with st.spinner("Loading model..."):
                model, device = load_model(checkpoint_path)
                mtcnn = load_mtcnn()

            with st.spinner("Extracting faces and analyzing..."):
                result = score_video(video_path, model, device, mtcnn, frame_interval, max_faces, checkpoint_filename=checkpoint_path)

            if "error" in result:
                st.error(result["error"])
                st.stop()

            score = result["score"]
            verdict = "REAL" if score > 0.5 else "FAKE"
            confidence = score if verdict == "REAL" else 1 - score

            waveform_divider()
            col1, col2 = st.columns([1, 1])
            with col1:
                if verdict == "REAL":
                    st.markdown(f'<div class="verdict-real">✓ REAL</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="verdict-fake">⚠ FAKE</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="mono">confidence {confidence*100:.1f}%</span>', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="mono">Sampled every {frame_interval}th frame, '
                    f'{result["n_faces"]} frames analyzed.</span>', unsafe_allow_html=True
                )
                if research_mode:
                    st.caption(f"Checkpoint: `{checkpoint_path}`")
                else:
                    st.caption(f"Analyzed using {DEFAULT_LABEL}.")

            with col2:
                fig, ax = plt.subplots(facecolor=BG)
                ax.set_facecolor(BG)
                ax.plot(result["probs"], color=MINT, marker="o", markersize=3)
                ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
                ax.set_ylim(0, 1)
                ax.tick_params(colors=MUTED)
                for spine in ax.spines.values():
                    spine.set_color("#2A3140")
                st.pyplot(fig)

            waveform_divider()

            with st.expander("See how this was calculated"):
                st.write(
                    f"Each sampled frame is passed through a face detector, then scored "
                    f"independently. The final verdict is the mean probability across "
                    f"{result['n_faces']} frames, thresholded at 0.5."
                )

            with st.expander("What the model is looking at"):
                st.caption("Left: detected face (pixel-domain input) — Right: frequency-domain signature")
                cols = st.columns(max(len(result["fft_samples"]), 1))
                for c, (face, fft_vis) in zip(cols, result["fft_samples"]):
                    with c:
                        st.image(face, use_container_width=True)
                        st.image((fft_vis * 255).astype("uint8"), use_container_width=True, clamp=True)

            waveform_divider()
            st.markdown("### Test compression robustness")
            st.caption(
                "Re-encodes your video at three compression levels and re-scores each — "
                "this is the model's core strength: staying accurate even under heavy compression."
            )

            if "compress_running" not in st.session_state:
                st.session_state.compress_running = False

            run_disabled = st.session_state.compress_running
            if st.button("Run compression comparison", disabled=run_disabled):
                st.session_state.compress_running = True
                levels_data = [("Original", score)]
                for label, (crf, scale) in CRF_PRESETS.items():
                    with st.spinner(f"Testing at {label} (CRF {crf})..."):
                        try:
                            reencoded = reencode_video(video_path, crf, scale, session_dir)
                            r = score_video(reencoded, model, device, mtcnn, frame_interval, max_faces)
                            os.remove(reencoded)
                            if "error" not in r:
                                levels_data.append((label, r["score"]))
                        except subprocess.CalledProcessError:
                            st.warning(f"Re-encoding failed for {label}.")
                st.session_state.compress_running = False

                fig2, ax2 = plt.subplots(facecolor=BG)
                ax2.set_facecolor(BG)
                labels = [l for l, _ in levels_data]
                scores = [s for _, s in levels_data]
                colors = [MINT if s > 0.5 else CORAL for s in scores]
                ax2.bar(labels, scores, color=colors)
                ax2.axhline(0.5, color=MUTED, linestyle="--")
                ax2.set_ylim(0, 1)
                ax2.tick_params(colors=MUTED)
                for spine in ax2.spines.values():
                    spine.set_color("#2A3140")
                st.pyplot(fig2)

        finally:
            shutil.rmtree(session_dir, ignore_errors=True)
    else:
        st.info("Upload a video to begin.")

# ─────────────────────────────────────────────────────────────────────────
# 8. TAB: EVALUATION RESULTS
# ─────────────────────────────────────────────────────────────────────────

with tab_eval:
    st.markdown("## Evaluation Results")
    waveform_divider()
    results = load_results()

    if results["ablation"]:
        st.markdown("### Ablation: does dual-stream + compression-aware training matter?")
        levels = ["original", "crf23", "crf28", "crf35"]
        fig, ax = plt.subplots(facecolor=BG)
        ax.set_facecolor(BG)
        colors_map = {"baseline": CORAL, "compression_aware": MINT, "rgb_only": "#7C8798", "fft_only": "#4C7CFF"}
        for name, data in results["ablation"].items():
            accs = [data[l]["accuracy"] for l in levels if l in data]
            ax.plot(levels, accs, marker="o", label=name, color=colors_map.get(name, TEXT))
        ax.legend(facecolor=PANEL, edgecolor="#2A3140", labelcolor=TEXT)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_color("#2A3140")
        st.pyplot(fig)
        st.markdown(
            '<div class="caption-note">Baseline collapses under heavy compression '
            "(97.3%→70.3%). Compression-aware training holds at 89.0%. n=182 test videos.</div>",
            unsafe_allow_html=True,
        )

    waveform_divider()

    if results["df40_family"]:
        st.markdown("### Cross-dataset generalization (DF40 benchmark)")
        levels = ["original", "crf23", "crf28", "crf35"]
        fig2, ax2 = plt.subplots(facecolor=BG)
        ax2.set_facecolor(BG)
        for family in ["GAN", "Diffusion", "FS", "FR", "FE"]:
            fdrs = [results["df40_family"][l][family]["mean_fake_detection_rate"]
                    for l in levels if family in results["df40_family"].get(l, {})]
            if fdrs:
                ax2.plot(levels[:len(fdrs)], fdrs, marker="o", label=family)
        ax2.legend(facecolor=PANEL, edgecolor="#2A3140", labelcolor=TEXT)
        ax2.set_ylim(0, 1)
        ax2.tick_params(colors=MUTED)
        for spine in ax2.spines.values():
            spine.set_color("#2A3140")
        st.pyplot(fig2)
        st.markdown(
            '<div class="caption-note">31 unseen generation methods across 5 families, '
            "40 real + 40 fake samples each, n=40 per cell. RDDM (Diffusion) showed an "
            "outlier at original quality, traced to a necessary 1px dimension-crop fix — "
            "diluted across 5 Diffusion methods but worth noting if isolating RDDM alone. "
            "SD-2.1 is labeled 'GAN based' in the source paper's table despite being a "
            "diffusion model — reported as-is per the paper, footnoted here.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("DF40 results not found — add `df40_family_level_results.json` to the `results/` folder.")

    waveform_divider()
    with st.expander("About this model"):
        st.markdown("""
- **Architecture**: Dual-stream EfficientNet-B0 (RGB + FFT), 2560-D fused features,
  Dropout(0.3), Linear(2560→1), sigmoid at inference only.
- **Trained on**: Celeb-DF v2 (408 real / 795 fake videos), video-level split, seed=42.
- **Aggregation**: mean of per-frame probabilities, threshold at 0.5.
        """)
