import streamlit as st
import cv2
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim

# ================= Page Setup =================
st.set_page_config(
    page_title="Adaptive Hybrid Denoising",
    page_icon="✨",
    layout="wide"
)

st.title("✨ Adaptive Hybrid Image Denoising")
st.caption("Adaptive Median + Decision Based Filtering with Weighted Fusion")

# ================= Utility Functions =================
def inject_sp_noise(img, level):
    noisy = img.copy()
    p = level / 100
    rnd = np.random.rand(*img.shape)

    noisy[rnd < p/2] = 0
    noisy[rnd > 1 - p/2] = 255
    return noisy


def adaptive_median(img, max_k):
    h, w = img.shape
    out = img.copy()

    for i in range(h):
        for j in range(w):
            k = 3
            while k <= max_k:
                r = k // 2
                win = img[max(i-r,0):i+r+1, max(j-r,0):j+r+1]
                zmin, zmax, zmed = win.min(), win.max(), np.median(win)

                if zmin < zmed < zmax:
                    if zmin < img[i,j] < zmax:
                        out[i,j] = img[i,j]
                    else:
                        out[i,j] = zmed
                    break
                k += 2
            else:
                out[i,j] = zmed
    return out.astype(np.uint8)


def decision_based_filter(img, max_k):
    h, w = img.shape
    out = img.copy()

    for i in range(h):
        for j in range(w):
            if img[i, j] not in [0, 255]:
                continue

            k = 3
            while k <= max_k:
                r = k // 2
                win = img[max(i-r,0):i+r+1, max(j-r,0):j+r+1]
                valid = win[(win > 0) & (win < 255)]

                if valid.size > 0:
                    out[i, j] = np.median(valid)
                    break
                k += 2
    return out.astype(np.uint8)


def weighted_fusion(img1, img2, alpha):
    return np.clip(alpha*img1 + (1-alpha)*img2, 0, 255).astype(np.uint8)


# ================= Sidebar =================
with st.sidebar:
    st.header("⚙️ Parameters")

    uploaded = st.file_uploader("Upload Grayscale Image", type=["png","jpg","jpeg"])
    noise = st.slider("Salt & Pepper Noise (%)", 0, 80, 30)
    max_window = st.slider("Max Window Size", 5, 31, 21, 2)
    alpha = st.slider("Fusion Weight (AMF)", 0.0, 1.0, 0.5)

    run = st.button("🚀 Run Denoising")

# ================= Processing =================
if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 0)

    noisy = inject_sp_noise(img, noise)

    if run:
        start = time.time()
        amf = adaptive_median(noisy, max_window)
        dbmf = decision_based_filter(noisy, max_window)
        hybrid = weighted_fusion(amf, dbmf, alpha)
        elapsed = time.time() - start

        psnr = cv2.PSNR(img, hybrid)
        ssim_val = ssim(img, hybrid, data_range=255)

        st.success(f"Processing done in {elapsed:.2f} sec")

        # ================= Tabs View =================
        tab1, tab2, tab3 = st.tabs(["📷 Images", "📊 Metrics", "🔍 Comparison"])

        with tab1:
            c1, c2, c3 = st.columns(3)
            c1.image(img, caption="Original", clamp=True)
            c2.image(noisy, caption="Noisy", clamp=True)
            c3.image(hybrid, caption="Hybrid Output", clamp=True)

        with tab2:
            st.metric("PSNR (dB)", f"{psnr:.2f}")
            st.metric("SSIM", f"{ssim_val:.4f}")

        with tab3:
            st.image(amf, caption="Adaptive Median Result", clamp=True)
            st.image(dbmf, caption="Decision Based Result", clamp=True)

else:
    st.info("⬅️ Upload an image from sidebar to start")