# 🔬 Wavelet--Canny Hybrid Edge Detection for Biomedical Images

## 📌 Project Title

**A Quantitatively Evaluated Multi-level Wavelet--Canny Hybrid Edge
Fusion Framework for Biomedical Images**

------------------------------------------------------------------------

## 📖 Overview

Edge detection is a fundamental step in biomedical image analysis, used
in applications such as medical diagnosis, image segmentation, and
computer-aided analysis. However, traditional edge detection methods
(such as Canny) often struggle in noisy medical images and may miss weak
or fine edges.

This project proposes a **hybrid edge detection framework** that
combines:

-   **Multi-level Wavelet Transform** (for multi-scale edge analysis)\
-   **Adaptive weighting-based wavelet edge fusion** (minor novelty)\
-   **Canny edge detection** (for precise edge localization)\
-   **Noise-aware hybrid fusion strategy** (major novelty)\
-   **Quantitative performance evaluation** using MSE, SNR, and Edge
    Preservation Index (EPI)

The goal is to obtain a **more accurate, robust, and noise-resistant
edge map** for biomedical images.

------------------------------------------------------------------------

## 🎯 Objectives

This project aims to:

1.  Preprocess biomedical images (grayscale conversion + noise
    reduction).\
2.  Perform multi-level wavelet decomposition (Level 1, 2, and 3).\
3.  Extract edges at each wavelet level.\
4.  Fuse multi-level wavelet edges using an **adaptive weighting
    scheme**.\
5.  Apply Canny edge detection for high-precision edge extraction.\
6.  Design a **noise-aware hybrid fusion strategy** combining wavelet
    and Canny edges.\
7.  Generate a final enhanced edge map.\
8.  Evaluate performance using:
    -   Mean Squared Error (MSE)
    -   Signal-to-Noise Ratio (SNR)
    -   Edge Preservation Index (EPI)\
9.  Compare:
    -   Wavelet-only edges\
    -   Canny-only edges\
    -   Proposed hybrid method

------------------------------------------------------------------------

## 🧠 Methodology (Pipeline)

### 🔹 Step 1 --- Biomedical Image Input

A biomedical image (MRI, CT, X-ray, or retinal image) is taken as input.

### 🔹 Step 2 --- Preprocessing

-   Convert to grayscale\
-   Apply noise reduction (Gaussian/Median filtering)

### 🔹 Step 3 --- Multi-level Wavelet Decomposition

Apply Discrete Wavelet Transform (DWT) at three levels: - Level 1 → fine
details\
- Level 2 → medium structures\
- Level 3 → coarse boundaries

### 🔹 Step 4 --- Edge Extraction at Each Level

Generate separate edge maps from each wavelet level.

### 🔹 Step 5 --- Adaptive Multi-level Wavelet Edge Fusion (Minor Novelty)

Fuse wavelet edges using adaptive weights:

    E_fused = w1*L1 + w2*L2 + w3*L3

where: - L1, L2, L3 are edge maps from each level\
- w1, w2, w3 are adaptive weights based on edge strength

### 🔹 Step 6 --- Canny Edge Detection

Apply Canny edge detection to obtain sharp and well-localized edges.

### 🔹 Step 7 --- Noise-Aware Hybrid Fusion (Major Novelty)

Combine wavelet and Canny edges using a spatially adaptive rule:

    E_final(x,y) = α(x,y) * E_wavelet + β(x,y) * E_canny

-   In noisy regions → trust wavelet more (higher α)\
-   In clean regions → trust Canny more (higher β)

### 🔹 Step 8 --- Quantitative Evaluation

Compare performance using: - **MSE** (Lower is better) - **SNR** (Higher
is better) - **Edge Preservation Index (EPI)** (Higher is better)

------------------------------------------------------------------------

## 📁 Project Structure

    Wavelet-Canny-Hybrid-Edge-Detection/
    │── data/
    │   └── biomedical_image.jpg
    │
    │── src/
    │   └── main.py
    │
    │── results/
    │   ├── wavelet_edges.png
    │   ├── canny_edges.png
    │   └── hybrid_edges.png
    │
    │── docs/
    │   └── project_report.pdf
    │
    │── README.md

------------------------------------------------------------------------

## 🛠️ Requirements

Install required Python libraries:

``` bash
pip install numpy opencv-python matplotlib pywavelets
```

------------------------------------------------------------------------

## ▶️ How to Run the Project

1.  Place your biomedical image inside the `data/` folder.\
2.  Run the script:

``` bash
python src/main.py
```

3.  Outputs will be saved in the `results/` folder:

-   `wavelet_edges.png`
-   `canny_edges.png`
-   `hybrid_edges.png`

------------------------------------------------------------------------

## 📊 Expected Results

The proposed hybrid method is expected to:

-   Detect more fine edges than Canny alone\
-   Be more robust to noise than traditional methods\
-   Preserve important biomedical structures better\
-   Achieve:
    -   Lower MSE\
    -   Higher SNR\
    -   Higher Edge Preservation Index

------------------------------------------------------------------------

## 🏥 Applications

This project can be used in:

-   Medical image analysis\
-   Brain tumor boundary detection in MRI\
-   Retinal blood vessel extraction\
-   Automated disease diagnosis\
-   Image segmentation in clinical systems

------------------------------------------------------------------------

## 📚 Future Work

Possible extensions include:

-   Using deep learning models (CNN, U-Net) for edge detection\
-   Applying this method to large medical datasets\
-   Real-time edge detection for medical imaging systems\
-   Integration with ML-based diagnostic pipelines

------------------------------------------------------------------------

## 👨‍💻 Author

**Guru Abijeth S**\
Register No: 3122245002025

------------------------------------------------------------------------

## 📜 License

This project is for academic and educational purposes.

## 🚀 Recent Updates (v2.0)

The pipeline has been upgraded with the following research-grade improvements:

1.  **Full Float Precision**: All intermediate edge maps are processed as floating-point (0-1) to avoid quantization errors.
2.  **Spatially Adaptive Fusion**: Wavelet levels are fused using **local energy** (Gaussian window) instead of global variance.
3.  **Noise-Aware Hybrid Fusion**: A **Bilateral Filter** based noise map dynamically adjusts the balance between Wavelet and Canny edges per pixel.
4.  **Structural Enhancement**: Added **Gradient Magnitude Preservation** and morphological refinement to ensure continuous, strong edges.

See `src/main.py` for implementation details.

## 🌟 Refinements (v2.1)

Further refinements were applied to improve edge precision:
1.  **Reduced Blur**: Coarse wavelet level (L3) influence is down-weighted to prevent edge thickening.
2.  **Micro-Edge Removal**: Weak background noise is suppressed using thresholding and morphological opening.
3.  **Edge Thinning**: A soft morphological thinning step ensures hybrid edges remain sharp and comparable to Canny edges.

## 🧠 Cortical Fold Preservation (v2.2)

To ensure fine biomedical details (like cortical folds) are not lost:
1.  **Soft Suppression**: Replaced hard thresholding with adaptive soft suppression (`> 0.02`) to keep faint but valid edges.
2.  **Gentle Closing**: Switched to **Elliptical Morphological Closing** to bridge gaps without erasing thin structures.
3.  **Gradient Support**: Boosted edges aligned with image gradients (`0.8 + 0.2 * grad`) for better structural fidelity.
