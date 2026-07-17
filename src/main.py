import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim
import os
import glob
import time
import pandas as pd
from scipy import stats

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_preprocess(image_path):
    """
    Step 1: Load and Preprocess Image
    - Read image in grayscale
    - Apply Gaussian denoising
    """
    print(f"Loading image from {image_path}...")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Apply Gaussian denoising
    # Using a 5x5 kernel and standard deviation of 0
    denoised_img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img, denoised_img

def wavelet_decomposition(image, wavelet='haar', level=3):
    """
    Step 2: Multi-level Wavelet Decomposition
    - Perform DWT at 3 levels
    - Extract edge maps from detail coefficients
    - RETURNS: Dictionary of edge maps in FLOAT format (0-1 range)
    """
    print(f"Performing Multi-level Wavelet Decomposition (Wavelet: {wavelet}, Level: {level})...")
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    
    # coeffs[0] is approximation (LL)
    # details = [ (LH, HL, HH)_level_n, ..., (LH, HL, HH)_level_1 ]
    details = coeffs[1:] 
    
    # Map levels: Index 0 -> Level 3, Index 1 -> Level 2, Index 2 -> Level 1
    # We want to access them by level number 1, 2, 3
    # details array is ordered from Coarse (Level N) to Fine (Level 1)
    
    # If level=3:
    # details[0] = Level 3 (Coarsest)
    # details[1] = Level 2
    # details[2] = Level 1 (Finest)
    
    level_indices = {1: -1, 2: -2, 3: -3}
    
    processed_edges = {}
    
    for lvl in [1, 2, 3]:
        if lvl > level: continue
        
        idx = level_indices[lvl]
        (cH, cV, cD) = details[idx]
        
        # FIX: Wavelet outputs contain excessive texture
        # Calculate Noise variance mathematically (Median Absolute Deviation of cD)
        # to soft-threshold wavelets and remove excessive texture
        sigma = np.median(np.abs(cD)) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(max(1, cD.size)))
        
        # Soft thresholding
        cH = pywt.threshold(cH, value=uthresh, mode='soft')
        cV = pywt.threshold(cV, value=uthresh, mode='soft')
        cD = pywt.threshold(cD, value=uthresh, mode='soft')
        
        # Combine details to get edge map for this level (magnitude)
        edge_map = np.sqrt(cH**2 + cV**2 + cD**2)
        
        # Resize to original image size for fusion using Linear interpolation
        edge_map_resized = cv2.resize(edge_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-1 float range for fusion (Improvement 1)
        # Avoid division by zero
        mx = np.max(edge_map_resized)
        if mx > 0:
            edge_map_norm = edge_map_resized / mx
        else:
            edge_map_norm = edge_map_resized
            
        processed_edges[lvl] = edge_map_norm
        
    return processed_edges

def adaptive_fusion(edge_maps):
    """
    Step 3: Spatially Adaptive Multi-level Wavelet Fusion (Improvement 2)
    - Compute local energy using GaussianBlur on squared edges
    - Compute spatially adaptive weights
    - Fuse per pixel: E = w1*L1 + w2*L2 + w3*L3
    - Refinement: Down-weight L3 to reduce blur
    """
    print("Performing Spatially Adaptive Wavelet Fusion...")
    
    L1 = edge_maps[1]
    L2 = edge_maps[2]
    L3 = edge_maps[3]
    
    # Compute Local Energy (7x7 window)
    # Energy = GaussianBlur(Edge^2)
    E1 = cv2.GaussianBlur(L1**2, (7, 7), 0)
    E2 = cv2.GaussianBlur(L2**2, (7, 7), 0)
    E3 = cv2.GaussianBlur(L3**2, (7, 7), 0)
    
    # Refinement Part 2: Down-weight L3 (coarse level) to reduce over-blur
    w3_factor = 0.5
    E3 = E3 * w3_factor
    
    # Compute Total Energy per pixel
    E_total = E1 + E2 + E3 + 1e-6
    
    # Compute Adaptive Weights per pixel
    w1 = E1 / E_total
    w2 = E2 / E_total
    w3 = E3 / E_total
    
    # Pixel-wise Fusion
    fused = w1 * L1 + w2 * L2 + w3 * L3
    
    # Normalize result to 0-1
    mx = np.max(fused)
    if mx > 0:
        fused = fused / mx
        
    return fused

def canny_detection(image):
    """
    Step 4: Canny Edge Detection
    - Returns float format (0-1) for consistency
    """
    print("Performing Canny Edge Detection...")
    
    # FIX: Canny output is overly dense and detects background noise
    # Use Otsu's threshold to find a proper foreground/background separation
    # instead of a purely median approach which zeroes out on black backgrounds
    high_thresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower = int(0.5 * high_thresh)
    upper = int(high_thresh)
    
    edges = cv2.Canny(image, lower, upper)
    
    # Convert to float 0-1
    return edges.astype(np.float64) / 255.0

def hybrid_fusion(image, wavelet_edges, canny_edges):
    """
    Step 5: Noise-aware Hybrid Fusion (Improvement 3 & 4)
    - Noise map using Bilateral Filter difference
    - Spatially adaptive alpha/beta
    - Gradient Magnitude Preservation (Refined)
    - Refinement: Gentle Closing & Soft Suppression (Preserves Cortical Folds)
    """
    print("Performing Improved Hybrid Fusion (Refined v2.2)...")
    
    # Create background mask to clear non-structural edges in empty regions
    _, mask_bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mask = cv2.dilate(mask_bin, np.ones((9,9), np.uint8)).astype(np.float64) / 255.0
    
    # --- Improvement 3: Noise-aware weights ---
    # Estimate noise using Bilateral Filter
    smooth = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Noise map = |Intermediate - Smooth|
    noise_map_raw = np.abs(image.astype(np.float64) - smooth.astype(np.float64))
    
    # FIX: Noise map incorrectly classifies structural regions as noise
    # Compute gradients on original image (Sobel)
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    # Normalize gradient magnitude to 0-1
    gm_max = np.max(grad_mag)
    if gm_max > 0:
        grad_norm = grad_mag / gm_max
    else:
        grad_norm = grad_mag
        
    # Penalize noise map in areas with strong gradients (where true structures exist)
    noise_map_true = noise_map_raw * (1.0 - grad_norm)
    
    # Normalize true noise map to 0-1
    nm_min, nm_max = np.min(noise_map_true), np.max(noise_map_true)
    if nm_max - nm_min > 0:
        noise_norm = (noise_map_true - nm_min) / (nm_max - nm_min)
    else:
        noise_norm = noise_map_true
        
    # Adaptive weights: Fix Fusion weight scaling
    # High noise -> High alpha (trust Wavelet)
    # Low noise -> High beta (trust Canny)
    # We drop the base alpha to 0.2 to rely more on Canny's spatial precision in clean areas
    alpha = 0.2 + 0.5 * noise_norm
    beta = 1.0 - alpha
    
    # Initial Hybrid Fusion
    E_hybrid = alpha * wavelet_edges + beta * canny_edges
    
    # --- Improvement 4: Gradient Magnitude Preservation (Refined) ---
    # Enhance strong structural edges
    enhancement_factor = 0.5 + 0.5 * grad_norm
    E_enhanced = E_hybrid * enhancement_factor
    
    # Mask out background regions
    E_enhanced = E_enhanced * bg_mask
    
    # --- Refinement Part 1: Edge Suppression Strategy ---
    # Filter out weak noise by increasing the threshold (from 0.02)
    # This keeps faint structures but completely eliminates fuzzy micro-noise
    E_suppressed = E_enhanced * (E_enhanced > 0.15)
    
    # --- Refinement Part 2: Morphological refinement strength ---
    # Scale to 0-1 appropriately
    mx = np.max(E_suppressed)
    if mx > 0:
        E_norm_temp = E_suppressed / mx
    else:
        E_norm_temp = E_suppressed
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    
    # Add an OPENing operation to aggressively clean up single-pixel speckles
    # before running the CLOSE operation to bridge gaps in ridges
    E_opened = cv2.morphologyEx(E_norm_temp, cv2.MORPH_OPEN, kernel)
    E_closed = cv2.morphologyEx(E_opened, cv2.MORPH_CLOSE, kernel)
    
    # --- Refinement Part 3: Gentle Edge Thinning (Safe Version) ---
    # Do NOT erode strongly. Blend lightly.
    eroded = cv2.erode(E_closed, kernel, iterations=1)
    E_thin = 0.9 * E_closed + 0.1 * eroded
    
    # Ensure final edges strictly obey background mask constraints
    E_thin = E_thin * bg_mask
    
    # Final normalization to 0-255 uint8 only at the very end
    E_final_norm = cv2.normalize(E_thin, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return E_final_norm

def compute_epi(reference, evaluated):
    ref_f = reference.astype(np.float64)
    eval_f = evaluated.astype(np.float64)
    
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    
    delta_R = cv2.filter2D(ref_f, -1, laplacian_kernel)
    delta_E = cv2.filter2D(eval_f, -1, laplacian_kernel)
    
    mean_R = np.mean(delta_R)
    mean_E = np.mean(delta_E)
    
    numerator = np.sum((delta_R - mean_R) * (delta_E - mean_E))
    denominator = np.sqrt(np.sum((delta_R - mean_R)**2) * np.sum((delta_E - mean_E)**2))
    
    if denominator == 0: return 0.0
    return np.clip(numerator / denominator, 0.0, 1.0)

def compute_snr(reference, evaluated):
    ref_norm = reference.astype(np.float64) / 255.0
    eval_norm = evaluated.astype(np.float64) / 255.0
    signal_power = np.sum(ref_norm ** 2)
    noise_power = np.sum((ref_norm - eval_norm) ** 2)
    if noise_power == 0: return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def sobel_detection(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.normalize(np.sqrt(gx**2 + gy**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def prewitt_detection(image):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    gx = cv2.filter2D(image, cv2.CV_64F, kernelx)
    gy = cv2.filter2D(image, cv2.CV_64F, kernely)
    return cv2.normalize(np.sqrt(gx**2 + gy**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def measure_runtime(func, *args):
    start_time = time.perf_counter()
    result = func(*args)
    return result, (time.perf_counter() - start_time)

def evaluate_dataset(dataset_dir, results_dir):
    image_paths = glob.glob(os.path.join(dataset_dir, '*.png')) + glob.glob(os.path.join(dataset_dir, '*.jpg'))
    if not image_paths: return
    
    results = []
    time_results = {'Sobel': [], 'Prewitt': [], 'Canny': [], 'Wavelet': [], 'Hybrid': []}
    
    print(f"\\n--- Evaluating dataset with {len(image_paths)} images ---")
    
    for path in image_paths:
        img_name = os.path.basename(path)
        img, denoised = load_and_preprocess(path)
        reference_structural = sobel_detection(denoised)
        
        sobel_res, t_sobel = measure_runtime(sobel_detection, denoised)
        prewitt_res, t_prewitt = measure_runtime(prewitt_detection, denoised)
        canny_res_float, t_canny = measure_runtime(canny_detection, denoised)
        
        t_start = time.perf_counter()
        wavelet_levels = wavelet_decomposition(denoised)
        fused_wavelet = adaptive_fusion(wavelet_levels)
        wavelet_res = (fused_wavelet * 255).astype(np.uint8)
        t_wavelet = time.perf_counter() - t_start
        
        t_start = time.perf_counter()
        hybrid_res = hybrid_fusion(denoised, fused_wavelet, canny_res_float)
        t_hybrid = time.perf_counter() - t_start + t_wavelet + t_canny
        
        canny_res = (canny_res_float * 255).astype(np.uint8)
        
        time_results['Sobel'].append(t_sobel)
        time_results['Prewitt'].append(t_prewitt)
        time_results['Canny'].append(t_canny)
        time_results['Wavelet'].append(t_wavelet)
        time_results['Hybrid'].append(t_hybrid)
        
        methods = {'Sobel': sobel_res, 'Prewitt': prewitt_res, 'Canny': canny_res, 'Wavelet': wavelet_res, 'Hybrid': hybrid_res}
        
        for m_name, m_img in methods.items():
            run_mse = mse(reference_structural, m_img)
            run_snr = compute_snr(reference_structural, m_img)
            run_epi = compute_epi(reference_structural, m_img)
            win_size = max(3, min(7, min(m_img.shape)-1) | 1)
            try:
                run_ssim = ssim(reference_structural, m_img, data_range=255, win_size=win_size)
            except:
                run_ssim = ssim(reference_structural, m_img, data_range=255)
            
            results.append({'Image': img_name, 'Method': m_name, 'MSE': run_mse, 'SNR': run_snr, 'EPI': run_epi, 'SSIM': run_ssim})
            
    df = pd.DataFrame(results)
    summary = df.groupby('Method').agg({'MSE': ['mean', 'std'], 'SNR': ['mean', 'std'], 'EPI': ['mean', 'std'], 'SSIM': ['mean', 'std']})
    
    print("\\n--- Quantitative Summary (Across Dataset) ---")
    formatted_summary = pd.DataFrame()
    for col in ['MSE', 'SNR', 'EPI', 'SSIM']:
        formatted_summary[col] = summary[(col, 'mean')].map('{:.3f}'.format) + " \u00B1 " + summary[(col, 'std')].map('{:.3f}'.format)
    print(formatted_summary)
    
    formatted_summary.to_csv(os.path.join(results_dir, 'dataset_evaluation_summary.csv'))
    
    print("\\n--- Statistical Significance (Paired T-Test) ---")
    hybrid_epi = df[df['Method'] == 'Hybrid']['EPI'].values
    canny_epi = df[df['Method'] == 'Canny']['EPI'].values
    wavelet_epi = df[df['Method'] == 'Wavelet']['EPI'].values
    if len(hybrid_epi) > 1:
        t_hc, p_hc = stats.ttest_rel(hybrid_epi, canny_epi)
        t_hw, p_hw = stats.ttest_rel(hybrid_epi, wavelet_epi)
        print(f"Hybrid vs Canny (EPI):   t-stat={t_hc:.3f}, p-value={p_hc:.5e} -> {'Significant' if p_hc < 0.05 else 'Not Significant'}")
        print(f"Hybrid vs Wavelet (EPI): t-stat={t_hw:.3f}, p-value={p_hw:.5e} -> {'Significant' if p_hw < 0.05 else 'Not Significant'}")
        
    print("\\n--- Average Runtime Per Image ---")
    for k, v in time_results.items(): print(f"{k}: {np.mean(v)*1000:.2f} ms")

def quantitative_evaluation(original, wavelet_e, canny_e, hybrid_e):
    """
    Step 6: Quantitative Evaluation (Improvement 7)
    - Compare Hybrid vs Canny and Hybrid vs Wavelet
    - Compute MSE, PSNR, and SSIM
    """
    print("\n--- Quantitative Evaluation ---")
    
    # Ensure inputs are uint8 for consistent metric calculation if they aren't already
    def to_uint8(img):
        if img.dtype != np.uint8:
            return (img * 255).astype(np.uint8)
        return img

    w_u8 = to_uint8(wavelet_e)
    c_u8 = to_uint8(canny_e)
    h_u8 = to_uint8(hybrid_e)
    
    # SSIM requires 'win_size' to be smaller than image dimensions (default 7 is fine for >7x7 images)
    # We compare the Hybrid result primarily against the component methods to see information retention
    
    metrics = {}
    
    # Compare Hybrid against Wavelet output
    m_hw = mse(w_u8, h_u8)
    p_hw = psnr(w_u8, h_u8)
    s_hw = ssim(w_u8, h_u8)
    print(f"Hybrid vs Wavelet -> MSE: {m_hw:.2f}, PSNR: {p_hw:.2f}, SSIM: {s_hw:.4f}")
    
    # Compare Hybrid against Canny output
    m_hc = mse(c_u8, h_u8)
    p_hc = psnr(c_u8, h_u8)
    s_hc = ssim(c_u8, h_u8)
    print(f"Hybrid vs Canny   -> MSE: {m_hc:.2f}, PSNR: {p_hc:.2f}, SSIM: {s_hc:.4f}")
    
    return metrics

def visualize_results(original, L1, L2, L3, fused_wavelet, canny, hybrid):
    """
    Step 7: Visualization
    """
    plt.figure(figsize=(15, 10))
    
    images = [original, L1, L2, L3, fused_wavelet, canny, hybrid]
    titles = ['Original/Denoised', 'Wavelet L1', 'Wavelet L2', 'Wavelet L3', 
              'Fused Wavelet', 'Canny Edges', 'Hybrid Edges']
    
    for i in range(7):
        plt.subplot(3, 3, i+1)
        # Handle float vs uint8 for display
        img_disp = images[i]
        
        plt.imshow(img_disp, cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig('results/visualization_summary.png')
    # plt.show() 

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    
    ensure_dir(results_dir)
    ensure_dir(data_dir)
    
    image_paths = glob.glob(os.path.join(data_dir, '*.jpg')) + glob.glob(os.path.join(data_dir, '*.png'))
    
    if len(image_paths) == 0:
        print("Warning: No images found.")
        print("Generating 5 synthetic biomedical images for dataset evaluation representation...")
        for i in range(5):
            syn_path = os.path.join(data_dir, f'synthetic_{i+1}.jpg')
            synthetic_img = np.zeros((512, 512), dtype=np.uint8)
            cv2.circle(synthetic_img, (256, 256), 100 + i*10, 200, -1)
            cv2.rectangle(synthetic_img, (50+i*5, 50-i*5), (150+i*5, 150-i*5), 150, -1)
            noise = np.random.normal(0, 20 + i*5, synthetic_img.shape).astype(np.uint8)
            synthetic_img = cv2.add(synthetic_img, noise)
            cv2.imwrite(syn_path, synthetic_img)
        print("Synthetic images created.")
        image_paths = glob.glob(os.path.join(data_dir, '*.jpg'))
        
    image_path = image_paths[0]

    print("\n==================================")
    print(" 1. Single Image Detailed Run")
    print("==================================")
    original, denoised = load_and_preprocess(image_path)
    cv2.imwrite(os.path.join(results_dir, 'preprocessed.png'), denoised)
    
    wavelet_levels = wavelet_decomposition(denoised)
    cv2.imwrite(os.path.join(results_dir, 'wavelet_L1.png'), (wavelet_levels[1] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(results_dir, 'wavelet_L2.png'), (wavelet_levels[2] * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(results_dir, 'wavelet_L3.png'), (wavelet_levels[3] * 255).astype(np.uint8))
    
    fused_wavelet = adaptive_fusion(wavelet_levels)
    cv2.imwrite(os.path.join(results_dir, 'wavelet_fused.png'), (fused_wavelet * 255).astype(np.uint8))
    
    canny_edges = canny_detection(denoised)
    cv2.imwrite(os.path.join(results_dir, 'canny_edges.png'), (canny_edges * 255).astype(np.uint8))
    
    hybrid_edges = hybrid_fusion(denoised, fused_wavelet, canny_edges)
    cv2.imwrite(os.path.join(results_dir, 'hybrid_edges.png'), hybrid_edges)
    
    quantitative_evaluation(denoised, fused_wavelet, canny_edges, hybrid_edges)
    
    visualize_results(denoised, 
                      wavelet_levels[1], wavelet_levels[2], wavelet_levels[3], 
                      fused_wavelet, canny_edges, hybrid_edges)
    
    print("\n==================================")
    print(" 2. Dataset Evaluation Run")
    print("==================================")
    evaluate_dataset(data_dir, results_dir)
    
    print("\nProcessing complete. All results saved in 'results/' folder.")

if __name__ == "__main__":
    main()
