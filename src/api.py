from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import time

import os
import sys

# Add the current directory to sys.path so 'main' can be imported whether run via python or uvicorn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import functions from main.py
from main import (
    wavelet_decomposition, 
    adaptive_fusion, 
    canny_detection, 
    hybrid_fusion,
    compute_epi,
    compute_snr,
    sobel_detection
)
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

app = FastAPI(title="Biomedical Image Edge Detection API")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(img):
    """Encodes an OpenCV image to base64 string for frontend display"""
    # Ensure it's scaled 0-255 uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    print(f"Received request: {file.filename}")
    start_time = time.perf_counter()
    
    try:
        # Read image from request
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img_gray is None:
            print("Error: Invalid image format")
            return {"error": "Invalid image format uploaded. Please provide a valid JPG or PNG."}
            
        print(f"Processing image shape: {img_gray.shape}")
        
        # Preprocessing (Gaussian Denoising)
        denoised = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        # Run the Pipeline
        wavelet_levels = wavelet_decomposition(denoised)
        fused_wavelet = adaptive_fusion(wavelet_levels)
        canny_res_float = canny_detection(denoised)
        hybrid_res = hybrid_fusion(denoised, fused_wavelet, canny_res_float)
        
        # Calculate quantitative baselines
        ref_structural = sobel_detection(denoised)
        
        # Extract Base64 encodings
        results = {
            "images": {
                "original": image_to_base64(denoised),
                "wavelet_l1": image_to_base64(wavelet_levels[1]),
                "wavelet_l2": image_to_base64(wavelet_levels[2]),
                "wavelet_l3": image_to_base64(wavelet_levels[3]),
                "fused_wavelet": image_to_base64(fused_wavelet),
                "canny": image_to_base64(canny_res_float),
                "hybrid": image_to_base64(hybrid_res),
            }
        }
        
        # Compute Metrics
        run_mse = float(mse(ref_structural, hybrid_res))
        run_snr = float(compute_snr(ref_structural, hybrid_res))
        run_epi = float(compute_epi(ref_structural, hybrid_res))
        
        win_size = max(3, min(7, min(hybrid_res.shape)-1) | 1)
        try:
            run_ssim = float(ssim(ref_structural, hybrid_res, data_range=255, win_size=win_size))
        except:
            run_ssim = float(ssim(ref_structural, hybrid_res, data_range=255))

        process_time = time.perf_counter() - start_time
        
        results["metrics"] = {
            "MSE": run_mse,
            "SNR": run_snr,
            "EPI": run_epi,
            "SSIM": run_ssim,
            "runtime": process_time
        }
        print(f"Process complete in {process_time:.4f}s")
        return results

    except Exception as e:
        import traceback
        error_msg = f"Internal Server Error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return {"error": error_msg}

if __name__ == "__main__":
    import uvicorn
    # Use 127.0.0.1 explicitly to match frontend request
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
