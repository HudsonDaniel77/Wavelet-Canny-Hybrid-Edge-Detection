# 🩺 Wavelet-Canny Hybrid Framework: Execution Guide

This guide explains how to launch both the **Python Backend API** and the **React Clinical Dashboard**.

## 1. Prerequisites
Ensure you have the following installed:
- Python 3.10+
- Node.js (v18+) & Desktop Browser (Chrome/Edge recommended)

---

## 2. Setting Up the Backend (Python)
The backend runs a FastAPI server that executes the hybrid edge detection logic using OpenCV and PyWavelets.

1. Open a new terminal in the project root:
   ```powershell
   # Activate virtual environment if not already active
   "C:\Project\Wavelet"
   
   # Run the API server
   uvicorn src.api:app --reload
   ```
2. The server will start at: `http://127.0.0.1:8000`

---

## 3. Setting Up the Frontend (React + Vite)
The frontend provides a professional web dashboard for medical researchers.

1. Open a **second** terminal window and navigate to the frontend folder:
   ```powershell
   cd frontend
   
   # Install dependencies (only required first time)
   npm install
   
   # Start the development server
   npm run dev
   ```
2. Open your browser to: **`http://localhost:5173`**

---

## 4. How to Use the Dashboard
1.  **Upload Images:** Drag and drop or browse for medical scans (JPG, PNG).
2.  **Execute Pipeline:** Click the black **"Execute Pipeline"** button.
3.  **Analyze Results:** 
    - The **Hybrid Fusion** result is displayed at the top.
    - **Quantitative Metrics** (MSE, SNR, EPI, SSIM) are automatically calculated.
    - Click any image to open the **Clinical Inspection Zoom** (Lightbox).
4.  **Export:** Use the **"Export Results (ZIP)"** button to save all processed layers and metrics locally.

---

## 🩺 Troubleshooting
- **Is the Backend Running?** Ensure the Python terminal shows "Uvicorn running...".
- **Styling Missing?** Ensure you are using the latest version of the code where `postcss.config.js` and `index.css` are configured for Tailwind v4.
- **Port Conflicts:** If `5173` or `8000` are in use, they will try the next available port automatically.
