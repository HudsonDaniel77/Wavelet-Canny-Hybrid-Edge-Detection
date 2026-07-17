import { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Image as ImageIcon, CheckCircle, Activity, RefreshCw, Layers, Sliders, Maximize2, Download } from 'lucide-react';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';

// --- API Service ---
const api = axios.create({
  baseURL: 'http://127.0.0.1:8000',
});

// --- Components ---

// --- Image Card ---
const ImageCard = ({ title, src, onClick }) => (
  <div
    className="bg-white rounded-xl shadow-md border border-slate-300 overflow-hidden hover:shadow-xl transition-all cursor-pointer relative group flex flex-col"
    onClick={() => onClick(src, title)}
  >
    <div className="bg-slate-800 p-2.5 flex items-center justify-between">
      <h3 className="text-xs font-bold text-white uppercase tracking-wider">
        {title}
      </h3>
      <Maximize2 size={14} className="text-medical-400 group-hover:text-white transition" />
    </div>
    <div className="flex-1 bg-black overflow-hidden relative flex items-center justify-center p-1">
      <img
        src={src}
        alt={title}
        className="max-h-full max-w-full object-contain pointer-events-none group-hover:scale-105 transition duration-500"
      />
    </div>
  </div>
);

// --- Metrics Table ---
const MetricsTable = ({ metrics }) => (
  <div className="bg-white rounded-xl shadow-md border border-slate-300 overflow-hidden flex flex-col h-full">
    <div className="bg-medical-700 p-3 border-b border-medical-800">
      <h3 className="text-sm font-bold text-white flex items-center gap-2">
        <Activity size={16} /> Diagnostic Performance Metrics
      </h3>
    </div>
    <div className="p-6 flex-1 flex flex-col justify-around">
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-slate-50 p-4 rounded-xl border-l-4 border-medical-500 shadow-sm">
          <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">MSE (↓)</p>
          <p className="text-2xl font-black text-slate-900">{metrics.MSE ? metrics.MSE.toFixed(2) : '0.00'}</p>
        </div>
        <div className="bg-slate-50 p-4 rounded-xl border-l-4 border-medical-500 shadow-sm">
          <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">PSNR (↑)</p>
          <p className="text-2xl font-black text-slate-900">{metrics.SNR ? metrics.SNR.toFixed(2) : '0.0'} <span className="text-sm font-bold">dB</span></p>
        </div>
        <div className="bg-slate-50 p-4 rounded-xl border-l-4 border-medical-500 shadow-sm">
          <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">EPI (↑)</p>
          <p className="text-2xl font-black text-slate-900">{metrics.EPI ? metrics.EPI.toFixed(3) : '0.000'}</p>
        </div>
        <div className="bg-slate-50 p-4 rounded-xl border-l-4 border-medical-500 shadow-sm">
          <p className="text-[10px] text-slate-500 font-bold uppercase mb-1">SSIM (↑)</p>
          <p className="text-2xl font-black text-slate-900">{metrics.SSIM ? metrics.SSIM.toFixed(3) : '0.000'}</p>
        </div>
      </div>
    </div>
  </div>
);

const ResultPanel = ({ result, filename, onImageClick }) => {
  return (
    <div className="bg-white rounded-2xl shadow-lg border border-slate-300 overflow-hidden mb-12 fade-in">
      {/* Header */}
      <div className="bg-slate-900 p-5 flex justify-between items-center">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-medical-500 flex items-center justify-center text-white shadow-inner">
            <ImageIcon size={24} />
          </div>
          <div>
            <h2 className="text-xl font-black text-white">{filename || 'Processed Biomedical Scan'}</h2>
            <p className="text-xs text-medical-300 font-bold uppercase tracking-widest">Hybrid Fusion Success</p>
          </div>
        </div>
      </div>

      {/* Primary Results Display */}
      <div className="p-8 bg-slate-100">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 mb-8">
          {/* Main Hybrid Output */}
          <div className="lg:col-span-8">
            <div className="bg-white rounded-2xl shadow-inner border border-slate-300 p-2 h-full">
              <div className="relative aspect-[16/9] bg-black rounded-xl overflow-hidden group cursor-pointer" onClick={() => onImageClick(`data:image/png;base64,${result.images.hybrid}`, "Final Hybrid Output")}>
                <img src={`data:image/png;base64,${result.images.hybrid}`} className="w-full h-full object-contain" alt="Final Hybrid" />
                <div className="absolute top-4 left-4 bg-medical-600 text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg">FINAL HYBRID FUSION</div>
                <div className="absolute inset-0 bg-white/0 group-hover:bg-white/10 transition flex items-center justify-center">
                  <Maximize2 size={48} className="text-white opacity-0 group-hover:opacity-100 transition duration-300 drop-shadow-md" />
                </div>
              </div>
            </div>
          </div>

          {/* Original Image Sidecar */}
          <div className="lg:col-span-4 flex flex-col gap-4">
            <div className="flex-1 bg-white rounded-2xl shadow-inner border border-slate-300 p-2">
              <div className="relative h-full bg-black rounded-xl overflow-hidden group cursor-pointer" onClick={() => onImageClick(`data:image/png;base64,${result.images.original}`, "Input (Denoised)")}>
                <img src={`data:image/png;base64,${result.images.original}`} className="w-full h-full object-contain" alt="Original" />
                <div className="absolute top-3 left-3 bg-slate-800 text-white px-2 py-1 rounded text-[10px] font-bold">SOURCE (DENOISED)</div>
              </div>
            </div>
            <MetricsTable metrics={result.metrics} />
          </div>
        </div>

        {/* Multi-level Breakdown */}
        <div className="pt-8 border-t border-slate-200">
          <h4 className="text-sm font-black text-slate-800 uppercase tracking-widest mb-6 flex items-center gap-2">
            <Layers size={18} className="text-medical-600" /> Multi-scale Decomposition Analysis
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            <ImageCard title="Wavelet L1 (Fine)" src={`data:image/png;base64,${result.images.wavelet_l1}`} onClick={onImageClick} />
            <ImageCard title="Wavelet L2 (Med)" src={`data:image/png;base64,${result.images.wavelet_l2}`} onClick={onImageClick} />
            <ImageCard title="Wavelet L3 (Coarse)" src={`data:image/png;base64,${result.images.wavelet_l3}`} onClick={onImageClick} />
            <ImageCard title="Fused Wavelet" src={`data:image/png;base64,${result.images.fused_wavelet}`} onClick={onImageClick} />
            <ImageCard title="Canny Edge" src={`data:image/png;base64,${result.images.canny}`} onClick={onImageClick} />
          </div>
        </div>
      </div>
    </div>
  );
};


// --- Main App ---
function App() {
  const [files, setFiles] = useState([]);
  const [previews, setPreviews] = useState([]);
  const [results, setResults] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [modalImage, setModalImage] = useState(null);

  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);

    // Validate MIME types
    const validFiles = selectedFiles.filter(f => f.type.startsWith('image/'));

    if (validFiles.length !== selectedFiles.length) {
      setError("Some files were skipped because they are not valid images.");
    } else {
      setError(null);
    }

    if (validFiles.length > 0) {
      setFiles(validFiles);

      // Generate object URLs for previews
      const urlPreviews = validFiles.map(file => ({
        url: URL.createObjectURL(file),
        name: file.name
      }));
      setPreviews(urlPreviews);
      setResults([]); // Clear previous results
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files) {
      fileInputRef.current.files = e.dataTransfer.files;
      handleFileChange({ target: { files: e.dataTransfer.files } });
    }
  };

  const processImages = async () => {
    if (files.length === 0) return;

    setIsProcessing(true);
    setError(null);
    setResults([]);

    try {
      const newResults = [];

      // Process sequentially to avoid memory leaks on heavy processing
      for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await api.post('/process', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });

        newResults.push({
          filename: file.name,
          data: response.data
        });

        // Update state progressively for UX
        setResults([...newResults]);
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.detail || "An error occurred during processing. Is the backend running?");
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = () => {
    setFiles([]);
    setPreviews([]);
    setResults([]);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const openModal = (src, title) => {
    setModalImage({ src, title });
  };

  const downloadAll = async () => {
    if (results.length === 0) return;
    const zip = new JSZip();

    results.forEach((res, i) => {
      const folder = zip.folder(res.filename || `Image_${i + 1}`);

      Object.keys(res.data.images).forEach(key => {
        folder.file(`${key}.png`, res.data.images[key], { base64: true });
      });

      // Save metrics as json
      folder.file('metrics.json', JSON.stringify(res.data.metrics, null, 2));
    });

    const content = await zip.generateAsync({ type: "blob" });
    saveAs(content, "hybrid_edge_detection_results.zip");
  };

  return (
    <div className="min-h-screen pb-20">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-tr from-medical-600 to-medical-400 p-2 rounded-lg text-white">
              <Layers size={22} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-800 tracking-tight leading-none">Hybrid Edge Framework</h1>
              <p className="text-[10px] uppercase font-bold text-medical-600 tracking-wider">Multi-level Wavelet–Canny Analysis</p>
            </div>
          </div>
          <div>
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-600 border border-emerald-200">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Ready
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">

        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-red-50 border border-red-200 text-red-700 font-medium flex items-center gap-3 fade-in">
            <span className="text-xl">⚠️</span> {error}
          </div>
        )}

        {/* Upload Zone */}
        {!isProcessing && results.length === 0 && (
          <div className="fade-in">
            <div className="text-center mb-8 max-w-2xl mx-auto">
              <h2 className="text-3xl font-bold text-slate-800 mb-3">Biomedical Image Analysis</h2>
              <p className="text-slate-500">Upload high-resolution scans for enhanced structural boundary preservation using our mathematically rigorous Wavelet-Canny hybrid architecture.</p>
            </div>

            <div
              className="border-2 border-dashed border-medical-200 bg-white rounded-2xl p-12 flex flex-col items-center justify-center cursor-pointer hover:border-medical-500 hover:bg-medical-50/50 transition duration-300 shadow-sm"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                multiple
                accept="image/*"
              />

              <div className="w-20 h-20 bg-medical-50 rounded-full flex items-center justify-center text-medical-600 mb-6 shadow-sm border border-medical-100">
                <Upload size={32} />
              </div>

              <h3 className="text-xl font-bold text-slate-700 mb-2">Drag & Drop Images</h3>
              <p className="text-slate-500 mb-6 text-center max-w-sm">Support for Single or Batch Uploads.<br />Formats: JPG, PNG, TIFF, DICOM (converted).</p>

              <button className="bg-slate-900 hover:bg-black text-white font-black py-3 px-8 rounded-xl shadow-lg hover:shadow-xl transition flex items-center gap-2 uppercase tracking-widest text-sm">
                <ImageIcon size={18} /> Browse Local Files
              </button>
            </div>
          </div>
        )}

        {/* Previews & Processing Zone */}
        {previews.length > 0 && results.length === 0 && (
          <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm fade-in">
            <div className="flex justify-between items-center mb-6 pb-4 border-b border-slate-100">
              <div>
                <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2"><ImageIcon size={20} className="text-medical-600" /> Batch Queue</h3>
                <p className="text-sm text-slate-500">{previews.length} target image(s) loaded.</p>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={handleReset}
                  className="flex items-center gap-2 px-4 py-2 text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-lg font-medium transition"
                  disabled={isProcessing}
                >
                  <RefreshCw size={16} /> Reset
                </button>
                <button
                  onClick={processImages}
                  disabled={isProcessing}
                  className="flex items-center gap-2 px-8 py-3 bg-slate-900 text-white rounded-xl hover:bg-black transition shadow-lg font-black uppercase tracking-widest disabled:opacity-70 disabled:cursor-not-allowed"
                >
                  {isProcessing ? (
                    <><Activity size={18} className="animate-spin" /> Computing...</>
                  ) : (
                    <><Sliders size={18} /> Execute Pipeline</>
                  )}
                </button>
              </div>
            </div>

            <div className={`grid grid-cols-2 md:grid-cols-4 gap-4 ${isProcessing ? 'opacity-50 pointer-events-none' : ''}`}>
              {previews.map((preview, index) => (
                <div key={index} className="relative rounded-xl overflow-hidden border border-slate-200 aspect-square group bg-slate-100">
                  <img src={preview.url} alt={preview.name} className="w-full h-full object-cover" />
                  <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-3 pt-8">
                    <p className="text-white text-xs truncate font-medium">{preview.name}</p>
                  </div>
                  {isProcessing && (
                    <div className="absolute inset-0 bg-medical-900/20 flex flex-col items-center justify-center backdrop-blur-sm">
                      <Activity size={32} className="text-white animate-spin mb-2" />
                      <span className="text-white font-bold text-sm tracking-wider uppercase">Processing</span>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {isProcessing && (
              <div className="mt-8">
                <div className="flex justify-between text-sm font-semibold text-slate-600 mb-2">
                  <span>Algorithmic Fusion in Progress</span>
                  <span>{results.length} / {files.length} Completed</span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div
                    className="bg-medical-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(results.length / files.length) * 100}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Results Render */}
        {results.length > 0 && (
          <div className="fade-in">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8 bg-medical-50 p-4 border border-medical-100 rounded-xl">
              <div className="flex items-center gap-3">
                <CheckCircle size={28} className="text-medical-600" />
                <div>
                  <h2 className="text-xl font-bold text-slate-800">Analysis Complete</h2>
                  <p className="text-sm text-slate-600">Processed {results.length} item(s) through hybrid pipeline.</p>
                </div>
              </div>
              <div className="flex gap-3">
                <button
                  onClick={downloadAll}
                  className="flex items-center justify-center flex-1 gap-2 px-4 py-2 bg-white text-medical-700 border border-medical-200 hover:bg-medical-100 rounded-lg font-bold shadow-sm transition"
                >
                  <Download size={18} /> Export Results (ZIP)
                </button>
                <button
                  onClick={handleReset}
                  className="flex justify-center items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-900 text-white rounded-lg font-bold shadow-sm transition"
                >
                  <RefreshCw size={18} /> New Analysis
                </button>
              </div>
            </div>

            <div className="space-y-6">
              {results.map((res, i) => (
                <ResultPanel
                  key={i}
                  result={res.data}
                  filename={res.filename}
                  onImageClick={openModal}
                />
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Lightbox Modal */}
      {modalImage && (
        <div
          className="fixed inset-0 z-50 bg-slate-900/90 backdrop-blur-sm flex items-center justify-center p-4 fade-in"
          onClick={() => setModalImage(null)}
        >
          <div className="bg-white rounded-2xl overflow-hidden max-w-5xl w-full max-h-[90vh] flex flex-col shadow-2xl" onClick={e => e.stopPropagation()}>
            <div className="p-4 border-b border-slate-100 flex justify-between items-center bg-slate-50">
              <h3 className="font-bold text-slate-800 text-lg">{modalImage.title}</h3>
              <button className="text-slate-400 hover:text-red-500 font-bold px-3 py-1 rounded" onClick={() => setModalImage(null)}>✕ Close</button>
            </div>
            <div className="flex-1 overflow-auto bg-slate-100 p-4 flex items-center justify-center min-h-[500px]">
              <img src={modalImage.src} alt={modalImage.title} className="max-w-full max-h-[75vh] object-contain shadow-sm border border-slate-200 bg-white" />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
