# VAJRA Deep Learning Meteorological Architecture & Live Data System

This directory houses the machine learning nowcasting specifications, PyTorch training scripts, checkpoint configurations, and live data API guides for the **VAJRA (Vajra Atmospheric Junction for Risk & Alerts)** system.

---

## 1. Machine Learning (ML) Architecture

### Where is the ML training?
The production ML training pipeline is built using **PyTorch** and runs on GPU servers (e.g., NVIDIA A100 / RTX 4090 / CUDA clusters):
- **Training Script:** [`server/ml/train_convgru.py`](./train_convgru.py)
- **Input Data Format:** Doppler Weather Radar (DWR) Level-II raw volume scans (HDF5 / NetCDF4 format) or Cartesian reflectivity grids ($Z$) sampled at 10-minute cadence.
- **Atmospheric Conditioning:** ERA5 / Open-Meteo sounding profiles (CAPE, Lifted Index, 0°C Freezing Level isotherm, NASA SRTM elevation).

### How is it training?
1. **Recurrent Unfolding:** A 2D Spatiotemporal Convolutional Gated Recurrent Unit (**ConvGRU**) takes 4 observed historical frames ($t - 30\text{m}, t - 20\text{m}, t - 10\text{m}, t = 0\text{m}$).
2. **Spatial Self-Attention:** A spatial attention mechanism calculates the correlation between distant cloud complexes and weights convective genesis hotspots where new thunderstorm cells are rapidly exploding.
3. **Loss Function:** **Balanced Mean Squared Error (B-MSE)**. Standard MSE causes neural networks to predict blurry, washed-out rain fields. B-MSE weights extreme reflectivity ($\ge 45\text{ dBZ}$) up to $12\times$ higher than background rain, forcing the network to accurately preserve severe cloudburst cores.
4. **Optimization:** AdamW optimizer with cosine annealing learning rate schedule and gradient clipping ($1.0$).

### Where are checkpoints storing?
- **Directory:** `server/ml/checkpoints/`
- **Saved Files:**
  - `convgru_epoch_01.pt`, `convgru_epoch_02.pt` ... (Epoch checkpoints with optimizer state & epoch loss)
  - `best_model.pt` (Best validation model state dictionary)
- **Exporting for Production Deployment:** The trained checkpoint is exported to **ONNX** (`convgru_nowcast.onnx`) or compiled with **TensorRT** for sub-100ms real-time inference.

### How is it predicting in the current dashboard?
In this preview dashboard:
- **Client-Side Simulation Engine:** Located at [`client/src/lib/mlNowcastingEngine.ts`](../../client/src/lib/mlNowcastingEngine.ts).
- **Formulas Used:**
  - **Marshall-Palmer Relation:** $Z = 200 \cdot R^{1.6} \iff R = \left(\frac{10^{Z/10}}{200}\right)^{0.625}$
  - **ConvGRU Recurrence:** Gate updates ($z_t, r_t$) simulated mathematically per cell.
  - **Spatial Attention Score:** Computed as $A_t = \text{Softmax}\left(\frac{Z - 30}{15}\right) \cdot \text{Intensity}$, focusing tracking on the highest-reflectivity cells.
  - **Extrapolation:** Provides $+15\text{m}$, $+30\text{m}$, $+45\text{m}$, and $+60\text{m}$ projected dBZ, trend classification (`intensifying` / `steady` / `decaying`), and cloudburst hazard warnings.

---

## 2. Live Data vs. Simulated Data Audit

| Feed / Event | Source | Live API URL | API Key Needed? | Current Dashboard Status |
|---|---|---|---|---|
| **Atmospheric Sounding (CAPE, LI, Freezing Level, Temp, Rain)** | [Open-Meteo](https://open-meteo.com) | `https://api.open-meteo.com/v1/forecast` | ❌ **No (Free)** | ✅ **LIVE** (Hook: `useWeatherForecast.ts`) |
| **Precipitation Radar Mosaic (10-min tiles)** | [RainViewer](https://rainviewer.com) | `https://api.rainviewer.com/public/weather-maps.json` | ❌ **No (Free)** | ✅ **LIVE** (Streamed every 10 min) |
| **Airport METAR (Wind shear & squalls)** | [AviationWeather.gov](https://aviationweather.gov) | `https://aviationweather.gov/api/data/metar` | ❌ **No (Free)** | ✅ **LIVE** (Hook: `useAirportMetar.ts`) |
| **Basemaps (Dark Vector & Satellite Hybrid)** | Google Maps JS API | `maps.googleapis.com` | ✅ **Yes (Active in `.env`)** | ✅ **LIVE** (`VITE_GOOGLE_MAPS_API_KEY`) |
| **1–3 km Nationwide Convective Mesh** | VAJRA Viewport Grid | In-browser dynamic generator | N/A | ✅ **DYNAMIC 1–3 KM VIEWPORT** |
| **Earthquakes (Disaster Feed)** | [USGS Earthquake API](https://earthquake.usgs.gov) | `https://earthquake.usgs.gov/fdsnws/event/1/query` | ❌ **No (Free)** | 🔌 Available to link |
| **Tropical Cyclones (Bay of Bengal / Arabian Sea)** | [IMD Cyclone Bulletin / JTWC](https://mausam.imd.gov.in) | IMD RSS / JTWC GeoJSON | ❌ **No (Free)** | 🔌 Available to link |
| **Floods & River Gauge Levels** | India-WRIS / Central Water Commission | `https://indiawris.gov.in` | ⚠️ **Govt Login Required** | 🔌 Requires institutional registration |

---

## 3. How to Run the PyTorch Training Script

To train the actual deep learning model on your machine:
```bash
# 1. Install PyTorch and dependencies
pip install torch torchvision torchaudio numpy scipy

# 2. Run training (CPU or CUDA GPU)
python server/ml/train_convgru.py --epochs 20 --batch_size 8 --device cpu
# Or for NVIDIA GPU:
python server/ml/train_convgru.py --epochs 50 --batch_size 16 --device cuda
```
The script will automatically create `checkpoints/` and save `convgru_epoch_XX.pt` and `best_model.pt`.
