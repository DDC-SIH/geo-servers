

# 🌍 Raster Layer Stacker & Time-Series Animator

A powerful Flask API for stacking, rescaling, and animating **Cloud-Optimized GeoTIFFs (COGs)** via **TiTiler URLs** or **local files**. Supports transparency, zIndex ordering, band selection, and temporal animations.

---

## 🚀 Features

### ✅ Raster Layer Stacking
- Stack multiple COGs by zIndex
- Apply per-layer transparency
- Alpha-blending with RGBA support

### ✅ TiTiler + Local Fallback
- Automatically downloads and validates TiTiler URLs
- Falls back to local files if TiTiler is unreachable

### ✅ Band Selection & Rescaling
- Select bands via `band_indices`
- Apply value `rescale=[min,max]` to stretch intensities

### ✅ Animation Mode
- Generate time-series animations as **GIFs**
- Supports fixed or variable bands over time
- Multithreaded frame processing

### ✅ Flexible Templated URLs
Use placeholders in `directURL`:
- `{DATE}` → `22MAR2025`
- `{TIME}` → `0915`  
Expanded automatically using `date_range` and `time_range`

### ✅ Output Formats
- `.tiff` (default)
- `.png`
- `.gif` via `?animation=yes`
- `.zip` via `?zip=yes` for bundling layers

---

## 📦 Installation

```bash
git clone https://github.com/your-repo/raster-stack-api.git
cd raster-stack-api
pip install -r requirements.txt
python app.py
```

---

## 🧪 Example `curl` Request

```bash
curl --location 'http://localhost:5000/stack-layers?animation=yes' \
--header 'Content-Type: application/json' \
--data '{
  "directURL": "http://localhost:8000/cog/bbox/72.02,15.75,100.76,34.22.tif?url=C:/data/3RIMG_{DATE}_{TIME}_L1C.cog.tif&rescale=0,1000&rescale=0,1000&rescale=0,1000",
  "date_range": ["2025-03-22", "2025-03-22"],
  "time_range": ["09:15", "09:15"],
  "transparency": [0.5, 0.8],
  "zIndex": [1000, 999],
  "band_indices": [
    [1, 2, 3],
    [1, 3, 4]
  ]
}' --output animation.gif
```

---

## 🛠️ API Endpoints

### `POST /stack-layers`

- **Input:** JSON configuration
- **Query Params:**
  - `animation=yes` → create GIF
  - `format=png` → return PNG
  - `zip=yes` → return ZIP with layers + output
- **Returns:** TIFF / PNG / GIF / ZIP

---

## 📤 JSON Config Structure

```json
{
  "directURL": "http://...{DATE}_{TIME}.tif?...",
  "date_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
  "time_range": ["HH:MM", "HH:MM"],
  "transparency": [0.5, 0.8],
  "zIndex": [1000, 999],
  "band_indices": [
    [1, 2, 3],
    [1, 3, 4]
  ]
}
```

---

## 🧠 Internals

- Uses `rasterio`, `imageio`, `concurrent.futures`
- URL parsing: `urllib`, `re`
- Handles malformed JSON with preprocessing
- Automatic cleanup of temp files

---

## 📁 Directory Structure

```
raster-stack-api/
├── app.py               # Main Flask app
├── README.md
├── requirements.txt
└── data/                # COG files (for local fallback)
```

