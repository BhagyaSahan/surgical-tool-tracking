# Real-Time Surgical Tool Tracking Using YOLOv8 and Active Visual Overlays for Edge Clinical Hardware

An end-to-end, lightweight computer vision framework designed to track laparoscopic surgical instruments in real time on standard CPU-based clinical edge hardware. 

Powered by a single-stage **YOLOv8** detector and an asynchronous **FastAPI** backend, the system delivers continuous **96.0% mAP@0.5** performance at **10–12 FPS on CPUs** (and **30+ FPS on GPUs**) using real-time dynamic bounding box overlays for spatial verification.

---

## 🌟 Key Features

* **Single-Stage Tracking Engine:** Fine-tuned YOLOv8 architecture trained on the m2cai16 surgical benchmark dataset across 7 instrument classes (*Grasper, Bipolar, Hook, Scissors, Clipper, Irrigator, Specimen Bag*).
* **Edge Hardware Optimization:** Utilizes 320x320 downsampling and 1-in-3 frame-skipping logic to enable smooth CPU execution without expensive GPU infrastructure.
* **Active Visual Overlays:** Replaces high-latency passive heatmaps with zero-latency, confidence-weighted bounding boxes for instant spatial verification.
* **Asynchronous Streaming Backend:** Built with FastAPI (ASGI) utilizing `multipart/x-mixed-replace` MJPEG streaming to eliminate HTTP buffering.
* **Memory-Efficient Caching:** Prevents RAM overflow during multi-gigabyte video stream ingestion via local SSD disk caching.

---

## 🛠️ System Architecture

[ Laparoscopic Stream ] ➔ [ OpenCV Preprocessing (320x320 & Frame Skip) ]
	│
	▼
[ YOLOv8 Tensor Inference ]
	│
	▼
[ Active Visual Box Overlay ]
	│
	▼
[ Asynchronous FastAPI Clinical Dashboard ]


---

## 📊 Performance Benchmark

| Metric | Performance |
| :--- | :--- |
| **Overall Accuracy (mAP@0.5)** | **96.0%** |
| **Precision / Recall** | 94.8% / 95.5% |
| **CPU Execution Speed (Intel Edge Hardware)** | 10–12 FPS |
| **GPU Execution Speed (NVIDIA RTX Series)** | 30+ FPS |
| **Top Performing Class** | Specimen Bag (99.2% mAP) |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash

pip install -r requirements.txt

2. Run the Dashboard

Bash

uvicorn main:app --reload

Open your browser and navigate to http://127.0.0.1:8000 to access the clinical dashboard.

📜 License
Distributed under the MIT License.