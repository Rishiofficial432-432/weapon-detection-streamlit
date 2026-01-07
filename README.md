---
title: Weapon Detection System
emoji: 🔫
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.32.0
app_file: app_thermal.py
pinned: false
license: mit
---

# 🔫 Weapon Detection System (Thermal & Hybrid)

A state-of-the-art **Streamlit** application designed to detect concealed weapons in images and videos. This system utilizes a **Hybrid Detection Pipeline**, combining a custom-trained **YOLOv8 AI Interaction Model** with **OpenCV Computer Vision** algorithms to identify potential threats in thermal-simulated environments.

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/Streamlit-1.32.0-red)
![AI Model](https://img.shields.io/badge/YOLOv8-Custom_Trained-orange)

---

## 🚀 Key Features

*   **🔥 Thermal Simulation**: Automatically converts standard RGB uploads into thermal heat-map representations (`COLORMAP_JET`) to simulate infrared scanner views.
*   **🤖 Custom AI Detection**:
    *   Powered by `custom_model_v2.pt`, a YOLOv8 model trained specifically on **Thermal Handgun Data**.
    *   **Strict Filtering**: Trained to ignore "Persons" and focus *exclusively* on weapons, reducing false positives.
*   **👁️ Hybrid Fallback (OpenCV)**:
    *   Includes a secondary Computer Vision detector that scans for "Thermal Anomalies" (hot/cold spots).
    *   Ensures that even if the AI misses a weapon, unusual heat signatures are flagged as potential threats.
*   **🎥 Video Support**:
    *   Upload MP4 videos for full-frame analysis.
    *   Processes video frame-by-frame and provides a threat summary.
*   **⚡ Priority Loading System**:
    *   Automatically attempts to load: `Custom Local Model` -> `GitHub Pre-trained Model` -> `Standard YOLOv8n` (Fallback).

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.8 or higher installed on your system.
*   (Optional) Virtual Environment recommended.

### 1. Clone the Repository
```bash
git clone https://github.com/Rishiofficial432-432/weapon-detection-streamlit.git
cd weapon-detection-streamlit
```

### 2. Run the Setup Script
We provide a helper script to set up your environment and launch the app in one go:
```bash
# Make the script executable (Mac/Linux)
chmod +x run.sh

# Run it
./run.sh
```
*This script will create a virtual environment, install all dependencies from `requirements.txt`, and start the Streamlit server.*

### Manual Installation
If you prefer to install manually:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_thermal.py
```

---

## 🖥️ Usage Guide

1.  **Launch**: Open your browser to the URL provided (usually `http://localhost:8501`).
2.  **Settings (Sidebar)**:
    *   **Confidence Threshold**: Adjust how sure the AI needs to be before flagging an item. (Default: 0.25).
    *   **Enable OpenCV Fallback**: Toggle the secondary anomaly detector.
3.  **Upload**:
    *   Drag & Drop an Image (`.jpg`, `.png`) or Video (`.mp4`).
4.  **Results**:
    *   **Images**: The processed image will appear with bounding boxes.
        *   **Red/Pink Boxes**: AI Detections (Handguns).
        *   **Yellow/Cyan Boxes**: CV Anomalies (Suspicious shapes).
    *   **Videos**: The app will iterate through the video and report the total number of threats found across all frames.

---

## 🧩 Technical Architecture

### 1. The Models
The system uses a tiered loading strategy:
*   **Tier 1: `custom_model_v2.pt` (Primary)**
    *   **Trained On**: `UCLM_Thermal_Imaging_Dataset` (500 selected frames).
    *   **Classes**: `0: Handgun`. (Excluded 'Person' to improve accuracy).
    *   **Status**: Locally trained and verified.
*   **Tier 2: `weapon_best.pt`**
    *   A pre-trained community model downloaded automatically if the local custom model is missing.
*   **Tier 3: `yolov8n.pt`**
    *   Standard COCO model used as a last resort fallback.

### 2. The Hybrid Pipeline
Why use both AI and CV?
*   **Deep Learning (YOLO)** is excellent at recognizing *shapes* (e.g., the outline of a gun).
*   **Computer Vision (OpenCV)** is excellent at finding *texture breaks* (e.g., a cold metal object against a warm body).
*   **Combined**: By running both, we minimize the chance of a concealed weapon going undetected.

### 3. Data Flow
`User Upload` -> `Temp File (if video)` -> `PIL Image` -> `Numpy Array` -> `Thermal Simulation (CLAHE + ColorMap)` -> `Inference` -> `Annotation` -> `UI Display`

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| **`app_thermal.py`** | The main application code containing UI and Hybrid Detection Logic. |
| **`custom_model_v2.pt`** | The custom YOLOv8 detection model weights. |
| **`prepare_data.py`** | (Dev) Script used to convert raw Video+JSON data into training images. |
| **`train_model.py`** | (Dev) Script used to train the YOLO model. |
| **`reduce_dataset.py`** | (Dev) Utility to optimize dataset size. |
| **`requirements.txt`** | List of Python dependencies. |
| **`run.sh` / `stop.sh`** | Utility scripts to Start and Stop the application. |

---

## ⚠️ Disclaimer
This software is intended for **research and educational purposes**. The detection system is a demonstration of hybrid computer vision techniques and should not be relied upon as the sole security measure in critical safety environments.
