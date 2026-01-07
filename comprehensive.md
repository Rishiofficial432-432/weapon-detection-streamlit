# Weapon Detection System - Comprehensive Documentation

This document provides a detailed technical breakdown of the Weapon Detection Streamlit Application. It covers the software architecture, data flow, dependencies, and the machine learning pipeline used to build and run the system.

---

## 1. Software Flow & Architecture

The application is a **Streamlit** web interface that wraps a hybrid computer vision pipeline. It is designed to take standard RGB images or videos, simulate a thermal imaging effect, and then detect concealed weapons using both Deep Learning (AI) and standard Computer Vision (CV).

### **High-Level Workflow**
1.  **Initialization**:
    *   App starts and loads the best available model (`custom_model_v2.pt` > `weapon_best.pt` > `yolov8n.pt`).
    *   UI components (headers, sidebars) are rendered.
2.  **User Input**:
    *   User uploads a file (JPG, PNG, or MP4) via the `st.file_uploader`.
3.  **Preprocessing**:
    *   **Image**: Converted to a numpy array.
    *   **Video**: Saved to a temporary file, then processed frame-by-frame using `cv2.VideoCapture`.
    *   **Thermal Simulation**: Input RGB frames are converted to grayscale, contrast-enhanced (CLAHE), and color-mapped (`COLORMAP_JET`) to simulate thermal heat signatures.
4.  **Detection Pipeline (Hybrid)**:
    *   **Stage 1: AI Detection (YOLO)**: The thermal image is passed to the loaded YOLOv8 model. It predicts bounding boxes and classes (e.g., "Handgun").
    *   **Stage 2: CV Fallback (OpenCV)**: If enabled, the system calculates gradients and finds contours ("hotspots") that deviate from the background. This catches potential threats the AI might miss.
5.  **Result Aggregation**:
    *   AI and CV detections are combined.
    *   Bounding boxes are drawn on the image/frame.
    *   Results are displayed to the user with a summary (e.g., "FOUND 3 THREATS").

---

## 2. Package Dependencies & Roles

The system relies on specific Python packages. Here is *why* they are used and *when* they are triggered:

| Package | Purpose | Why it's there | Trigger Point |
| :--- | :--- | :--- | :--- |
| **`streamlit`** | Web Interface | Provides the frontend UI (buttons, uploaders, image display) without needing HTML/JS. | Triggered immediately on `app_thermal.py` launch. |
| **`ultralytics`** | AI/Deep Learning | Contains the YOLOv8 framework used for object detection inference and training. | Triggered during `load_model()` and detection calls `model(image)`. |
| **`opencv-python-headless`** | Computer Vision | Used for image manipulation (resizing, color conversion, thermal simulation) and video processing. | Triggered during `convert_to_thermal`, `process_video`, and `detect_opencv_anomalies`. |
| **`numpy`** | Data Handling | Handles image data as matrices/arrays (essential for OpenCV and PyTorch). | Triggered constantly whenever an image is loaded or processed. |
| **`pillow` (PIL)** | Image Loading | User-friendly image library to load uploaded files before converting to numpy arrays. | Triggered after `file_uploader` receives a file. |
| **`urllib`** | Network Requests | Used to download the fallback pre-trained model from GitHub if no local model exists. | Triggered in `load_model()` only if `custom_model_v2.pt` is missing. |

---

## 3. Data Pipeline & Training

The core intelligence of the system comes from the **Custom Trained Model (v2)**.

### **3.1 The Dataset**
*   **Source**: `UCLM_Thermal_Imaging_Dataset` (and a duplicate folder).
*   **Format**: The original data consisted of **Thermal MP4 Videos** and a **JSON Label File**.
*   **Challenge**: YOLO cannot train directly on Video+JSON. It requires Images+TXT.
*   **Solution**: A custom processing script (`prepare_data.py`) was built.

### **3.2 Data Preparation (`prepare_data.py`)**
1.  **Parsing**: The script read `label.json` to find which video frames contained weapons.
2.  **Extraction**: It opened the MP4 files using OpenCV and extracted only the frames that had matching annotations.
3.  **Conversion**: It converted the JSON bounding box format `[x, y, w, h]` (pixel values) into YOLO format `[class_id, x_center, y_center, width, height]` (normalized 0-1).
4.  **Filtering**: 
    *   Originally, the dataset contained "Handgun" and "Person".
    *   **Refinement**: We modified the script to **ignore 'Person' labels completely**. This forces the model to learn *only* the weapon features, preventing false alarms on human bodies.

### **3.3 Optimization (`reduce_dataset.py`)**
*   To enable rapid training/prototyping, we downsampled the extracted dataset (thousands of images) to a high-quality subset of **500 images**.

### **3.4 Training Configuration**
*   **Script**: `train_model.py`
*   **Base Model**: `yolov8n.pt` (Nano version for speed).
*   **Epochs**: 5 (Sufficient for this subset to learn the specific thermal signature of a handgun).
*   **Classes**: 1 class only (`0: Handgun`).

---

## 4. Image & Video Processing Logic

### **Thermal Simulation**
Since users might upload standard RGB photos, the app simulates a thermal view to match the training data domain:
1.  **Grayscale**: `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` - Removes color info.
2.  **Blur**: `cv2.GaussianBlur` - Smooths noise (thermal cameras are often low-res/blurry).
3.  **CLAHE**: Adaptive Histogram Equalization - Boosts contrast locally to highlight "hot" vs "cold" areas.
4.  **Colormap**: `cv2.applyColorMap(..., cv2.COLORMAP_JET)` - Applies the blue-to-red gradient typical of heat maps.

### **Video Handling**
Streamlit cannot natively process a video stream frame-by-frame from a file upload easily, so we use a workaround:
1.  **Temp File**: The uploaded video bytes are written to a temporary file on disk.
2.  **Iterative Processing**: `cv2.VideoCapture` reads this temp file.
3.  **Loop**:
    *   Read Frame -> Convert to RGB -> Simulate Thermal -> Detect -> Overlay Boxes -> Display.
    *   A progress bar updates with `frame_count / total_frames`.

---

## 5. Summary of Files

*   **`app_thermal.py`**: The main application entry point. Contains UI, Detection Logic, and Hybrid System.
*   **`prepare_data.py`**: Utility to convert raw Video+JSON data into YOLO Images+TXT format.
*   **`train_model.py`**: Script to launch the YOLO training process.
*   **`reduce_dataset.py`**: Utility to shrink the dataset for faster iteration.
*   **`custom_data.yaml`**: Configuration file telling YOLO where to find images and what classes to look for.
*   **`custom_model_v2.pt`**: The final trained model weights (The "Brain" of the system).
