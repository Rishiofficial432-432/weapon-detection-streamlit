import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import urllib.request

st.set_page_config(
    page_title="🔫 Weapon Detection + Thermal",
    page_icon="🔫",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {font-size:30px !important; font-weight:bold; color:#FF4B4B;}
.thermal-badge {background-color:#FF8B66; padding:10px; border-radius:10px; border:2px solid #FF4B4B;}
.stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load weapon detection model with caching and priority"""
    weapon_model_url = "https://github.com/Musawer1214/Weapon-Detection-YOLO/raw/main/weapon_detection_best.pt"
    weapon_model_path = "weapon_best.pt"
    custom_model_path = "custom_model_v2.pt"
    
    model = None
    model_source = "None"

    # Priority 1: Custom Trained Model (Local)
    if os.path.exists(custom_model_path):
        try:
            model = YOLO(custom_model_path)
            model_source = "Custom Trained Model (Verified)"
        except Exception as e:
            st.warning(f"Failed to load custom model: {e}")

    # Priority 2: Pre-trained GitHub Model
    if model is None:
        if not os.path.exists(weapon_model_path):
            with st.spinner("� Downloading specialized weapon detection model..."):
                try:
                    urllib.request.urlretrieve(weapon_model_url, weapon_model_path)
                except:
                    pass
        
        if os.path.exists(weapon_model_path):
            try:
                model = YOLO(weapon_model_path)
                model_source = "GitHub Pre-trained Model"
            except:
                pass

    # Priority 3: Fallback YOLOv8n
    if model is None:
        st.warning("⚠️ Using standard YOLOv8n fallback model (Low Accuracy for Weapons)")
        model = YOLO('yolov8n.pt')
        model_source = "YOLOv8n Fallback"
    
    return model, model_source

def convert_to_thermal(image):
    """Convert normal image to thermal-style image (Simulated)"""
    img_array = np.array(image)
    if len(img_array.shape) == 2: # Already grayscale
        gray = img_array
    else:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    thermal = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
    thermal_rgb = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
    return thermal_rgb

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def detect_opencv_anomalies(image):
    """
    OpenCV Fallback: Detect anomalies that might be potential concealed objects.
    Logic: In thermal, hidden objects often appear as 'cold' (darker) spots on a 'warm' body 
    OR 'hot' spots depending on the environment.
    We'll look for significant contours that break the smooth localized texture.
    """
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 1. Morphological Gradient (Edges/Texture changes)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    
    # 2. Thresholding to find high gradients (edges of objects)
    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    annotated_img = img_array.copy()
    
    min_area = 500 # Minimum area to be considered an object
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Filter distinct shapes (weapons usually not extremely elongated or huge)
            aspect_ratio = float(w)/h
            if 0.2 < aspect_ratio < 5.0: 
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(annotated_img, "Anomaly", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                detections.append({
                    'class': 'anomaly (opencv)',
                    'confidence': 0.0, # N/A
                    'bbox': [x, y, x+w, y+h]
                })
                
    return annotated_img, detections

def detect_weapons_hybrid(model, image, confidence_threshold=0.3):
    """Run Hybrid (AI + OpenCV) detection"""
    
    # 1. AI Detection
    results = model(image, conf=confidence_threshold)
    ai_annotated = results[0].plot()
    
    ai_detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = results[0].names[cls_id]
        
        # Filter: If model is 'custom', classes are 0:Handgun, 1:Person
        # We generally want to ignore 'Person' if looking for weapons, unless user wants both.
        # User said "Weapon detection", so let's highlight Handgun primarily.
        
        if class_name.lower() in ['person', 'man', 'woman']:
             # Optional: don't count person as a 'threat'
             pass
        
        detection = {
            'class': class_name,
            'confidence': float(box.conf[0]),
            'bbox': box.xyxy[0].tolist(),
            'source': 'AI'
        }
        ai_detections.append(detection)
    
    # 2. OpenCV Fallback (Only if AI finds nothing or user wants extra check)
    cv_annotated, cv_detections = detect_opencv_anomalies(image)
    
    # Combine logic:
    # If AI found something, prefer AI visualization but maybe show CV count?
    # For now, let's return AI annotated if AI found anything. 
    # If AI empty, return CV annotated.
    
    final_annotated = ai_annotated
    final_detections = ai_detections
    
    if not ai_detections:
        final_annotated = cv_annotated
        final_detections = cv_detections
    return final_annotated, final_detections

def process_video(video_path, model, confidence_threshold):
    """Process video file frame by frame"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error opening video"

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output setup (Use mp4v for basic compatibility, or just process select frames for display)
    # Streamlit video playback often requires H.264 which is hard with basic OpenCV.
    # So we will process frames and show a progress bar, then maybe converting to GIF or showing keyframes.
    # OR better: Show a "Live" processing view using st.image placeholder.
    
    st_frame = st.empty()
    st_progress = st.progress(0)
    
    frame_count = 0
    detections_summary = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR (OpenCV) to RGB (Streamlit/PIL) logic
        # But wait, our pipeline is: Input -> Thermal Conversion -> Detection
        
        # 1. Convert to RGB for Thermal function
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Thermal Conversion
        thermal_rgb = convert_to_thermal(frame_rgb)
        
        # 3. Detect
        annotated_frame, frame_detections = detect_weapons_hybrid(model, thermal_rgb, confidence_threshold)
        
        if frame_detections:
            detections_summary += len(frame_detections)
        
        # 4. Display Live
        st_frame.image(annotated_frame, caption=f"Processing Frame {frame_count}/{total_frames}", use_container_width=True)
        
        frame_count += 1
        if total_frames > 0:
            st_progress.progress(min(frame_count / total_frames, 1.0))
            
    cap.release()
    st_progress.empty()
    return detections_summary

# Main UI
st.markdown('<p class="big-font">🔫 Weapon Detection System</p>', unsafe_allow_html=True)

# Load model
model, source_name = load_model()
st.caption(f"🤖 Active Model: **{source_name}**")

# Sidebar
st.sidebar.header("⚙️ Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
enable_opencv = st.sidebar.checkbox("Enable OpenCV Fallback", value=True, help="Use Computer Vision to detect anomalies if AI fails")

uploaded_file = st.file_uploader("photos", type=['jpg', 'jpeg', 'png', 'mp4'], label_visibility="hidden")

if uploaded_file:
    # Check file type
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == 'video' or uploaded_file.name.endswith('.mp4'):
        st.info("🎥 Processing Video Input...")
        
        # Save temp file
        import tempfile
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        try:
            total_threats = process_video(tfile.name, model, confidence)
            
            if total_threats > 0:
                st.error(f"🚨 VIDEO PROCESSING COMPLETE: Found {total_threats} threats across frames.")
            else:
                st.success("✅ Video Clean: No threats detected.")
        except Exception as e:
            st.error(f"Error processing video: {e}")
        finally:
            # Cleanup
            tfile.close()
            # os.unlink(tfile.name) # Keep for now or delete
            
    else:
        # Image Processing
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process
        with st.spinner("Running detection algorithms..."):
            thermal_img = convert_to_thermal(image)
            annotated_img, detections = detect_weapons_hybrid(model, thermal_img, confidence)
            
            st.markdown("### 🎯 Results")
            st.image(annotated_img, caption=f"Processed Image ({len(detections)} detections)", use_container_width=True)
            
            if detections:
                st.error(f"🚨 FOUND {len(detections)} POTENTIAL THREATS")
                for d in detections:
                    source_badge = "🤖 AI" if d.get('source') == 'AI' else "👁️ CV"
                    st.write(f"**{source_badge}**: {d['class']} ({d['confidence']:.2f})")
            else:
                st.success("✅ Area Clear")

