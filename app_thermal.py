import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import urllib.request
import os

st.set_page_config(
    page_title="🔫 Weapon Detection + Thermal",
    page_icon="🔫",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {font-size:30px !important; font-weight:bold; color:#FF4B4B;}
.thermal-badge {background-color:#FF6B6B; padding:10px; border-radius:5px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load weapon detection model with caching"""
    weapon_model_url = "https://github.com/Musawer1214/Weapon-Detection-YOLO/raw/main/weapon_detection_best.pt"
    weapon_model_path = "weapon_best.pt"
    
    if not os.path.exists(weapon_model_path):
        try:
            with st.spinner('🔄 Downloading specialized weapon detection model...'):
                urllib.request.urlretrieve(weapon_model_url, weapon_model_path)
            st.success("✅ Model downloaded successfully!")
        except:
            st.warning("⚠️ Using YOLOv8n fallback model...")
            weapon_model_path = 'yolov8n.pt'
    
    model = YOLO(weapon_model_path)
    return model

# Weapon class mapping (for proper labeling)
WEAPON_CLASSES = {
    'pistol': '🔫 Handgun/Pistol',
    'rifle': '🎯 Rifle',
    'knife': '🔪 Knife',
    'gun': '🔫 Gun',
    'firearm': '🔫 Firearm'
}

def apply_thermal_effect(image_array):
    """Convert image to thermal-like visualization"""
    # Convert to grayscale first
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    # Apply thermal colormap
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    # Convert back to RGB
    thermal_rgb = cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)
    return thermal_rgb

def detect_weapons_with_thermal(image, confidence, show_thermal):
    """Detect weapons and optionally apply thermal imaging"""
    if image is None:
        return None, "⚠️ Please upload an image"
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Apply thermal effect if requested
    if show_thermal:
        img_array = apply_thermal_effect(img_array)
    
    # Run weapon detection
    model = load_model()
    results = model(img_array, conf=confidence)
    result = results[0]
    
    # Get annotated image
    img_with_boxes = result.plot()
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    
    # Build detection summary
    boxes = result.boxes
    detection_text = f"**🎯 Total Detections: {len(boxes)}**\n\n"
    
    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            cls_name = result.names[cls]
            
            # Map to weapon-friendly names
            display_name = WEAPON_CLASSES.get(cls_name.lower(), f"⚠️ {cls_name.upper()}")
            detection_text += f"**{i+1}. {display_name}**\n"
            detection_text += f"   - Confidence: {conf:.1%}\n"
            detection_text += f"   - Class: {cls_name}\n\n"
    else:
        detection_text += f"✅ No weapons detected (confidence >= {confidence:.0%})\n"
        detection_text += "🔹 Try lowering the threshold or upload a different image."
    
    return img_rgb, detection_text

# Main app
st.markdown('<p class="big-font">🛡️ Professional Weapon Detection System</p>', unsafe_allow_html=True)
st.markdown("**🔥 With Thermal Imaging Capability**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Higher values = more confident detections only"
    )
    
    thermal_checkbox = st.checkbox(
        "🌡️ Enable Thermal Imaging",
        value=False,
        help="Apply thermal colormap for heat signature visualization"
    )
    
    st.markdown("---")
    st.header("📊 Model Info")
    st.info("""
    **Model**: YOLOv8 Weapon Detection
    **Classes**: Guns, Rifles, Knives
    **Accuracy**: High precision
    **Thermal**: OpenCV colormap
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to detect weapons"
    )

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    
    # Display original image
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    
    # Run detection
    with st.spinner("🔍 Running detection..."):
        annotated_img, detection_text = detect_weapons_with_thermal(
            image, confidence, thermal_checkbox
        )
    
    # Display results
    with col2:
        st.subheader("🎯 Detection Results")
        if annotated_img is not None:
            st.image(annotated_img, caption="Detected Objects", use_container_width=True)
    
    # Detection summary
    st.markdown("---")
    st.subheader("📋 Detection Summary")
    st.markdown(detection_text)
else:
    st.info("👈 Please upload an image to begin detection")
    
    # Demo info
    st.markdown("---")
    st.subheader("🚀 How to Use")
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("""
        **1. Upload Image** 📤
        
        Upload any image containing weapons.
        """)
    
    with col_info2:
        st.markdown("""
        **2. Adjust Settings** ⚙️
        
        Fine-tune confidence threshold.
        """)
    
    with col_info3:
        st.markdown("""
        **3. View Results** 🎯
        
        See detected weapons with bounding boxes.
        """)

st.markdown("---")
st.caption("Built with Streamlit & YOLOv8 | Professional Weapon Detection Demo")
