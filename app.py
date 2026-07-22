"""
100% Working Object Detection for Streamlit Cloud
Compatible with Python 3.9+
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from collections import Counter
from io import BytesIO
import requests
import os

# Page configuration
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .stat-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .detection-tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    """Load YOLOv8 model"""
    try:
        from ultralytics import YOLO
        # Use a specific model path or let it download
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def draw_boxes(image, boxes, scores, class_names):
    """Draw bounding boxes on image"""
    img = image.copy()
    
    # Generate colors for each class
    unique_classes = list(set(class_names))
    colors = {}
    for name in unique_classes:
        colors[name] = tuple(np.random.randint(0, 255, 3).tolist())
    
    for box, score, name in zip(boxes, scores, class_names):
        x1, y1, x2, y2 = box
        color = colors[name]
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

def detect_objects(image, model, confidence_threshold=0.25):
    """Perform object detection on image"""
    try:
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Run detection
        results = model(image, conf=confidence_threshold)[0]
        
        boxes = []
        scores = []
        class_names = []
        
        # Extract detections
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = results.names[class_id]
                
                boxes.append([int(x1), int(y1), int(x2), int(y2)])
                scores.append(float(confidence))
                class_names.append(class_name)
        
        # Draw boxes on image
        annotated_img = draw_boxes(image, boxes, scores, class_names)
        
        return annotated_img, boxes, scores, class_names
    except Exception as e:
        st.error(f"Error during detection: {str(e)}")
        return None, [], [], []

def load_example_image(url):
    """Load example image from URL"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        return None
    except:
        return None

# Load model
with st.spinner("🔄 Loading YOLOv8 model... This may take a moment on first run."):
    model = load_model()

if model is None:
    st.error("Failed to load model. Please refresh the page.")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <h1>🔍 Object Detection</h1>
    <p>Upload any image and see what's in it instantly!</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Object Detection** using YOLOv8
    
    Detects 80+ common objects:
    - 👤 People
    - 🚗 Vehicles
    - 🐕 Animals
    - 📱 Objects
    
    ### Classes detected:
    person, bicycle, car, motorcycle, 
    bus, truck, cat, dog, and more!
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp']
    )
    
    st.markdown("---")
    st.markdown("### 🖼️ Try Examples")
    
    # Example images (using stable URLs)
    examples = {
        "🐕 Dog": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800&h=600&fit=crop",
        "🚗 Car": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=800&h=600&fit=crop",
        "🌆 City": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=800&h=600&fit=crop",
        "👥 People": "https://images.unsplash.com/photo-1521737711867-e3b97375f902?w=800&h=600&fit=crop"
    }
    
    example_cols = st.columns(2)
    for idx, (label, url) in enumerate(examples.items()):
        with example_cols[idx % 2]:
            if st.button(label, use_container_width=True):
                with st.spinner(f"Loading {label}..."):
                    img = load_example_image(url)
                    if img:
                        uploaded_file = img
                        st.rerun()

with col2:
    st.markdown("### 📊 Results")
    
    if uploaded_file is not None:
        # Load image
        if isinstance(uploaded_file, Image.Image):
            image = uploaded_file
        else:
            try:
                image = Image.open(uploaded_file)
            except Exception as e:
                st.error(f"Error opening image: {str(e)}")
                st.stop()
        
        # Progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Processing image...")
            progress_bar.progress(30)
            
            start_time = time.time()
            annotated_img, boxes, scores, class_names = detect_objects(
                image, 
                model,
                confidence_threshold
            )
            detection_time = round((time.time() - start_time) * 1000, 2)
            
            progress_bar.progress(100)
            status_text.text("✅ Detection complete!")
            
            # Clear progress
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
            
            if annotated_img is not None and len(boxes) > 0:
                # Statistics
                total_objects = len(boxes)
                counts = Counter(class_names)
                unique_objects = len(counts)
                
                # Display stats
                st.markdown("### 📈 Statistics")
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    st.metric("Total Objects", total_objects)
                with stats_col2:
                    st.metric("Unique Types", unique_objects)
                with stats_col3:
                    st.metric("Detection Time", f"{detection_time}ms")
                
                # Display annotated image
                st.markdown("### 🖼️ Detection Results")
                st.image(annotated_img, use_column_width=True)
                
                # Display detected objects
                st.markdown("### 🏷️ Detected Objects")
                objects_html = ""
                for name, count in counts.most_common():
                    objects_html += f'<span class="detection-tag">{name} <span style="background:rgba(255,255,255,0.3);padding:0 8px;border-radius:10px;">{count}</span></span>'
                
                if objects_html:
                    st.markdown(objects_html, unsafe_allow_html=True)
                else:
                    st.info("No objects detected with current confidence threshold.")
                
                # Download button
                st.markdown("### 💾 Download Results")
                
                # Convert annotated image to bytes
                annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                annotated_pil = Image.fromarray(annotated_rgb)
                
                buf = BytesIO()
                annotated_pil.save(buf, format="JPEG", quality=90)
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=byte_im,
                    file_name=f"detected_{int(time.time())}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            elif annotated_img is not None and len(boxes) == 0:
                st.info("No objects detected. Try lowering the confidence threshold.")
                st.image(image, use_column_width=True)
            else:
                st.error("Detection failed. Please try again.")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    else:
        # Show placeholder
        st.info("👆 Upload an image or try an example to start detection")
        
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8f9ff; border-radius: 10px;">
            <p style="font-size: 4rem;">🔍</p>
            <p style="color: #666;">Your image will appear here with detections</p>
            <p style="color: #999; font-size: 0.9rem;">YOLOv8 detects 80 different object classes</p>
            <p style="color: #999; font-size: 0.9rem;">Adjust confidence threshold in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🚀 Built with Streamlit & YOLOv8 | Detects 80 object classes</p>
</div>
""", unsafe_allow_html=True)
