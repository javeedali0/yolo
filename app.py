import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from collections import Counter
from io import BytesIO

# Page config
st.set_page_config(
    page_title="Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .tag {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    return model

def draw_boxes(image, detections):
    img = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        label = f"{detection['name']} {detection['confidence']:.2f}"
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

# Header
st.markdown('<div class="header"><h1>🔍 Object Detection</h1><p>Upload an image to detect objects</p></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("Uses YOLOv8 to detect 80+ objects")

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

with col2:
    st.subheader("📊 Results")

# Process image
if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file)
        
        with st.spinner("Loading model..."):
            model = load_model()
        
        # Convert for detection
        img_array = np.array(image)
        
        with st.spinner("Detecting objects..."):
            start_time = time.time()
            results = model(img_array, conf=confidence)[0]
            detection_time = (time.time() - start_time) * 1000
        
        # Extract detections
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                name = results.names[cls]
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'name': name,
                    'class_id': cls
                })
        
        # Draw boxes
        annotated = draw_boxes(img_array, detections)
        
        # Show in columns
        with col1:
            st.image(image, caption="Original", use_column_width=True)
        
        with col2:
            if detections:
                # Stats
                st.metric("Total Objects", len(detections))
                st.metric("Detection Time", f"{detection_time:.0f}ms")
                
                # Annotated image
                st.image(annotated, caption="Detected", use_column_width=True)
                
                # Object summary
                counts = Counter([d['name'] for d in detections])
                st.markdown("### 🏷️ Detected")
                tags = " ".join([f'<span class="tag">{name} ({count})</span>' for name, count in counts.items()])
                st.markdown(tags, unsafe_allow_html=True)
                
                # Download
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                annotated_pil = Image.fromarray(annotated_rgb)
                buf = BytesIO()
                annotated_pil.save(buf, format="JPEG", quality=90)
                
                st.download_button(
                    label="📥 Download",
                    data=buf.getvalue(),
                    file_name="detected.jpg",
                    mime="image/jpeg"
                )
            else:
                st.info("No objects detected. Try lowering the confidence threshold.")
                st.image(image, use_column_width=True)
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please try with a different image")
else:
    # Placeholder
    with col2:
        st.info("👆 Upload an image to start")
        st.markdown("""
        <div style="text-align:center;padding:2rem;background:#f8f9ff;border-radius:10px;">
            <p style="font-size:3rem;">🔍</p>
            <p>Detects 80+ objects including:</p>
            <p>👤 People 🚗 Cars 🐕 Dogs 📱 Phones</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#666;'>Built with Streamlit & YOLOv8</p>", unsafe_allow_html=True)
