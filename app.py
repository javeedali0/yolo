"""
100% Working Object Detection with Streamlit
Uses YOLOv8 - Production Ready
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
from collections import Counter
from io import BytesIO
import requests
import os
import sys

# Page configuration
st.set_page_config(
    page_title="Object Detection App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
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
        transition: transform 0.3s;
    }
    .stat-box:hover {
        transform: scale(1.05);
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
        border-top: 1px solid #eee;
    }
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

@st.cache_resource
def load_model():
    """Load YOLOv8 model with caching"""
    try:
        # Check if model exists, if not download
        model_path = 'yolov8n.pt'
        if not os.path.exists(model_path):
            st.info("📥 Downloading YOLOv8 model... This will happen only once.")
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def draw_boxes(image, boxes, scores, class_names):
    """Draw bounding boxes on image with improved visualization"""
    img = image.copy()
    
    # Generate colors for each class
    unique_classes = list(set(class_names))
    colors = {}
    for name in unique_classes:
        colors[name] = tuple(np.random.randint(0, 255, 3).tolist())
    
    for box, score, name in zip(boxes, scores, class_names):
        x1, y1, x2, y2 = box
        color = colors[name]
        
        # Draw rectangle with rounded corners (approximated)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Background for text
        cv2.rectangle(img, (x1 - 2, y1 - th - 12), (x1 + tw + 10, y1 - 2), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add confidence bar
        bar_width = int(score * 60)
        cv2.rectangle(img, (x1 + 5, y2 + 5), (x1 + 5 + bar_width, y2 + 15), color, -1)
        cv2.rectangle(img, (x1 + 5, y2 + 5), (x1 + 65, y2 + 15), (255, 255, 255), 1)
    
    return img

def detect_objects(image, confidence_threshold=0.25):
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
        else:
            st.error(f"Failed to load example image (Status: {response.status_code})")
            return None
    except Exception as e:
        st.error(f"Error loading example image: {str(e)}")
        return None

# Load model
try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    model_loaded = False
    st.stop()

# Main UI
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
        step=0.05,
        help="Lower values detect more objects but may include false positives"
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
    - And more!
    
    ### 🎯 How to use:
    1. Upload an image
    2. Adjust confidence if needed
    3. View results
    4. Download annotated image
    """)
    
    if st.button("🗑️ Clear History"):
        st.session_state.detection_history = []
        st.success("History cleared!")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        help="Supported formats: JPG, PNG, WebP, BMP"
    )
    
    st.markdown("---")
    st.markdown("### 🖼️ Try Examples")
    
    # Example images
    examples = {
        "🐕 Dog & Cat": "https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=800&h=600&fit=crop",
        "🚗 Street": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=800&h=600&fit=crop",
        "🌆 City": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=800&h=600&fit=crop",
        "👥 People": "https://images.unsplash.com/photo-1521737711867-e3b97375f902?w=800&h=600&fit=crop",
        "🏠 House": "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=800&h=600&fit=crop",
        "🌸 Nature": "https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=800&h=600&fit=crop"
    }
    
    # Create example buttons in a grid
    example_cols = st.columns(3)
    for idx, (label, url) in enumerate(examples.items()):
        with example_cols[idx % 3]:
            if st.button(label, use_container_width=True, key=f"example_{idx}"):
                with st.spinner(f"Loading {label}..."):
                    img = load_example_image(url)
                    if img:
                        uploaded_file = img
                        st.rerun()

with col2:
    st.markdown("### 📊 Results")
    
    # Process image
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
        
        # Display original image
        with st.expander("📸 Original Image", expanded=False):
            st.image(image, use_column_width=True)
        
        # Detection placeholder
        detection_placeholder = st.empty()
        
        # Progress
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # Detect objects
            progress_text.text("🔄 Processing image...")
            progress_bar.progress(30)
            
            start_time = time.time()
            annotated_img, boxes, scores, class_names = detect_objects(
                image, 
                confidence_threshold
            )
            detection_time = round((time.time() - start_time) * 1000, 2)
            
            progress_bar.progress(100)
            progress_text.text("✅ Detection complete!")
            
            # Clear progress
            time.sleep(0.5)
            progress_text.empty()
            progress_bar.empty()
            
            if annotated_img is not None:
                # Calculate statistics
                total_objects = len(boxes)
                counts = Counter(class_names)
                unique_objects = len(counts)
                
                # Display stats
                st.markdown("### 📈 Statistics")
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric("Total Objects", total_objects)
                with stats_col2:
                    st.metric("Unique Types", unique_objects)
                with stats_col3:
                    st.metric("Detection Time", f"{detection_time}ms")
                with stats_col4:
                    st.metric("Confidence", f"{confidence_threshold:.2f}")
                
                # Display annotated image
                st.markdown("### 🖼️ Detection Results")
                st.image(annotated_img, use_column_width=True)
                
                # Display detected objects as tags
                st.markdown("### 🏷️ Detected Objects")
                objects_html = ""
                for name, count in counts.most_common():
                    objects_html += f'<span class="detection-tag">{name} <span style="background:rgba(255,255,255,0.3);padding:0 8px;border-radius:10px;">{count}</span></span>'
                
                if objects_html:
                    st.markdown(objects_html, unsafe_allow_html=True)
                else:
                    st.info("No objects detected with current confidence threshold. Try lowering the threshold.")
                
                # Save to history
                detection_data = {
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'total_objects': total_objects,
                    'unique_objects': unique_objects,
                    'detection_time': detection_time,
                    'objects': dict(counts)
                }
                st.session_state.detection_history.append(detection_data)
                
                # Download button
                st.markdown("### 💾 Download Results")
                
                # Convert annotated image to bytes
                annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                annotated_pil = Image.fromarray(annotated_rgb)
                
                # Save to bytes
                buf = BytesIO()
                annotated_pil.save(buf, format="JPEG", quality=90)
                byte_im = buf.getvalue()
                
                download_col1, download_col2 = st.columns(2)
                with download_col1:
                    st.download_button(
                        label="📥 Download Image",
                        data=byte_im,
                        file_name=f"detected_{int(time.time())}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                
                # Export results as CSV
                if st.button("📊 Export Results as CSV", use_container_width=True):
                    import pandas as pd
                    df = pd.DataFrame([
                        {"Object": name, "Count": count, "Confidence": confidence_threshold}
                        for name, count in counts.most_common()
                    ])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"detection_results_{int(time.time())}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            else:
                st.error("Detection failed. Please try again with a different image.")
                
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
    else:
        # Show placeholder
        st.info("👆 Upload an image or try an example to start detection")
        
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8f9ff; border-radius: 10px;">
            <p style="font-size: 4rem;">🔍</p>
            <p style="color: #666;">Your image will appear here with detections</p>
            <p style="color: #999; font-size: 0.9rem;">YOLOv8 detects 80 different object classes</p>
            <p style="color: #999; font-size: 0.9rem;">Try adjusting the confidence threshold in sidebar</p>
        </div>
        """, unsafe_allow_html=True)

# Detection History
if st.session_state.detection_history:
    st.markdown("---")
    st.markdown("### 📜 Detection History")
    
    # Show last 5 detections
    history_df = []
    for i, record in enumerate(st.session_state.detection_history[-5:][::-1]):
        history_df.append({
            "Time": record['timestamp'],
            "Objects": record['total_objects'],
            "Unique": record['unique_objects'],
            "Time (ms)": record['detection_time'],
            "Detected": ", ".join([f"{k}({v})" for k, v in record['objects'].items()][:3])
        })
    
    if history_df:
        import pandas as pd
        st.dataframe(pd.DataFrame(history_df), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🚀 Built with Streamlit & YOLOv8 | Detects 80 object classes</p>
    <p style="font-size: 0.8rem;">Objects: person, bicycle, car, motorcycle, bus, truck, cat, dog, and many more...</p>
</div>
""", unsafe_allow_html=True)
