import streamlit as st
import cv2, time
import tempfile
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk
import os
import base64
import io

# Initialize session state for model_choice and other widgets
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = 'YOLOv5 Small'
if 'augmentation_type' not in st.session_state:
    st.session_state.augmentation_type = 'None'

# Define a callback function to reset dependent widgets
def reset_widgets():
    st.session_state.augmentation_type = 'None'

st.sidebar.markdown("## About")
st.sidebar.info("""
This app was developed to assist in marine object detection and environmental metadata logging.
You use YOLOv8-based models trained on custom underwater dataset.
""")


# Sidebar for model selection with callback
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose a YOLO Model",
    ["YOLOv5 Small", "YOLOv8 Small", "YOLOv8 Extra Large", "YOLOv8_10 Net Detector", "YOLOv8_25 Net Detector", "YOLOv8_50 Net Detector", "YOLOv8_75 Net Detector"],
    key='model_choice',
    on_change=reset_widgets,  # Assign the callback function
    help="Select a pre-trained YOLO model. Larger models may be more accurate but slower."
)

# Map model choices to their corresponding file paths
model_paths = {
    "YOLOv5 Small": "models/yolov5su.pt",
    "YOLOv8 Small": "models/yolov8n-oiv7.pt",
    "YOLOv8 Extra Large": "models/yolov8x-oiv7.pt",
    "YOLOv8_10 Net Detector": "models/best_10.pt",
    "YOLOv8_25 Net Detector": "models/best_25.pt",
    "YOLOv8_50 Net Detector": "models/best_50.pt",
    "YOLOv8_75 Net Detector": "models/best_75.pt"
}


# Load the selected model
model_path = model_paths.get(st.session_state.model_choice)
if model_path:
    try:
        model = YOLO(model_path)
        st.success(f"Loaded model: {st.session_state.model_choice}")
    except Exception as e:
        st.error(f"Failed to load model '{st.session_state.model_choice}': {e}")
else:
    st.error("Invalid model selection. Please choose a valid model.")
#🎣 🧵 🐠 🐟 🐡
st.title("🐠🧵🐡🎣App Detection🎣🐡🧵🐠")


def flip_image(image, flip_type):
    if flip_type == "Horizontal Flip":
        return transforms.RandomHorizontalFlip(p=1.0)(image)
    elif flip_type == "Vertical Flip":
        return transforms.RandomVerticalFlip(p=1.0)(image)
    return image


def crop_image(image):
    return transforms.RandomResizedCrop(size=(640, 640))(image)


def change_colors(image, brightness=1.0, contrast=1.0,
                  saturation=1.0, hue=0.0):
    transform = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue
    )
    return transform(image)


def random_affine(image, degrees=15, translate=0.1, scale=1.0, shear=10):
    transform = transforms.RandomAffine(
        degrees=degrees,
        translate=(translate, translate),
        scale=(scale, scale),
        shear=shear
    )
    return transform(image)


def gaussian_blur(image, kernel_size=(5, 5), sigma=1.0):
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(image)


def random_erasing(image, scale_min=0.02, scale_max=0.33):
    image_tensor = TF.to_tensor(image)
    transform = transforms.RandomErasing(
        p=1.0,
        scale=(scale_min, scale_max),
        value="random"
    )
    erased_tensor = transform(image_tensor)
    return TF.to_pil_image(erased_tensor)

def encode_image_base64(path):
    try:
        with open(path, "rb") as f:
            img_bytes = f.read()
        encoded = base64.b64encode(img_bytes).decode()
        return f'<img src="data:image/jpeg;base64,{encoded}" width="120"/>'
    except:
        return "Image not found"



# Buttons for augmentations
augmentation_type = st.sidebar.radio(
    "Choose Augmentation", 
    ["None", "Horizontal Flip", "Vertical Flip", "Random Crop", 
     "Change Colors", "Random Affine", "Gaussian Blur", "Random Erasing"],
     help="Apply real-time data augmentation before running object detection."
)

# Extra sliders for adjustments
if augmentation_type == "Change Colors":
    brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, step=0.1)
    contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, step=0.1)
    saturation = st.sidebar.slider("Saturation", 0.5, 2.0, 1.0, step=0.1)
    hue = st.sidebar.slider("Hue", 0.0, 0.2, 0.5, step=0.05)

if augmentation_type == "Random Affine":
    degrees = st.sidebar.slider("Rotation (degrees)", 0, 45, 15)
    translate = st.sidebar.slider("Translate %", 0.0, 0.5, 0.1)
    scale = st.sidebar.slider("Scale Range", 0.5, 1.5, 1.0)
    shear = st.sidebar.slider("Shear", 0.0, 30.0, 10.0)

elif augmentation_type == "Gaussian Blur":
    kernel_size = st.sidebar.selectbox("Kernel Size", [(3, 3), (5, 5), (9, 9)])
    sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0)

elif augmentation_type == "Random Erasing":
    scale_min = st.sidebar.slider("Erasing Scale Min", 0.01, 0.1, 0.02)
    scale_max = st.sidebar.slider("Erasing Scale Max", 0.1, 0.5, 0.35)


# # Function to draw bounding boxes
def draw_bounding_boxes(image, results, show_streamlit_output=True):
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                label = f"{name}: {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw label
                cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3, cv2.LINE_AA)

                # Streamlit output
                if show_streamlit_output:
                    st.write(f"Detected: **{name}** with confidence **{conf:.2f}**")

    return image


#Set the confidence threshold value
conf_threshold = st.sidebar.slider("Confidence Threshold", 
                                   min_value=0.01, max_value=0.5, 
                                   value=0.1, 
                                   step=0.01,
                                   help="Set the minimum confidence score for object detection. Lower values show more results but may include false positives."
                                   )

# Upload image or video
uploaded_file = st.sidebar.file_uploader("Upload an Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])


# Process image 
if uploaded_file and uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
    UPLOAD_FOLDER = "uploaded_images"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Save image
    uploaded_image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(uploaded_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    
    image = Image.open(uploaded_file)

    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Explicit resizing
    image = image.resize((640, 640))






    # Apply chosen augmentation
    if augmentation_type == "Horizontal Flip":
        #augmented_image = flip_image_d2l(image, "Horizontal Flip")
        augmented_image = flip_image(image, "Horizontal Flip")
    elif augmentation_type == "Vertical Flip":
        #augmented_image = flip_image_d2l(image, "Vertical Flip")
        augmented_image = flip_image(image, "Vertical Flip")
    elif augmentation_type == "Random Crop":
        #augmented_image = crop_image_d2l(image)
        augmented_image = crop_image(image)
    elif augmentation_type == "Change Colors":
        augmented_image = change_colors(image, brightness, contrast, saturation, hue)
        #augmented_image = change_colors_d2l(image, brightness, contrast, saturation, hue)
        
    elif augmentation_type == "Random Affine":
        #augmented_image = random_affine_d2l(image, degrees, translate, scale, shear)
        augmented_image = random_affine(image, degrees, translate, scale, shear)
    elif augmentation_type == "Gaussian Blur":
        #augmented_image = gaussian_blur_d2l(image, kernel_size, sigma)
        augmented_image = gaussian_blur(image, kernel_size, sigma)
    elif augmentation_type == "Random Erasing":
        #augmented_image = random_erasing_d2l(image, scale_min, scale_max)
        augmented_image = random_erasing(image, scale_min, scale_max)
    else:
        augmented_image = image

    # ------------------------------
    # 
        
    image_np = np.array(image)  # Convert to numpy array
    # augmented_np = np.array(augmented_image)
    # # Convert RGB (PIL/Streamlit) to BGR (YOLO expects BGR when using OpenCV images)
    # image_bgr = cv2.cvtColor(augmented_np, cv2.COLOR_RGB2BGR)

    # # Run YOLOv8 detection
    # results = model(image_bgr, conf=conf_threshold)
    # Convert augmented image safely to RGB numpy

# Convert augmented image safely to RGB numpy
# Convert augmented image safely before YOLO prediction
    if isinstance(augmented_image, torch.Tensor):
        augmented_image = TF.to_pil_image(augmented_image)

    if augmented_image.mode != "RGB":
        augmented_image = augmented_image.convert("RGB")

    augmented_np = np.array(augmented_image)

    if augmented_np.dtype != np.uint8:
        augmented_np = augmented_np.astype(np.uint8)

    # Run YOLO detection
    results = model(augmented_np, conf=conf_threshold)

    # Draw bounding boxes
    #image_with_boxes = draw_bounding_boxes(augmented_np.copy(), results)

# Display directly, no BGR/RGB conversion needed

    # Draw bounding boxes
    #image_with_boxes = draw_bounding_boxes(image_bgr.copy(), results)
    image_with_boxes = draw_bounding_boxes(augmented_np.copy(), results)        

    # Convert images to RGB for Streamlit
    st.image(image_with_boxes, use_container_width=True)
    # ------------------------------

    # Display Images
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(augmented_image, caption="Augmented Image", use_container_width=True)
    with col3:
        st.image(image_with_boxes, caption="Detected Objects", use_container_width=True)

    st.markdown("---")
    st.subheader("🌍 Environmental Metadata Annotation")

    with st.form("env_form"):
        temperature = st.slider("🌡 Temperature (°C)", min_value=-5.0, max_value=40.0, value=20.0, step=0.5)
        depth = st.number_input("📏 Depth (m)", min_value=0, max_value=1000, value=10)
        latitude = st.number_input("🌍 Latitude", min_value=-90.0, max_value=90.0, value=0.0, step=0.0001, format="%.4f")
        longitude = st.number_input("🌍 Longitude", min_value=-180.0, max_value=180.0, value=0.0, step=0.0001, format="%.4f")
        annotator_name = st.text_input("🧑 Annotator Name", placeholder="Your Name")
        
        submit_button = st.form_submit_button("💾 Save Annotation")

    if submit_button:
        st.success("Annotation saved!")
        
        # Example of how you could structure the saved data
        annotation = {
            "filename": uploaded_file.name if uploaded_file else "none",
            "filepath": uploaded_image_path, 
            "temperature": temperature,
            "depth": depth,
            "latitude": latitude,
            "longitude": longitude,
            "annotator": annotator_name
        }

        # Print the annotation 
        st.json(annotation)

        # Save to CSV or JSON
        pd.DataFrame([annotation]).to_csv("annotations.csv", mode='a', index=False, header=not os.path.exists("annotations.csv"))


    if os.path.exists("annotations.csv"):
        try:
            df = pd.read_csv("annotations.csv")
            df = df.dropna(subset=["latitude", "longitude"])
            df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
            df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
            df = df.dropna(subset=["latitude", "longitude"])

            if not df.empty:
                df["tooltip"] = df.apply(lambda row:
                    f"{encode_image_base64(row['filepath'])}<br>"
                    f"📷 {row['filename']}<br>🌡 {row['temperature']}°C<br>📏 {row['depth']}m", axis=1)

                st.pydeck_chart(pdk.Deck(
                    map_style='mapbox://styles/mapbox/light-v9',
                    initial_view_state=pdk.ViewState(
                        latitude=df["latitude"].mean(),
                        longitude=df["longitude"].mean(),
                        zoom=3,
                        pitch=50,
                    ),
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=df,
                            get_position='[longitude, latitude]',
                            get_color='[200, 30, 0, 160]',
                            get_radius=100000,
                            pickable=True,
                        ),
                    ],
                    tooltip={"html": "{tooltip}", "style": {"color": "white"}}
                ))
        except Exception as e:
            st.error(f"Error loading annotations: {e}")

    # Check if annotations.csv exists
    csv_file = "annotations.csv"

    if os.path.exists(csv_file):
        with open(csv_file, "rb") as file:
            btn = st.download_button(
                label="📥 Download Annotations CSV",
                data=file,
                file_name="annotations.csv",
                mime="text/csv"
            )
    else:
        st.warning("No annotations available to download yet.")


elif uploaded_file and uploaded_file.type in ["video/mp4", "video/avi"]:
    # Add a slider to control playback speed
    playback_speed = st.sidebar.slider("Playback Speed (Seconds per Frame)", 0.01, 0.5, 0.1, step=0.01)

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 object detection on each frame
        results = model(frame)

        # Draw bounding boxes
        frame = draw_bounding_boxes(frame, results, show_streamlit_output=True)

        # Convert to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the processed frame
        stframe.image(frame, use_container_width=True)

        # Add Delay for slower playback
        time.sleep(playback_speed)

    cap.release()

if st.sidebar.checkbox("🔄 Enable Webcam", key="webcam_key"):

    if st.sidebar.button("🎥 Start Webcam", key="start_webcam_btn"):
        st.session_state["webcam_active"] = True

    if st.sidebar.button("🛑 Stop Webcam", key="stop_webcam_btn"):
        st.session_state["webcam_active"] = False

    # Ensure the state exists
    if "webcam_active" not in st.session_state:
        st.session_state["webcam_active"] = False

    # Run the webcam only if it's active
    if st.session_state["webcam_active"]:
        cam = cv2.VideoCapture(0)
        stframe = st.empty()

        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break

            #results = model(frame, conf=conf_threshold)
            results = model.predict(frame, conf=conf_threshold, verbose=False)
            frame = draw_bounding_boxes(frame, results, show_streamlit_output=True)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            stframe.image(frame, channels="RGB", use_container_width=True)

            # Break if the stop button was clicked during loop
            if not st.session_state["webcam_active"]:
                st.warning("🛑 Webcam stopped.")
                break

        cam.release()


st.markdown("""
        ## Contact

        If you have questions or feedback, feel free to reach out:  
        👉 [alexandra.vultureanu@edu.ucv.ro](mailto:alexandra.vultureanu@edu.ucv.ro)
        """)
