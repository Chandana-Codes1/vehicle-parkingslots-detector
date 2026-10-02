import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("Smart Parking Occupancy Counter")
st.write("Upload a parking lot image to count occupied vehicles.")

model = YOLO("best3.pt")

total_spots = st.number_input("Total parking spots in this lot", min_value=1, value=60)

uploaded_file = st.file_uploader("Choose a parking lot image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    results = model.predict(np.array(image), conf=0.4)
    result_img = results[0].plot()

    # Count detected vehicles
    occupied = len(results[0].boxes)
    available = max(total_spots - occupied, 0)
    occupancy_pct = (occupied / total_spots) * 100 if total_spots > 0 else 0

    st.image(result_img, caption="Detected Vehicles", use_container_width=True)

    st.subheader("Parking Status")
    col1, col2, col3 = st.columns(3)
    col1.metric("Occupied", occupied)
    col2.metric("Available", available)
    col3.metric("Occupancy %", f"{occupancy_pct:.1f}%")

    if occupancy_pct >= 90:
        st.error("Lot nearly full!")
    elif occupancy_pct >= 60:
        st.warning("Lot getting busy.")
    else:
        st.success("Plenty of space available.")