Vehicle Detection & Parking Occupancy Counter

A YOLOv8-based object detection pipeline that detects vehicles (car, bus, truck, motorbike, van, threewheel) and estimates parking lot occupancy from an uploaded image.

#features
- Fine-tuned YOLOv8n on a 3000-image vehicle dataset (6 classes)
- Achieved 97.7% mAP50 on clean validation data
- Robustness tested against blur, noise, and low-light degradation
- Mitigated noise vulnerability via augmented retraining (mAP50 recovered from 17.7% to 88.7%)
- Streamlit web app for live parking occupancy counting

#TOOLS

Python, Ultralytics YOLOv8, OpenCV, Streamlit

\`\`\`bash
pip install ultralytics streamlit
streamlit run vehicle-detector-demo/app.py
\`\`\`

## Results Summary
| Condition | mAP50 | mAP50-95 |
|---|---|---|
| Clean | 0.977 | 0.908 |
| Blur | 0.801 | 0.687 |
| Noise (before fix) | 0.177 | 0.102 |
| Noise (after retraining) | 0.887 | — |
| Dark | 0.976 | 0.872 |
