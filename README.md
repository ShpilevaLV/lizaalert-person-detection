# Person Detection for LizaAlert (Capstone Project)

**Objective:** Automate detection of missing persons in drone images using YOLOv8 to reduce manual review time.

**Dataset:** Real-world search-and-rescue imagery (1,330 images, ~1,700 annotated persons) from LizaAlert.  
*The dataset cannot be publicly shared due to privacy concerns.*

**Model:** YOLOv8m fine-tuned for 100 epochs with heavy augmentation (mosaic, mixup, copy-paste).  
**Baseline (vanilla YOLOv8m):** mAP@0.5 ≈ 0.000 → **Fine-tuned:** mAP@0.5 = 0.770, Recall = 0.775.

**Key results:**
- mAP@0.5 = 0.770
- Recall = 0.775
- Precision = 0.867
- Inference speed: ~16 ms/image on Tesla T4 (well within real-time requirement for edge deployment).

**Pipeline:** EDA → Training → Validation → Prediction visualization → Interpretability (LIME, SHAP, SOM).

**Repository structure:**
- `capstone_person_detection.ipynb` – full pipeline
- `best.pt` – trained model weights
- `images/` – example outputs (predictions, LIME, SHAP, SOM)
- `README.md` – this file

**Authors:** Valid Amavi, Vasenyova Valeria, Lina Shpileva – 2026
