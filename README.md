

# 🛰️ GeoChange-AI
### Satellite Infrastructure Change Detection Using Deep Learning

**GeoChange-AI** is an end-to-end computer vision project that detects infrastructure changes between satellite images captured at different points in time.

Given two satellite images of the same geographical location, the system identifies changed regions through pixel-level segmentation and visualizes the detected changes using an interactive Streamlit dashboard.

The project uses a **Siamese U-Net architecture with a shared MobileNetV2 encoder**, implemented using TensorFlow and Keras.

---

## 🎯 Problem Statement

Infrastructure changes such as new buildings, construction activity and demolition can alter geographical landscapes over time.

Manually comparing large numbers of satellite images is time-consuming and difficult.

GeoChange-AI explores how deep learning can automate this process by comparing two satellite images and identifying areas where changes have occurred.

### Proposed Solution

The system takes:

- Image A — Satellite image captured earlier.
- Image B — Satellite image captured later.

It produces:

- A pixel-level change probability map.
- A binary change mask.
- Number of detected changed regions.
- Percentage of the image classified as changed.
- An overlay highlighting detected regions on the later image.

---

## 🧠 Model Architecture

GeoChange-AI uses a **Siamese U-Net with a shared MobileNetV2 encoder**.

Both input images pass through the same encoder, allowing the model to extract comparable features from each image.

The absolute difference between corresponding feature maps is calculated to identify potential changes.

These difference features are passed through a U-Net-style decoder to generate a pixel-level change map.

### Architecture Pipeline

1. Two satellite images are provided as input.
2. A shared MobileNetV2 encoder extracts multi-scale features.
3. Absolute differences are calculated between corresponding feature maps.
4. A U-Net decoder combines the difference features.
5. A sigmoid activation produces the final change-probability map.
6. Thresholding and connected-component filtering generate the final binary mask.

**Architecture details:**

| Component | Configuration |
|---|---|
| Framework | TensorFlow / Keras |
| Architecture | Siamese U-Net |
| Encoder | MobileNetV2 |
| Encoder weights during training | ImageNet |
| Input size | 256 × 256 × 3 per image |
| Output | 256 × 256 × 1 |
| Parameters | 8,058,977 |
| Loss experiments | BCE and BCE + Dice |
| Final application model | BCE + Dice |

---

## 📂 Dataset

**Dataset:** LEVIR-CD

LEVIR-CD is a building-change detection dataset containing pairs of satellite images and corresponding binary change masks.

| Split | Original Samples |
|---|---:|
| Training | 7,120 |
| Validation | 1,024 |
| Testing | 2,048 |
| Total | 10,192 |

Image dimensions: 256 × 256 pixels.

Approximate spatial resolution: 0.5 m/pixel.

### Data Quality Checks

Before training, dataset quality checks were performed.

These included:

- Exact duplicate image detection.
- Cross-split duplicate identification.
- Duplicate-label consistency checks.
- Removal of identified problematic training samples.
- Creation of a deterministic training-cleaning manifest.

The cleaned training split contains **7,095 samples**.

Validation and test splits were kept unchanged.

---

## 📊 Experimental Results

Two loss configurations were investigated:

1. Binary Cross-Entropy (BCE).
2. Binary Cross-Entropy + Dice Loss.

The final application uses the BCE + Dice model.

### Independent Test Results

The final test evaluation was performed using frozen model settings selected before accessing the test results.

| Metric | BCE + Dice | BCE-only |
|---|---:|---:|
| Precision | 0.9019 | 0.9085 |
| Recall | 0.8952 | 0.8935 |
| F1-score | 0.8985 | 0.9009 |
| IoU | 0.8158 | 0.8197 |
| Pixel accuracy | 0.9897 | 0.9900 |

### Final Application Settings

| Setting | Value |
|---|---:|
| Selected model | BCE + Dice |
| Probability threshold | 0.30 |
| Minimum connected-region area | 50 pixels |

The settings were selected using the validation split and frozen before final test evaluation.

**Note:** BCE-only achieved a slightly higher test F1 point estimate. The paired bootstrap confidence interval for the F1 difference included zero narrowly, so the test results do not establish a clear advantage for either loss configuration. BCE + Dice remains the preselected application model.

---

## 🔍 Additional Analysis

Beyond training and evaluation, several experiments were conducted.

### 1. Loss Function Comparison

Compared BCE-only and BCE + Dice using:

- Precision
- Recall
- F1-score
- Intersection over Union
- Empty-image false-alarm rate
- Region-count accuracy

### 2. Image Misalignment Analysis

Evaluated model sensitivity to spatial misalignment between input images.

Artificial shifts of 1, 3 and 5 pixels were introduced.

Both models showed degradation as misalignment increased.

This highlights the importance of image registration in satellite change detection.

### 3. Error Analysis

Investigated false-positive and false-negative predictions.

Observed error patterns included:

- Missed individual buildings.
- Partially detected large structures.
- False-positive regions in areas without labelled changes.
- Difficulty identifying subtle changes.

### 4. Change Intensity Analysis

Test performance was also examined across different ground-truth change-area categories.

The analysis showed that performance varied depending on the amount of change present in an image crop.

---

## 🖥️ Streamlit Application

GeoChange-AI includes an interactive Streamlit interface.

### Features

- Upload two satellite images.
- Select prepared demonstration pairs.
- Adjust detection threshold.
- Adjust minimum region area.
- Generate binary change masks.
- Visualize detected regions on the later image.
- Display change-probability heatmaps.
- Calculate changed-region count.
- Calculate changed-area percentage.
- Display inference time.
- Compare predictions with available demonstration ground-truth masks.

### Demonstration Examples

The project includes six locally prepared examples:

| Example | Purpose |
|---|---|
| Moderate change 1 | Moderate infrastructure change |
| Moderate change 2 | Another moderate-change example |
| Large change | Extensive infrastructure change |
| Small change | Relatively limited change |
| No change | Correctly predicted empty mask |
| Honest failure | Demonstrates a false-positive limitation |

The demonstration images are not redistributed in this repository because their dataset redistribution terms have not been established. They can be generated locally using the project scripts and the permitted dataset files.

---

## ⚙️ Installation

### Requirements

- Python 3.10
- TensorFlow 2.21.0
- Keras 3.x
- NumPy
- OpenCV
- Pillow
- Streamlit
- PyArrow

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/GeoChange-AI.git

cd GeoChange-AI
```

### 2. Create a Conda environment

```bash
conda create -n geochange python=3.10 -y

conda activate geochange
```

### 3. Install dependencies

```bash
pip install -r app/requirements.txt
```

### 4. Prepare model files

Place the exported model files in:

```text
app/model/
├── geochange_v1.weights.h5
├── settings.json
└── EXPORT_INFO.txt
```

The trained weights are distributed separately if they are not included in the repository.

### 5. Run the application

From the project root:

```bash
streamlit run app/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

The application supports CPU inference. A GPU is not required for deployment.

---

## 📁 Project Structure

```text
GeoChange-AI/
│
├── app/
│   ├── model/
│   │   ├── geochange_v1.weights.h5
│   │   ├── settings.json
│   │   └── EXPORT_INFO.txt
│   │
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── demo_pairs/              # Generated locally
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── raw/                     # Dataset files, not committed
│   └── processed/
│
├── experiments/
│   ├── results/
│   └── runs/
│
├── scripts/
│   ├── export_for_app.py
│   ├── make_demo_pairs.py
│   ├── eval_val_both.py
│   ├── misalignment.py
│   └── step8_val_analysis.py
│
├── src/
│   ├── model.py
│   ├── data.py
│   ├── train.py
│   ├── infer.py
│   ├── losses.py
│   ├── metrics.py
│   └── postprocess.py
│
├── tests/
│
├── .gitignore
└── README.md
```

---

## ⚠️ Limitations

GeoChange-AI is an independent academic portfolio project and should not be treated as an operational geospatial intelligence system.

Current limitations include:

- Training and evaluation use LEVIR-CD building-change imagery.
- Performance on other satellite sensors and geographical regions has not been independently validated.
- Spatial misalignment can reduce prediction quality.
- Small infrastructure changes may be missed.
- False-positive detections can occur.
- Input images are resized to 256 × 256 pixels, which may reduce small-object visibility.
- Model performance depends on image quality, registration and similarity to the training data.

The project demonstrates a change-detection workflow, not a fully validated real-world monitoring service.

---

## 🛠️ Technology Stack

- Python
- TensorFlow
- Keras
- MobileNetV2
- U-Net
- NumPy
- Pandas
- OpenCV
- PyArrow
- Scikit-learn
- Matplotlib
- Streamlit

---

## 👩‍💻 Project Information

**Developed by:** RUSHDA ASLLAMI

**Project:** GeoChange-AI

**Category:** Computer Vision | Deep Learning | Remote Sensing

**Purpose:** Academic and portfolio project.

Developed as an exploration of deep learning-based satellite infrastructure change detection, model evaluation and CPU-based deployment.

---

## Acknowledgements

The project uses the LEVIR-CD dataset for building-change detection research.

Please refer to the original dataset publication and its terms when obtaining or using the dataset.

-----------

---

**GeoChange-AI — From Satellite Images to Infrastructure Change Insights.**
