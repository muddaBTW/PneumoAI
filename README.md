# 🫁 Pneumonia Detection AI System

A full-stack AI application that detects **Pneumonia from Chest X-ray images** using Deep Learning and Transfer Learning.  
The system includes model training, API deployment, and a web frontend.

---

## 🚀 Features

- Deep Learning model trained on medical X-ray dataset
- Transfer Learning using ResNet50
- FastAPI backend for model inference
- Streamlit frontend for user interaction
- Image preprocessing pipeline
- Confidence score and probability output

---

## 🧠 Model Overview

| Component           | Description                                 |
| ------------------- | ------------------------------------------- |
| Base Model          | ResNet50 (ImageNet pretrained)              |
| Task                | Binary Classification (Normal vs Pneumonia) |
| Input Size          | 224 × 224 RGB                               |
| Final Test Accuracy | ~89%                                        |
| Output              | Prediction + Probability + Confidence       |

---

## 📂 Project Structure

```
pneumonia_ai_project/
│
├── model/                # Saved trained model
├── api/                  # FastAPI backend
│   ├── main.py
│   └── schemas.py
├── frontend/             # Streamlit UI
│   └── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer            | Technology         |
| ---------------- | ------------------ |
| Deep Learning    | TensorFlow / Keras |
| Backend API      | FastAPI            |
| Data Validation  | Pydantic           |
| Frontend         | Streamlit          |
| Image Processing | Pillow, NumPy      |

---

## 🛠 Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
uv venv
.venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
uv pip install -r requirements.txt
```

---

## ▶️ Running the Application

### Start Backend API

```bash
uvicorn api.main:app --reload
```

API Docs available at:

```
http://127.0.0.1:8000/docs
```

### Start Frontend

```bash
streamlit run frontend/app.py
```

---

## 🔍 API Response Example

```json
{
  "prediction": "Pneumonia",
  "probability": 0.93,
  "confidence": 93.0
}
```

---

## 📈 Model Performance

| Dataset    | Accuracy |
| ---------- | -------- |
| Training   | ~95%     |
| Validation | ~94–95%  |
| Test       | ~89%     |

---

## 📌 Future Improvements

- Grad-CAM explainability
- Docker containerization
- Cloud deployment
- Model versioning

---

## 👤 Author

**Muddassir**  
AI & Data Science Enthusiast
