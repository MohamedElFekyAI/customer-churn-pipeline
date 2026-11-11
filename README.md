# 📊 End-to-End Customer Churn Prediction System

An end-to-end Machine Learning pipeline and REST API to predict customer churn risk for subscription-based services.

## 🚀 Features
- **Data Engineering:** Automated synthetic dataset generation & preprocessing pipeline.
- **Machine Learning:** Random Forest Classifier with feature scaling.
- **Evaluation:** Precision, Recall, Accuracy, and ROC-AUC metrics tracking.
- **API Deployment:** Production-ready FastAPI interface with request validation.
- **Containerization:** Dockerized setup for quick deployment.
- **CI/CD:** Automated testing pipeline using GitHub Actions.

## 🛠️ How to Run Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the ML Pipeline:**
   ```bash
   python src/preprocess.py
   python src/train.py
   python src/evaluate.py
   ```

3. **Launch the FastAPI Server:**
   ```bash
   uvicorn app.main:app --reload
   ```
   Access Swagger UI at `http://127.0.0.1:8000/docs`

## 🐳 Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```
