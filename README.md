# 🐙 GitHub Issue Resolution Time Predictor

Predict the expected resolution time of any **public GitHub issue** using Machine Learning.

This project collects historical GitHub issues from popular open-source repositories, performs feature engineering, trains multiple regression models, and deploys the best-performing model through a Streamlit web application.

---

## 🚀 Live Demo

**Demo:** https://gitub-issue-predictor-lkkyqeupcg9vw7pvuxhhsr.streamlit.app/

---

## 📌 Features

- 🔗 Predict resolution time for any public GitHub issue
- 🤖 Machine Learning-based prediction
- 📊 Repository statistics fetched live from GitHub API
- 🏆 Automatic best model selection
- 📈 Model comparison with multiple algorithms
- 🌐 Interactive Streamlit web application

---

# 🏗 Project Pipeline

```
GitHub API
      │
      ▼
Data Collection
      │
      ▼
Feature Engineering
      │
      ▼
Model Training
      │
      ▼
Model Comparison
      │
      ▼
Streamlit Deployment
```

---

# 📂 Project Structure

```
github-issue-predictor/

├── app.py
├── collect_data.py
├── build_features.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── github_api.py
│   ├── predictor.py
│   ├── feature_builder.py
│   └── styles.py
│
├── models/
│   ├── best_model.joblib
│   └── model_results.csv
│
├── data/
│   ├── raw/
│   └── processed/
```

---

# 📊 Dataset

The model was trained on closed GitHub issues collected from **30 popular open-source repositories** across multiple domains including AI, Machine Learning, Web Development, Programming Languages, Developer Tools, and Cloud Infrastructure.

### Repositories

- Microsoft VS Code
- Facebook React
- TensorFlow
- PyTorch
- Node.js
- FastAPI
- Django
- Flask
- NumPy
- Pandas
- Scikit-learn
- Requests
- Hugging Face Transformers
- LangChain
- OpenAI Python
- Streamlit
- Next.js
- NestJS
- Vite
- Electron
- Go
- Rust
- Kubernetes
- Apache Spark
- Elasticsearch
- Redis
- Ansible
- Home Assistant
- OpenCV
- CPython

---

# ⚙ Feature Engineering

The model uses a rich set of engineered features extracted from GitHub issues.

### Repository Features

- ⭐ Stars
- 🍴 Forks
- 👀 Watchers
- 🐞 Open Issues
- 📦 Repository Size
- 📅 Repository Age
- 🗂 Default Branch
- 🖥 Primary Language
- 📁 Archived Status

### Issue Features

- Title Length
- Body Length
- Word Count
- Number of Labels
- Comments
- Reactions
- URLs
- Mentions
- Code Blocks

### Time Features

- Hour Created
- Day of Week
- Month
- Weekend
- Business Hours

### Author Features

- Author Association
- Bot Detection
- Owner / Member Detection

### Label Features

- Bug
- Enhancement
- Documentation
- Help Wanted
- Good First Issue
- Milestone

---

# 🤖 Models Compared

| Model | MAE | RMSE | R² |
|------|------:|------:|------:|
| 🥇 Random Forest | **1.4088** | **1.8483** | **0.4419** |
| XGBoost | 1.4633 | 1.8646 | 0.4320 |
| HistGradientBoosting | 1.4628 | 1.8658 | 0.4313 |
| Ridge Regression | 1.8064 | 2.1748 | 0.2273 |

---

# 🏆 Best Model

**Random Forest Regressor**

Saved as

```
models/best_model.joblib
```

---

# 🛠 Tech Stack

- Python
- Scikit-learn
- Random Forest
- XGBoost
- Pandas
- NumPy
- GitHub REST API
- Streamlit
- Joblib

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/Mohitkataria361/gitub-issue-predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GITHUB_TOKEN=your_github_token
```

Run the application

```bash
streamlit run app.py
```

---

# 📈 Future Improvements

- SHAP Explainable AI
- Confidence Score
- Support for Private Repositories
- Fine-tuned XGBoost
- Deep Learning-based NLP Features
- Issue Priority Classification
- Predict Whether an Issue Will Close

---

# 👨‍💻 Author

**Mohit Kataria**

B.Tech Information Technology  
Delhi Technological University (DTU)

---

## ⭐ If you found this project useful, consider giving it a star.