# 🎓 StudentIQ

### Academic Performance & Early Intervention Platform

StudentIQ is a **machine learning-powered academic analytics platform** built with Python and Streamlit. It helps teachers analyze student performance, identify students who may require academic support, understand subject-wise weaknesses, and generate actionable reports.

## 🚀 Features

- 📂 Upload student data in **CSV or Excel** format
- 🧹 Automatic data cleaning and preprocessing
- 📊 Class-level performance analysis
- 📚 Subject-wise performance insights
- 👤 Individual student profiles
- ⚠️ Identify students requiring academic support
- 🤖 Random Forest-based performance prediction
- 💡 Generate personalized improvement recommendations
- 📈 Interactive charts and visualizations
- 📄 Export academic reports to Excel
- ⚡ Optimized data processing for large datasets

## 🧠 Machine Learning

StudentIQ uses **Random Forest Regression** to estimate expected subject performance based on other available academic scores.

The difference between the **actual and expected score** is used to identify performance gaps and generate smart academic signals that can help highlight students who may need additional attention.

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Plotly
- **Web Framework:** Streamlit
- **Reports:** OpenPyXL

## 📁 Project Structure

StudentIQ/
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
└── student.csv

##⚙️ Run Locally

1. Clone the repository
git clone https://github.com/YOUR_USERNAME/studentiq.git
cd studentiq
2. Install dependencies
pip install -r requirements.txt
3. Run the application
python -m streamlit run app.py

Open http://localhost:8501 in your browser.

## 📊 How It Works
Upload Student Data
        ↓
Data Preprocessing
        ↓
Performance Analysis
        ↓
ML-Based Academic Signals
        ↓
Risk & Support Identification
        ↓
Insights & Recommendations
        ↓
Excel Report

##🎯 Use Case

StudentIQ is designed for teachers, academic mentors, and educational institutions to quickly understand student performance and identify learners who may benefit from additional academic support.

🌐 Deployment

The application can be deployed using Streamlit Community Cloud for easy access through a web browser.
