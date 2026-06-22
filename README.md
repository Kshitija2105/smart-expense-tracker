# 💰 Smart Expense Tracker with AI Insights

A full-stack web application for tracking personal expenses with ML-powered 
auto-categorisation and an interactive analytics dashboard.

## 🚀 Features
- **AI Auto-Categorisation** — ML model predicts expense category from description in real time
- **Interactive Dashboard** — Pie chart, bar chart, and spending trend visualisations
- **Budget Tracker** — Set monthly budgets per category with colour-coded alerts
- **CSV Export** — Download expense history as CSV

## 🛠️ Tech Stack
- **Backend:** Python, Flask, SQLAlchemy
- **Database:** SQLite
- **ML:** scikit-learn (TF-IDF + Logistic Regression)
- **Visualisation:** Plotly
- **Frontend:** HTML, CSS, JavaScript

## ⚙️ Setup & Run
```bash
git clone https://github.com/YOUR_USERNAME/smart-expense-tracker.git
cd smart-expense-tracker
pip install flask flask-sqlalchemy scikit-learn pandas plotly joblib
python3 app.py
```
Open http://127.0.0.1:5000
