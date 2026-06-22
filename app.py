import csv
import io
from flask import Response
import plotly.graph_objects as go
import plotly
import json
from datetime import datetime, date
from collections import defaultdict
from model import predict_category, train_model
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from config import Config
from datetime import date

app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

CATEGORIES = ['Food', 'Travel', 'Shopping', 'Bills', 'Health', 'Entertainment', 'Education', 'Other']

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(255), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(100))
    date = db.Column(db.Date, nullable=False)

class Budget(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(100), nullable=False)
    monthly_limit = db.Column(db.Float, nullable=False)

@app.route('/')
def index():
    expenses = Expense.query.order_by(Expense.date.desc()).all()
    return render_template('index.html', expenses=expenses, categories=CATEGORIES, today=date.today())

@app.route('/add', methods=['POST'])
def add_expense():
    description = request.form['description']
    amount = float(request.form['amount'])
    category = request.form['category']
    date_val = date.fromisoformat(request.form['date'])
    expense = Expense(description=description, amount=amount, category=category, date=date_val)
    db.session.add(expense)
    db.session.commit()
    flash('Expense added successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/delete/<int:id>')
def delete_expense(id):
    expense = Expense.query.get_or_404(id)
    db.session.delete(expense)
    db.session.commit()
    flash('Expense deleted.', 'info')
    return redirect(url_for('index'))

with app.app_context():
    db.create_all()

@app.route('/predict-category', methods=['POST'])
def predict_category_route():
    from flask import jsonify
    description = request.json.get('description', '')
    if description:
        category = predict_category(description)
        return jsonify({'category': category})
    return jsonify({'category': 'Other'})

@app.route('/dashboard')
def dashboard():
    expenses = Expense.query.order_by(Expense.date.asc()).all()

    # Chart 1 — Spending by category (Pie chart)
    category_totals = defaultdict(float)
    for e in expenses:
        category_totals[e.category] += e.amount

    pie_fig = go.Figure(data=[go.Pie(
        labels=list(category_totals.keys()),
        values=list(category_totals.values()),
        hole=0.45,
        pull=[0.05] * len(category_totals),
        marker_colors=['#1A56A0','#2ecc71','#e74c3c','#f39c12','#9b59b6','#1abc9c','#e67e22','#95a5a6'],
        textinfo='label+percent',
        textposition='outside' 
    )])
    pie_fig.update_layout(
        title='Spending by Category',
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=12),
        margin=dict(t=50, b=20, l=20, r=20)
    )  

    # Chart 2 — Daily spending trend (Line chart)
    daily_totals = defaultdict(float)
    for e in expenses:
        day_str = e.date.strftime('%Y-%m-%d') if hasattr(e.date, 'strftime') else str(e.date)[:10]
        daily_totals[day_str] += e.amount
    sorted_dates = sorted(daily_totals.keys())

    line_fig = go.Figure(data=[go.Scatter(
        x=sorted_dates,
        y=[daily_totals[d] for d in sorted_dates],
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(26, 86, 160, 0.1)',
        line=dict(color='#1A56A0', width=3),
        marker=dict(size=10, color='#1A56A0', symbol='circle')
    )])
    line_fig.update_layout(
        title='Daily Spending Trend',
        xaxis_title='Date',
        yaxis_title='Amount (₹)',
        xaxis=dict(type='category', tickangle=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=13),
        margin=dict(t=50, b=60, l=60, r=20)
    )

    # Chart 3 — Top categories bar chart
    sorted_cats = sorted(category_totals.items(), key=lambda x: x[1], reverse=True)
    short_names = {'Entertainment': 'Entertain.', 'Education': 'Education', 
               'Shopping': 'Shopping', 'Travel': 'Travel', 
               'Food': 'Food', 'Bills': 'Bills', 'Health': 'Health', 'Other': 'Other'}

    bar_fig = go.Figure(data=[go.Bar(
        x=[short_names.get(c[0], c[0]) for c in sorted_cats],
        y=[c[1] for c in sorted_cats],
        marker_color=['#1A56A0','#2ecc71','#e74c3c','#f39c12','#9b59b6','#1abc9c','#e67e22','#95a5a6'],
        text=[f'₹{c[1]:.0f}' for c in sorted_cats],
        textposition='outside'
    )])
    bar_fig.update_layout(
        title='Total Spent per Category',
        xaxis_title='',
        yaxis_title='Amount (₹)',
        xaxis=dict(tickangle=0, tickfont=dict(size=11)),
        yaxis=dict(range=[0, max(category_totals.values()) * 1.2]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=12),
        margin=dict(t=50, b=40, l=50, r=20),
        bargap=0.4
    )
    # Summary stats
    total_spent = sum(e.amount for e in expenses)
    total_expenses = len(expenses)
    top_category = max(category_totals, key=category_totals.get) if category_totals else 'N/A'
    
    # Budget summary for dashboard
    all_budgets = Budget.query.all()
    total_budget_limit = sum(b.monthly_limit for b in all_budgets)
    current_month = datetime.now().month
    current_year = datetime.now().year
    total_budget_spent = db.session.query(db.func.sum(Expense.amount)).filter(
        db.extract('month', Expense.date) == current_month,
        db.extract('year', Expense.date) == current_year
    ).scalar() or 0
    budget_percent = round((total_budget_spent / total_budget_limit) * 100, 1) if total_budget_limit > 0 else 0
    
    return render_template('dashboard.html',
                           pie_chart=json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder),
                           line_chart=json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder),
                           bar_chart=json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder),
                           total_spent=total_spent,
                           total_expenses=total_expenses,
                           top_category=top_category,
                           total_budget_limit=total_budget_limit,
                           total_budget_spent=total_budget_spent,
                           budget_percent=budget_percent
                           )

@app.route('/budgets', methods=['GET', 'POST'])
def budgets():
    if request.method == 'POST':
        category = request.form['category']
        limit = float(request.form['limit'])
        existing = Budget.query.filter_by(category=category).first()
        if existing:
            existing.monthly_limit = limit
        else:
            db.session.add(Budget(category=category, monthly_limit=limit))
        db.session.commit()
        flash(f'Budget set for {category}!', 'success')
        return redirect(url_for('budgets'))

    from datetime import datetime
    current_month = datetime.now().month
    current_year = datetime.now().year

    all_budgets = Budget.query.all()
    budget_data = []
    for b in all_budgets:
        spent = db.session.query(db.func.sum(Expense.amount)).filter(
            Expense.category == b.category,
            db.extract('month', Expense.date) == current_month,
            db.extract('year', Expense.date) == current_year
        ).scalar() or 0
        percent = round((spent / b.monthly_limit) * 100, 1)
        status = 'danger' if percent >= 90 else 'warning' if percent >= 70 else 'good'
        budget_data.append({
            'category': b.category,
            'limit': b.monthly_limit,
            'spent': spent,
            'percent': min(percent, 100),
            'status': status
        })

    return render_template('budgets.html', budget_data=budget_data, categories=CATEGORIES)

@app.route('/export')
def export_csv():
    expenses = Expense.query.order_by(Expense.date.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Description', 'Amount', 'Category', 'Date'])
    for e in expenses:
        writer.writerow([e.id, e.description, e.amount, e.category, e.date])
    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=expenses.csv'}
    )

if __name__ == '__main__':
    app.run(debug=True)