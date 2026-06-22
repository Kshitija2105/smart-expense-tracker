import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Training data — description → category
TRAINING_DATA = [
    # Food
    ("swiggy order", "Food"), ("zomato dinner", "Food"), ("dominos pizza", "Food"),
    ("mcdonalds burger", "Food"), ("cafe coffee", "Food"), ("restaurant lunch", "Food"),
    ("grocery store", "Food"), ("milk bread eggs", "Food"), ("bigbasket order", "Food"),
    ("blinkit groceries", "Food"), ("dunzo food", "Food"), ("mess fees", "Food"),
    ("canteen lunch", "Food"), ("chai snacks", "Food"), ("instamart grocery", "Food"),

    # Travel
    ("uber ride", "Travel"), ("ola cab", "Travel"), ("metro card recharge", "Travel"),
    ("bus ticket", "Travel"), ("auto rickshaw", "Travel"), ("rapido bike", "Travel"),
    ("flight ticket", "Travel"), ("train ticket", "Travel"), ("irctc booking", "Travel"),
    ("toll tax", "Travel"), ("petrol pump", "Travel"), ("parking fee", "Travel"),
    ("makemytrip hotel", "Travel"), ("redbus ticket", "Travel"), ("fuel diesel", "Travel"),

    # Shopping
    ("amazon order", "Shopping"), ("flipkart purchase", "Shopping"), ("myntra clothes", "Shopping"),
    ("ajio shopping", "Shopping"), ("meesho order", "Shopping"), ("nykaa cosmetics", "Shopping"),
    ("clothes shopping", "Shopping"), ("shoes purchase", "Shopping"), ("electronics gadget", "Shopping"),
    ("headphones bought", "Shopping"), ("furniture purchase", "Shopping"), ("decathlon sports", "Shopping"),

    # Bills
    ("electricity bill", "Bills"), ("water bill", "Bills"), ("wifi recharge", "Bills"),
    ("mobile recharge", "Bills"), ("gas cylinder", "Bills"), ("rent payment", "Bills"),
    ("broadband bill", "Bills"), ("dth recharge", "Bills"), ("society maintenance", "Bills"),
    ("postpaid bill", "Bills"), ("insurance premium", "Bills"), ("emi payment", "Bills"),

    # Health
    ("pharmacy medicine", "Health"), ("doctor consultation", "Health"), ("apollo pharmacy", "Health"),
    ("gym membership", "Health"), ("hospital bill", "Health"), ("diagnostic test", "Health"),
    ("medplus medicine", "Health"), ("yoga class", "Health"), ("health checkup", "Health"),
    ("netmeds order", "Health"), ("1mg medicine", "Health"), ("physiotherapy session", "Health"),

    # Entertainment
    ("netflix subscription", "Entertainment"), ("spotify premium", "Entertainment"),
    ("amazon prime", "Entertainment"), ("movie tickets", "Entertainment"),
    ("bookmyshow booking", "Entertainment"), ("hotstar subscription", "Entertainment"),
    ("youtube premium", "Entertainment"), ("gaming purchase", "Entertainment"),
    ("concert ticket", "Entertainment"), ("disney plus", "Entertainment"),

    # Education
    ("udemy course", "Education"), ("coursera subscription", "Education"),
    ("books purchase", "Education"), ("college fees", "Education"),
    ("tuition fees", "Education"), ("coaching class", "Education"),
    ("stationery pens", "Education"), ("notebook purchase", "Education"),
    ("online class", "Education"), ("exam fees", "Education"),

    # Other
    ("atm withdrawal", "Other"), ("bank charges", "Other"), ("donation charity", "Other"),
    ("gift purchase", "Other"), ("salon haircut", "Other"), ("laundry service", "Other"),
    ("miscellaneous expense", "Other"), ("newspaper subscription", "Other"),
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'category_model.pkl')

def train_model():
    descriptions = [d for d, _ in TRAINING_DATA]
    categories   = [c for _, c in TRAINING_DATA]
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ('clf',   LogisticRegression(max_iter=1000))
    ])
    pipeline.fit(descriptions, categories)
    joblib.dump(pipeline, MODEL_PATH)
    print("Model trained and saved!")
    return pipeline

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return train_model()

def predict_category(description: str) -> str:
    model = load_model()
    return model.predict([description.lower()])[0]