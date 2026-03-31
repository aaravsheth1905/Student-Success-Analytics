import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

final_model = joblib.load(os.path.join(BASE_DIR, "models/final_risk_model.pkl"))
short_model = joblib.load(os.path.join(BASE_DIR, "models/short_term_model.pkl"))