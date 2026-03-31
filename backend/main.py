from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from backend.db.database import engine, Base, get_db
from backend.db.models import User, AttendanceRecord
from backend.chatbot.academic_bot import academic_chat_response
from backend.ml.feature_engineering import build_features
from backend.ml.model_loader import final_model, short_model
from google import genai
import os
import re
import json
import base64

app = FastAPI()

Base.metadata.create_all(bind=engine)

SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
security = HTTPBearer()


def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


def clean_subject_name(name: str):
    name = re.sub(r"(T\d+|P\d+|U\d+|J\d+)", "", name)
    name = re.sub(r"-BTDS", "", name)
    return name.strip()


@app.post("/auth/register")
def register(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(email=email, password=hash_password(password))
    db.add(user)
    db.commit()
    return {"message": "User registered successfully"}


@app.post("/auth/login")
def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid credentials")

    if not verify_password(password, user.password):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_access_token({"sub": user.email})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/chat/ask")
async def ask_chatbot(
    prompt: str = Form(...),
    file: UploadFile = File(None),
    current_user: User = Depends(get_current_user)
):
    file_path = None

    if file:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

    response = academic_chat_response(prompt, file_path)

    return {
        "question": prompt,
        "answer": response
    }


@app.post("/attendance/upload")
async def upload_attendance_report(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    file_bytes = await file.read()
    encoded = base64.b64encode(file_bytes).decode()

    prompt = """
Extract attendance data from this document.
Return strictly valid JSON only.

Format:
{
  "report_start_date": "...",
  "report_end_date": "...",
  "subjects": [
    {
      "subject": "...",
      "lectures_conducted": number,
      "lectures_attended": number
    }
  ]
}
"""

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": encoded
                        }
                    }
                ]
            }
        ]
    )

    raw_text = response.text.strip()
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)

    if not json_match:
        raise HTTPException(status_code=500, detail="AI did not return valid JSON")

    extracted = json.loads(json_match.group(0))

    subjects_output = []
    db.query(AttendanceRecord).delete()
    db.commit()


    for item in extracted["subjects"]:
        subject = item["subject"]
        conducted = item["lectures_conducted"]
        attended = item["lectures_attended"]

        record = AttendanceRecord(
            subject=subject,
            lectures_conducted=conducted,
            lectures_attended=attended
        )

        db.add(record)

        subjects_output.append({
            "subject": subject,
            "lectures_conducted": conducted,
            "lectures_attended": attended,
            "lectures_missed": conducted - attended,
            "attendance_percentage": round((attended / conducted) * 100, 2)
        })

    db.commit()

    return {
        "message": "Attendance uploaded successfully",
        "report_start_date": extracted["report_start_date"],
        "report_end_date": extracted["report_end_date"],
        "subjects": subjects_output
    }

def canonical_subject_key(name: str):
    name = name.lower()
    name = re.sub(r"(t\d+|p\d+|u\d+|j\d+)", "", name)
    name = re.sub(r"-btds", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name

@app.get("/attendance/merged-subjects")
def get_merged_subjects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    records = db.query(AttendanceRecord).all()

    merged = {}

    for record in records:

        key = canonical_subject_key(record.subject)

        if key not in merged:
            clean_display = re.sub(r"(T\d+|P\d+|U\d+|J\d+)", "", record.subject)
            clean_display = re.sub(r"-BTDS", "", clean_display).strip()

            merged[key] = {
                "subject": clean_display,
                "lectures_conducted": 0,
                "lectures_attended": 0
            }

        merged[key]["lectures_conducted"] += record.lectures_conducted
        merged[key]["lectures_attended"] += record.lectures_attended

    result = []

    for data in merged.values():
        conducted = data["lectures_conducted"]
        attended = data["lectures_attended"]
        missed = conducted - attended
        percentage = round((attended / conducted) * 100, 2) if conducted > 0 else 0

        result.append({
            "subject": data["subject"],
            "lectures_conducted": conducted,
            "lectures_attended": attended,
            "lectures_missed": missed,
            "attendance_percentage": percentage
        })

    return {"merged_subjects": result}


@app.post("/attendance/can-i-miss")
def can_i_miss(
    subject: str = Form(...),
    weekly_hours: int = Form(...),
    semester_weeks: int = Form(...),
    required_percentage: float = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    records = db.query(AttendanceRecord).all()

    merged = {}

    for r in records:

        key = canonical_subject_key(r.subject)

        if key not in merged:
            clean_display = re.sub(r"(T\d+|P\d+|U\d+|J\d+)", "", r.subject)
            clean_display = re.sub(r"-BTDS", "", clean_display).strip()

            merged[key] = {
                "subject": clean_display,
                "lectures_conducted": 0,
                "lectures_attended": 0
            }

        merged[key]["lectures_conducted"] += r.lectures_conducted
        merged[key]["lectures_attended"] += r.lectures_attended

    subject_key = canonical_subject_key(subject)

    if subject_key not in merged:
        raise HTTPException(status_code=404, detail="Subject not found")

    total_conducted = merged[subject_key]["lectures_conducted"]
    total_attended = merged[subject_key]["lectures_attended"]

    total_planned = weekly_hours * semester_weeks

    current_missed = total_conducted - total_attended

    max_miss_allowed = total_planned - int((required_percentage / 100) * total_planned)

    remaining = max_miss_allowed - current_missed

    current_percentage = round((total_attended / total_conducted) * 100, 2)

    future_percentage = round((total_attended / (total_conducted + 1)) * 100, 2)

    if future_percentage >= required_percentage:

        output = (
            f"You can miss the next hour. Your current attendance is {current_percentage}%. "
            f"If you miss, it will become {future_percentage}%. "
            f"You can still miss {remaining} more hours while maintaining {required_percentage}% attendance."
        )

    else:

        output = (
            f"You cannot miss the next hour. Your current attendance is {current_percentage}%. "
            f"If you miss, it will drop to {future_percentage}% which is below {required_percentage}%."
        )

    return {
        "subject": subject,
        "total_planned_hours": total_planned,
        "lectures_conducted": total_conducted,
        "lectures_attended": total_attended,
        "lectures_missed": current_missed,
        "current_percentage": current_percentage,
        "remaining_hours_you_can_miss": remaining,
        "output": output
    }

@app.post("/attendance/predict-risk")
def predict_risk(
    subject: str = Form(...),
    weekly_hours: int = Form(...),
    semester_weeks: int = Form(...),
    required_percentage: float = Form(...),
    K: int = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    records = db.query(AttendanceRecord).all()

    merged = {}

    for r in records:

        key = canonical_subject_key(r.subject)

        if key not in merged:
            clean_display = re.sub(r"(T\d+|P\d+|U\d+|J\d+)", "", r.subject)
            clean_display = re.sub(r"-BTDS", "", clean_display).strip()

            merged[key] = {
                "subject": clean_display,
                "lectures_conducted": 0,
                "lectures_attended": 0
            }

        merged[key]["lectures_conducted"] += r.lectures_conducted
        merged[key]["lectures_attended"] += r.lectures_attended

    subject_key = canonical_subject_key(subject)

    if subject_key not in merged:
        raise HTTPException(status_code=404, detail="Subject not found")

    total_conducted = merged[subject_key]["lectures_conducted"]
    total_attended = merged[subject_key]["lectures_attended"]

    total_planned = weekly_hours * semester_weeks

    features = build_features(
        total_conducted,
        total_attended,
        total_planned,
        weekly_hours,
        required_percentage,
        semester_weeks,
        K
    )

    final_prob = float(final_model.predict_proba(features)[0][1])
    short_prob = float(short_model.predict_proba(features)[0][1])

    final_percent = round(final_prob * 100, 2)
    short_percent = round(short_prob * 100, 2)

    current_percentage = round((total_attended / total_conducted) * 100, 2)

    message = (
        f"Your current attendance in {subject} is {current_percentage}%. "
        f"You currently have {final_percent}% risk of falling below {required_percentage}% attendance by the end of the semester. "
        f"However, if lectures are missed continuously for the next {K} weeks, "
        f"the risk of falling below the required attendance increases to {short_percent}%."
    )

    return {
        "subject": subject,
        "output": message
    }

@app.post("/attendance/simulate")
def simulate_attendance(
    subject: str = Form(...),
    lectures_to_miss: int = Form(...),
    weekly_hours: int = Form(...),
    semester_weeks: int = Form(...),
    required_percentage: float = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    records = db.query(AttendanceRecord).all()

    merged = {}

    for r in records:

        key = canonical_subject_key(r.subject)

        if key not in merged:
            clean_display = re.sub(r"(T\d+|P\d+|U\d+|J\d+)", "", r.subject)
            clean_display = re.sub(r"-BTDS", "", clean_display).strip()

            merged[key] = {
                "subject": clean_display,
                "lectures_conducted": 0,
                "lectures_attended": 0
            }

        merged[key]["lectures_conducted"] += r.lectures_conducted
        merged[key]["lectures_attended"] += r.lectures_attended

    subject_key = canonical_subject_key(subject)

    if subject_key not in merged:
        raise HTTPException(status_code=404, detail="Subject not found")

    total_conducted = merged[subject_key]["lectures_conducted"]
    total_attended = merged[subject_key]["lectures_attended"]

    current_percentage = round((total_attended / total_conducted) * 100, 2)

    new_conducted = total_conducted + lectures_to_miss
    new_attended = total_attended

    future_percentage = round((new_attended / new_conducted) * 100, 2)

    total_planned = weekly_hours * semester_weeks

    features = build_features(
        new_conducted,
        new_attended,
        total_planned,
        weekly_hours,
        required_percentage,
        semester_weeks,
        1
    )

    risk_probability = float(final_model.predict_proba(features)[0][1])
    risk_percent = round(risk_probability * 100, 2)

    if future_percentage >= required_percentage:

        message = (
            f"If you miss {lectures_to_miss} lectures of {subject}, "
            f"your attendance will change from {current_percentage}% to {future_percentage}%. "
            f"You will still remain above the required {required_percentage}% attendance. "
            f"The predicted risk of falling below the threshold later in the semester is {risk_percent}%."
        )

    else:

        message = (
            f"If you miss {lectures_to_miss} lectures of {subject}, "
            f"your attendance will drop from {current_percentage}% to {future_percentage}%. "
            f"This will place you below the required {required_percentage}% attendance. "
            f"The predicted risk of remaining below the threshold is {risk_percent}%."
        )

    return {
        "subject": subject,
        "output": message
    }

@app.post("/cgpa/plan")
def cgpa_plan(
    current_cgpa: float = Form(...),
    target_cgpa: float = Form(...),
    semesters_completed: int = Form(...),
    total_semesters: int = Form(...)
):

    remaining_semesters = total_semesters - semesters_completed

    if remaining_semesters <= 0:
        raise HTTPException(status_code=400, detail="Invalid semester values")

    points_so_far = current_cgpa * semesters_completed
    required_points = target_cgpa * total_semesters
    points_needed = required_points - points_so_far

    required_average = points_needed / remaining_semesters

    balanced_plan = []
    aggressive_plan = []

    if required_average <= 10:

        balanced_sgpa = round(required_average, 2)

        for i in range(remaining_semesters):
            balanced_plan.append({
                "semester": semesters_completed + i + 1,
                "target_sgpa": balanced_sgpa
            })

        remaining_points = points_needed

        for i in range(remaining_semesters):

            if i < 2:
                sgpa = min(10, round(required_average + 0.5, 2))
            else:
                sgpa = round((remaining_points / (remaining_semesters - i)), 2)

            aggressive_plan.append({
                "semester": semesters_completed + i + 1,
                "target_sgpa": min(10, sgpa)
            })

            remaining_points -= min(10, sgpa)

        message = (
            f"You currently have a CGPA of {current_cgpa}. "
            f"To reach your target CGPA of {target_cgpa} by the end of "
            f"{total_semesters} semesters, you should aim for an average "
            f"SGPA of about {round(required_average,2)} in the remaining "
            f"{remaining_semesters} semesters."
        )

    else:

        max_possible_points = points_so_far + (10 * remaining_semesters)
        max_possible_cgpa = max_possible_points / total_semesters

        message = (
            f"Reaching a CGPA of {target_cgpa} is not possible. "
            f"Even if you score a perfect 10 SGPA in all the remaining "
            f"{remaining_semesters} semesters, the highest CGPA you can "
            f"reach is approximately {round(max_possible_cgpa,2)}."
        )

        for i in range(remaining_semesters):
            balanced_plan.append({
                "semester": semesters_completed + i + 1,
                "target_sgpa": 10
            })

            aggressive_plan.append({
                "semester": semesters_completed + i + 1,
                "target_sgpa": 10
            })

    return {
        "current_cgpa": current_cgpa,
        "target_cgpa": target_cgpa,
        "remaining_semesters": remaining_semesters,
        "balanced_plan": balanced_plan,
        "aggressive_plan": aggressive_plan,
        "output": message
    }