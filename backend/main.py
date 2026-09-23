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
from backend.ml.model_loader import final_model
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
        model="models/gemini-3.6-flash",
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


@app.post("/attendance/predict-risk")
def predict_risk(
    subject: str = Form(...),
    weekly_hours: int = Form(...),
    semester_weeks: int = Form(...),
    required_percentage: float = Form(...),
    hours_to_miss: int = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    if hours_to_miss < 0:
        raise HTTPException(
            status_code=400,
            detail="hours_to_miss cannot be negative"
        )

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

    clean_subject = merged[subject_key]["subject"]
    total_conducted = merged[subject_key]["lectures_conducted"]
    total_attended = merged[subject_key]["lectures_attended"]

    if total_conducted <= 0:
        raise HTTPException(
            status_code=400,
            detail="Conducted hours must be greater than zero"
        )

    total_planned = weekly_hours * semester_weeks
    remaining_hours = total_planned - total_conducted

    if hours_to_miss > remaining_hours:
        raise HTTPException(
            status_code=400,
            detail=f"hours_to_miss ({hours_to_miss}) exceeds remaining semester hours ({remaining_hours})"
        )

    # -------------------------
    # CURRENT STATE
    # -------------------------
    current_percentage = round((total_attended / total_conducted) * 100, 2)

    current_features = build_features(
        total_conducted,
        total_attended,
        total_planned,
        weekly_hours,
        required_percentage,
        semester_weeks
    )

    current_prob = float(
        final_model.predict_proba(current_features)[0][1]
    )
    current_risk_percent = round(current_prob * 100, 2)

    # -------------------------
    # SCENARIO STATE (HOURS MISSED)
    # -------------------------
    simulated_conducted = total_conducted + hours_to_miss
    simulated_attended = total_attended

    projected_percentage = round(
        (simulated_attended / simulated_conducted) * 100, 2
    )

    scenario_features = build_features(
        simulated_conducted,
        simulated_attended,
        total_planned,
        weekly_hours,
        required_percentage,
        semester_weeks
    )

    scenario_prob = float(
        final_model.predict_proba(scenario_features)[0][1]
    )
    scenario_risk_percent = round(scenario_prob * 100, 2)

    # -------------------------
    # RESPONSE MESSAGE
    # -------------------------
    base_text = (
        f"Your current attendance in {clean_subject} is {current_percentage}%. "
        f"You currently have an estimated {current_risk_percent}% risk of falling below {required_percentage}% attendance by the end of the semester."
    )

    if hours_to_miss == 0:
        message = (
            f"Your current attendance in {clean_subject} is {current_percentage}%. "
            f"No additional hours missed. Your current estimated semester risk is {current_risk_percent}%."
        )
    else:
        hour_word = "hour" if hours_to_miss == 1 else "hours"
        if scenario_risk_percent > current_risk_percent:
            scenario_text = (
                f"If you miss the next {hours_to_miss} {hour_word}, "
                f"your projected attendance will be {projected_percentage}% "
                f"and your estimated risk will increase to {scenario_risk_percent}%."
            )
        else:
            scenario_text = (
                f"If you miss the next {hours_to_miss} {hour_word}, "
                f"your projected attendance will be {projected_percentage}% "
                f"and your estimated risk will be {scenario_risk_percent}%."
            )
        message = f"{base_text} {scenario_text}"

    return {
        "subject": clean_subject,
        "current_attendance": current_percentage,
        "current_estimated_semester_risk": current_risk_percent,
        "hours_to_miss": hours_to_miss,
        "projected_attendance": projected_percentage,
        "estimated_risk_after_missed_hours": scenario_risk_percent,
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