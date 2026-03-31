from sqlalchemy import Column, Integer, String, Float, Date
from backend.db.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)

class AttendanceRecord(Base):
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)

    subject = Column(String)
    lectures_conducted = Column(Integer)
    lectures_attended = Column(Integer)
    semester_weeks = Column(Integer)
    attendance_percentage = Column(Float)

    report_start_date = Column(Date, nullable=True)
    report_end_date = Column(Date, nullable=True)