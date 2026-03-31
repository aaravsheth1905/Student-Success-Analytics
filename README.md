Student Success Analytics System

A machine learning–driven academic analytics backend that helps university students monitor attendance, predict academic risk, plan CGPA targets, and interact with an academic AI assistant.

This project combines data engineering, machine learning, and AI APIs to build an intelligent academic decision-support system.

Overview

Universities often enforce strict attendance requirements (typically 75–80%). Students frequently struggle to track whether they can miss lectures without falling into the defaulter list.
This system solves that problem by providing:
	•	Automated attendance report parsing
	•	Attendance analytics and projections
	•	Lecture-miss simulation
	•	Academic risk prediction using machine learning
	•	CGPA trajectory planning
	•	AI academic assistant for student queries


Core Features:

Attendance Report Upload

Students upload their official attendance report PDF.
The system automatically extracts:
Subjects
Lectures conducted
Lectures attended
Attendance percentage
Report start date
Report end date


Merged Subject Analytics:

Some universities split a subject into multiple components (e.g., Theory + Practical). The system intelligently merges these entries and produces a unified attendance view:
Mathematics
Lectures Conducted: 23
Lectures Attended: 22
Attendance: 95.6%

“Can I Miss?” Attendance Simulator:

Students can simulate missing a lecture.

The system calculates:
Current attendance,
Attendance after missing a lecture,
Remaining lectures that can be missed,
Whether the student will remain above the required threshold.

Example:
You cannot miss the next lecture.
Your current attendance is 76.4%.
If you miss the next lecture, it will drop to 72.2%, which is below the required 80%.


Academic Risk Prediction (Machine Learning):

Two ML models predict whether a student is likely to fall below the required attendance.

Models used:
Logistic Regression
Short-term risk prediction
Long-term semester projection

Features used:
Current attendance percentage
Miss ratio
Buffer ratio
Remaining weeks
Weekly lecture load
Required attendance threshold

Example:
You currently have a 14% risk of falling below 80% attendance by the end of the semester.

If you continue missing lectures for the next 3 weeks, your risk may rise to 63%.

CGPA Target Planner:

Students can plan their academic goals.

Inputs:
Current CGPA
Target CGPA
Current semester
Total semesters

The system calculates the SGPA required in upcoming semesters.

Example:
You currently have a CGPA of 8.0.

To reach a target CGPA of 9.0 by semester 8,
you should aim for approximately 9.2 SGPA in the remaining semesters.

AI Academic Assistant:

An integrated AI chatbot powered by Google Gemini.

Capabilities:
Answer academic questions,
Explain technical concepts,
Analyze uploaded files or images,
Help with assignments or study strategies.

Example:
Question: Explain eigenvalues in simple terms.

Answer: Eigenvalues represent how a transformation scales a vector...


System Architecture:

Client
   ↓
FastAPI Backend
   ↓
Database (SQLite)
   ↓
ML Models (Scikit-Learn)
   ↓
Google Gemini API

Components:
FastAPI — backend API framework
SQLAlchemy — database ORM
Scikit-learn — machine learning models
pdfplumber — attendance report parsing
Google Gemini API — AI chatbot

Installation:

Clone repository:

git clone <repository-url>
cd student-risk-ai

Install dependencies:
pip install -r requirements.txt

Start the backend server:
uvicorn backend.main:app --reload

Open the API documentation:
http://127.0.0.1:8000/docs


Example API Endpoints:

POST /auth/register
POST /auth/login

POST /attendance/upload-report
GET  /attendance/merged-subjects

POST /attendance/can-i-miss
POST /attendance/predict-risk

POST /cgpa/planner

POST /chat/ask

Technologies Used:

Python
FastAPI
SQLAlchemy
Scikit-learn
pdfplumber
Google Gemini API

Author:

Aarav Sheth
B.Tech Data Science