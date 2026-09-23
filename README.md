Student Success Analytics System

A machine learning–driven academic analytics backend that helps university students monitor attendance, predict academic risk, plan CGPA targets, and interact with an academic AI assistant.

This project combines data engineering, machine learning, and AI APIs to build an intelligent academic decision-support system.

Overview

Universities often enforce strict attendance requirements (typically 75–80%). Students frequently struggle to track whether they can miss lectures without falling into the defaulter list.
This system solves that problem by providing:
	•	Automated attendance report parsing
	•	Attendance analytics and projections
	•	Academic risk and scenario prediction using machine learning
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

Academic Risk & Attendance Scenario Simulator (Machine Learning):

Students can check their current academic risk and simulate missing future attendance hours.

The system uses a single semester-risk Logistic Regression model to evaluate:
1. Current risk based on the student's current attendance state.
2. Scenario risk after simulating a specified number of future attendance hours missed ("Hours You Plan to Miss").

Inputs:
- Subject
- Weekly hours
- Semester weeks
- Required attendance percentage
- Hours you plan to miss (`hours_to_miss`)

Features used:
- Current attendance percentage
- Miss ratio
- Buffer ratio
- Attendance gap
- Remaining weeks
- Weekly hours
- Required attendance threshold

The system calculates and reports:
- Current attendance percentage
- Current estimated semester risk
- Hours the student plans to miss
- Projected attendance after those missed hours
- Estimated risk after those missed hours

Example:
Your current attendance in Mobile Application Development is 76.47%. You currently have an estimated 12.4% risk of falling below 80.0% attendance by the end of the semester. If you miss the next 2 hours, your projected attendance will be 68.42% and your estimated risk will increase to 71.8%.


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

POST /attendance/predict-risk

POST /cgpa/planner

POST /chat/ask

## Environment Variables

Create a `.env` file in the project root.

Example:

GOOGLE_API_KEY=YOUR_GEMINI_API_KEY

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
