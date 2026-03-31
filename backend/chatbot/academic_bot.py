from google import genai
import os
import base64
import pdfplumber
from fastapi import UploadFile

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL_NAME = "models/gemini-2.5-flash"


def extract_text_from_pdf(upload_file: UploadFile):
    text = ""
    with pdfplumber.open(upload_file.file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text


def academic_chat_response(prompt: str, file: UploadFile = None):

    # -------- CASE 1: No file --------
    if not file:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text

    # -------- CASE 2: PDF --------
    if file.filename.endswith(".pdf"):
        extracted_text = extract_text_from_pdf(file)

        full_prompt = f"""
You are a helpful academic assistant.

Here is the uploaded document content:
{extracted_text}

User Question:
{prompt}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt
        )
        return response.text

    # -------- CASE 3: Image --------
    if file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        file_bytes = file.file.read()

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": file.content_type,
                                "data": base64.b64encode(file_bytes).decode()
                            }
                        }
                    ]
                }
            ]
        )
        return response.text

    # -------- CASE 4: Text file --------
    file_text = file.file.read().decode("utf-8", errors="ignore")

    full_prompt = f"""
You are a helpful academic assistant.

Here is the uploaded file content:
{file_text}

User Question:
{prompt}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt
    )

    return response.text