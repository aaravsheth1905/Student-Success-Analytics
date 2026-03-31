import pdfplumber
import re
from collections import defaultdict
from datetime import datetime


def extract_attendance_from_pdf(file_path: str):

    subject_data = defaultdict(lambda: {
        "lectures_conducted": 0,
        "lectures_attended": 0
    })

    all_dates = []
    date_pattern = r"\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}"

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:

            text = page.extract_text()

            # ------------------------
            # DATE EXTRACTION
            # ------------------------
            if text:
                matches = re.findall(date_pattern, text)
                for date_str in matches:
                    cleaned = date_str.replace("/", "-").replace(".", "-")
                    for fmt in ["%d-%m-%Y", "%d-%m-%y"]:
                        try:
                            parsed = datetime.strptime(cleaned, fmt)
                            all_dates.append(parsed)
                            break
                        except:
                            continue

            tables = page.extract_tables()

            if not tables:
                continue

            # ------------------------
            # TABLE PROCESSING
            # ------------------------
            for table in tables:
                for row in table:

                    if not row or len(row) < 6:
                        continue

                    subject = row[1].strip() if row[1] else ""
                    attendance_status = row[5].strip() if row[5] else ""

                    # Skip empty subject
                    if not subject:
                        continue

                    # Skip header or label rows
                    if subject.lower() in [
                        "course name",
                        "subject",
                        "course",
                        "name"
                    ]:
                        continue

                    # Only process valid attendance entries
                    if attendance_status.upper() not in ["P", "A"]:
                        continue

                    subject_data[subject]["lectures_conducted"] += 1

                    if attendance_status.upper() == "P":
                        subject_data[subject]["lectures_attended"] += 1

    start_date = min(all_dates).strftime("%d-%m-%Y") if all_dates else None
    end_date = max(all_dates).strftime("%d-%m-%Y") if all_dates else None

    return {
        "report_start_date": start_date,
        "report_end_date": end_date,
        "subjects": subject_data
    }