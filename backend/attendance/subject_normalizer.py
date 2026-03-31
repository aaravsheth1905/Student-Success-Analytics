import re


def normalize_subject_name(subject: str) -> str:
    """
    Removes section identifiers like T1, P1, U1, batch codes, etc.
    Returns base subject name.
    """

    # Remove batch codes like -J1, -J2
    subject = re.sub(r"-J\d+", "", subject)

    # Remove course codes like -BTDS
    subject = re.sub(r"-[A-Z]+$", "", subject)

    # Remove T1, P1, U1 patterns
    subject = re.sub(r"[TPU]\d+", "", subject)

    # Remove extra spaces
    subject = subject.strip()

    return subject