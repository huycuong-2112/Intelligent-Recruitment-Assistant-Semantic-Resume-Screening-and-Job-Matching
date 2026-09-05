from __future__ import annotations

import re
from typing import Dict, Tuple

# Contact & Identity Patterns (VN & International)
EMAIL_REGEX = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)
PHONE_REGEX = re.compile(
    r'(?:\+?84|0)(?:[ -]?[0-9]){9,10}\b'
)
SOCIAL_URL_REGEX = re.compile(
    r'https?://(?:www\.)?(?:linkedin\.com/in/[a-zA-Z0-9_\-]+|github\.com/[a-zA-Z0-9_\-]+|facebook\.com/[a-zA-Z0-9_.\-]+|[a-zA-Z0-9\-]+\.(?:github\.io|gitlab\.io|dev|app|me))/?',
    re.IGNORECASE
)
NATIONAL_ID_REGEX = re.compile(
    r'(?i)(?:cccd|cmnd|căn cước|cmt)\s*[:\-\s]*([0-9]{9,12})\b'
)
DOB_REGEX = re.compile(
    r'(?i)(?:ngày sinh|d\.?o\.?b|date of birth)\s*[:\-\s]*([0-9]{1,2}[/\-\.][0-9]{1,2}[/\-\.][0-9]{2,4})'
)
EXPLICIT_NAME_REGEX = re.compile(
    r'(?i)(?:họ\s*(?:và)?\s*tên|full\s*name|candidate\s*name)\s*[:\-\s]*([A-ZÀ-Ỹa-zà-ỹ\s]{3,35})(?:\n|$)'
)
HR_CONTACT_REGEX = re.compile(
    r'(?i)(?:liên hệ|ứng tuyển|contact|send cv to|hr contact)\s*[:\-\s]*([^\n]+)'
)


def scrub_resume_pii(text: str) -> Tuple[str, Dict[str, str]]:
    """Strips candidate PII locally before sending text to the cloud LLM."""
    scrubbed = text
    vault: Dict[str, str] = {}

    for idx, match in enumerate(set(EMAIL_REGEX.findall(scrubbed)), 1):
        token = f"[REDACTED_EMAIL_{idx}]"
        vault[token] = match
        scrubbed = scrubbed.replace(match, token)

    for idx, match in enumerate(set(PHONE_REGEX.findall(scrubbed)), 1):
        token = f"[REDACTED_PHONE_{idx}]"
        vault[token] = match
        scrubbed = scrubbed.replace(match, token)

    for idx, match in enumerate(set(SOCIAL_URL_REGEX.findall(scrubbed)), 1):
        token = f"[REDACTED_PORTFOLIO_{idx}]"
        vault[token] = match
        scrubbed = scrubbed.replace(match, token)

    for match in NATIONAL_ID_REGEX.finditer(scrubbed):
        scrubbed = scrubbed.replace(match.group(0), "[REDACTED_NATIONAL_ID]")

    for match in DOB_REGEX.finditer(scrubbed):
        scrubbed = scrubbed.replace(match.group(0), "[REDACTED_DOB]")

    for match in EXPLICIT_NAME_REGEX.finditer(scrubbed):
        scrubbed = scrubbed.replace(match.group(0), "Candidate Profile:\n")

    lines = scrubbed.splitlines()
    if lines:
        first_line = lines[0].strip().strip("#* -")
        words = first_line.split()
        if 2 <= len(words) <= 5 and all(w[0].isupper() for w in words if w.isalpha()):
            if not any(header in first_line.lower() for header in ["curriculum", "resume", "cv", "developer", "engineer"]):
                vault["[CANDIDATE_NAME]"] = first_line
                lines[0] = "[CANDIDATE_NAME]"
                scrubbed = "\n".join(lines)

    return scrubbed, vault


def scrub_jd_pii(text: str, default_company: str | None = None) -> Tuple[str, Dict[str, str]]:
    """Masks internal company names and HR contact info from Job Descriptions."""
    scrubbed = text
    vault: Dict[str, str] = {}

    for idx, match in enumerate(set(EMAIL_REGEX.findall(scrubbed)), 1):
        token = f"[HR_EMAIL_{idx}]"
        vault[token] = match
        scrubbed = scrubbed.replace(match, token)

    for idx, match in enumerate(set(PHONE_REGEX.findall(scrubbed)), 1):
        token = f"[HR_PHONE_{idx}]"
        vault[token] = match
        scrubbed = scrubbed.replace(match, token)

    if default_company and len(default_company.strip()) > 2:
        comp_clean = default_company.strip()
        vault["[CONFIDENTIAL_COMPANY]"] = comp_clean
        pattern = re.compile(re.escape(comp_clean), re.IGNORECASE)
        scrubbed = pattern.sub("[CONFIDENTIAL_COMPANY]", scrubbed)

    for match in HR_CONTACT_REGEX.finditer(scrubbed):
        scrubbed = scrubbed.replace(match.group(0), "[HR_CONTACT_SECTION_REDACTED]")

    return scrubbed, vault