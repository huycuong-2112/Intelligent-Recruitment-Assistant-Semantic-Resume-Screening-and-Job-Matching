import requests

API_BASE_URL = "http://localhost:8000/api/v1"

def call_matching_api(resume_text: str, job_description: str) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/matching/score",
        json={"resume_text": resume_text, "job_description": job_description}
    )
    response.raise_for_status()
    return response.json()