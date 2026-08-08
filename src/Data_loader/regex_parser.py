import re
import json

def replace_words_with_nums(text):

    mapping = {
        'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    
    for word, digit in mapping.items():
        text = re.sub(r'\b' + word + r'\b', digit, text, flags=re.IGNORECASE)

    return text


def extract_skills_regex(text):
    pattern = r'(?:Skills|Technical Skills|Core Competencies)[:\s]*\n?(.*?)(?:\n\n|\n[A-Z][a-z]+:|\Z)'
    match_skills = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

    if match_skills:
        skills = match_skills.group(1)
        skills = re.split(r'[,•\n|]+', skills)

        return [s.strip() for s in skills if len(s.strip()) < 30]
    
    return []

def extract_experience_regex(text):
    patterns = [
        r'(\d+)\+?\s*(?:years?)\s*(?:of\s*)?(?:experience)?',
        r'(?:experience)[:\s]*(\d+)\+?\s*(?:years?)'
    ]

    text = replace_words_with_nums(text)

    exp_years = []
    for pattern in patterns:
        match_exp = re.findall(pattern, text, re.IGNORECASE)
        for m in match_exp:
            try:
                exp_years.append(int(m))
            except ValueError:
                continue
                
    if exp_years:
        valid_years = [y for y in exp_years if y <= 30]
        return max(valid_years)
    return 0

def extract_education_regex(text):
    degrees = {
        "Doctorate/PhD": r'\b(?:Ph\.?D|Doctor of Philosophy)\b',
        "Master": r'\b(?:Master|M\.S|M\.Sc|M\.A)\b',
        "Bachelor": r'\b(?:Bachelor|B\.S|B\.Sc|B\.A)\b',
        "Diploma/Associate": r'\b(?:Diploma|Associate)\b'
    }
    
    for degree_name, pattern in degrees.items():
        if re.search(pattern, text, re.IGNORECASE):
            return degree_name
            
    return "Bachelor"

def process_json(json_input, json_output):
    with open(json_input, 'r', encoding='utf-8') as f:
        resumes_list = json.load(f)
        
    extracted_resumes = []

    for cv in resumes_list:
        cv_id = cv.get("id")
        filename = cv.get("filename")
        
        cv_text = cv.get("text", "") 
        
        extracted_skills = extract_skills_regex(cv_text)
        extracted_exp = extract_experience_regex(cv_text)
        extracted_edu = extract_education_regex(cv_text)
        
        extracted_resumes.append({
            "id": cv_id,
            "filename": filename,
            "skills": extracted_skills,
            "experience_years": extracted_exp,
            "education": extracted_edu
        })
        
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(extracted_resumes, f, indent=2, ensure_ascii=False)
        

input_json = "../../Data/Processed/cleaned_resumes.json"
output_json = "../../Data/Processed/analyzed_resumes.json"

process_json(input_json, output_json)
