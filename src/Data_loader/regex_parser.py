import re
import json
import os


def replace_words_with_nums(text):
    mapping = {
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10'
    }

    for word, digit in mapping.items():
        text = re.sub(
            r'\b' + word + r'\b',
            digit,
            text,
            flags=re.IGNORECASE
        )

    return text


def extract_skills_regex(text):
    pattern = (
        r'(?:Skills|Technical Skills|Core Competencies)'
        r'[:\s]*\n?(.*?)'
        r'(?:\n\n|\n[A-Z][a-z]+:|\Z)'
    )

    match_skills = re.search(
        pattern,
        text,
        re.IGNORECASE | re.DOTALL
    )

    if match_skills:
        skills = match_skills.group(1)
        skills = re.split(r'[,•\n|]+', skills)

        return [
            s.strip()
            for s in skills
            if len(s.strip()) < 30
        ]

    return []


def extract_experience_regex(text):

    # ============================================================
    # 1. Ưu tiên số năm kinh nghiệm được CV khai báo trực tiếp
    # ============================================================

    explicit_patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?experience',
        r'(?:experience)[:\s]*(\d+)\+?\s*(?:years?|yrs?)'
    ]

    text = replace_words_with_nums(text)

    exp_years = []

    for pattern in explicit_patterns:
        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE
        )

        for match in matches:
            try:
                exp_years.append(int(match))
            except ValueError:
                continue

    # Nếu CV nói rõ số năm kinh nghiệm → ưu tiên giá trị này
    if exp_years:
        return max(exp_years)

    # ============================================================
    # 2. Fallback: tìm các khoảng thời gian làm việc
    # ============================================================

    month_pattern = (
        r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|'
        r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|'
        r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    )

    # Ví dụ:
    # May 2011 to November 2014
    # March 2017 - June 2020
    date_range_patterns = [

        # Month Year -> Month Year/Current
        rf'({month_pattern})\s+(\d{{4}})\s*'
        rf'(?:to|-|–)\s*'
        rf'({month_pattern})?\s*'
        rf'(\d{{4}}|Current|Present)',

        # MM/YYYY -> MM/YYYY/Current
        r'(\d{1,2})/(\d{4})\s*'
        r'(?:to|-|–)\s*'
        r'(\d{1,2})?/(\d{4})',

        # YYYY -> YYYY/Current
        r'\b(\d{4})\s*'
        r'(?:to|-|–)\s*'
        r'(\d{4}|Current|Present)'
    ]

    current_year = 2026

    employment_periods = []

    # ------------------------------------------------------------
    # 2.1 Month Year -> Month Year
    # ------------------------------------------------------------

    month_matches = re.findall(
        date_range_patterns[0],
        text,
        re.IGNORECASE
    )

    month_names = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12
    }

    for match in month_matches:
        start_month = match[0]
        start_year = int(match[1])
        end_month = match[2]
        end_year = match[3]

        if end_year.lower() in ("current", "present"):
            end_year = current_year
            end_month_num = 12
        else:
            end_year = int(end_year)
            end_month_num = month_names.get(
                end_month.lower(),
                12
            ) if end_month else 12

        start_month_num = month_names.get(
            start_month.lower(),
            1
        )

        start_total = start_year * 12 + start_month_num
        end_total = end_year * 12 + end_month_num

        if end_total >= start_total:
            employment_periods.append(
                (start_total, end_total)
            )

    # ------------------------------------------------------------
    # 2.2 MM/YYYY -> MM/YYYY
    # ------------------------------------------------------------

    numeric_matches = re.findall(
        date_range_patterns[1],
        text
    )

    for match in numeric_matches:
        start_month = int(match[0])
        start_year = int(match[1])

        end_month = match[2]
        end_year = match[3]

        if not end_year:
            continue

        end_year = int(end_year)

        if end_month:
            end_month = int(end_month)
        else:
            end_month = 12

        start_total = start_year * 12 + start_month
        end_total = end_year * 12 + end_month

        if end_total >= start_total:
            employment_periods.append(
                (start_total, end_total)
            )

    # ------------------------------------------------------------
    # 2.3 YYYY -> YYYY / Current
    # ------------------------------------------------------------

    year_matches = re.findall(
        date_range_patterns[2],
        text
    )

    for match in year_matches:
        start_year = int(match[0])
        end_value = match[1]

        if end_value.lower() in ("current", "present"):
            end_year = current_year
        else:
            end_year = int(end_value)

        if end_year >= start_year:
            start_total = start_year * 12
            end_total = end_year * 12

            employment_periods.append(
                (start_total, end_total)
            )

    # ============================================================
    # 3. Tính tổng thời gian kinh nghiệm từ employment periods
    # ============================================================

    if employment_periods:

        # Sắp xếp theo thời gian bắt đầu
        employment_periods.sort()

        # Gộp những khoảng thời gian bị overlap
        merged_periods = []

        current_start, current_end = employment_periods[0]

        for start, end in employment_periods[1:]:

            if start <= current_end:
                current_end = max(current_end, end)

            else:
                merged_periods.append(
                    (current_start, current_end)
                )

                current_start = start
                current_end = end

        merged_periods.append(
            (current_start, current_end)
        )

        total_months = sum(
            end - start
            for start, end in merged_periods
        )

        return round(total_months / 12, 1)

    # ============================================================
    # 4. Không tìm thấy thông tin kinh nghiệm
    # ============================================================

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

    return "Unknown"


def process_json(json_input, json_output):

    with open(json_input, 'r', encoding='utf-8') as f:
        resumes_list = json.load(f)

    extracted_resumes = []

    for cv in resumes_list:

        cv_id = cv.get("id")
        filename = cv.get("filename")
        category = cv.get("category")
        cv_text = cv.get("text", "")

        extracted_skills = extract_skills_regex(cv_text)
        extracted_exp = extract_experience_regex(cv_text)
        extracted_edu = extract_education_regex(cv_text)

        extracted_resumes.append({
            "id": cv_id,
            "filename": filename,
            "category": category,
            "skills": extracted_skills,
            "experience_years": extracted_exp,
            "education": extracted_edu
        })

    os.makedirs(
        os.path.dirname(json_output),
        exist_ok=True
    )

    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(
            extracted_resumes,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Processed {len(extracted_resumes)} resumes.")
    print(f"Output: {json_output}")


if __name__ == "__main__":

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_root = os.path.abspath(
        os.path.join(current_dir, "../..")
    )

    input_json = os.path.join(
        project_root,
        "Data",
        "Processed",
        "cleaned_resumes.json"
    )

    output_json = os.path.join(
        project_root,
        "Data",
        "Processed",
        "analyzed_resumes.json"
    )

    print("Input:")
    print(input_json)
    print()

    process_json(
        input_json,
        output_json
    )