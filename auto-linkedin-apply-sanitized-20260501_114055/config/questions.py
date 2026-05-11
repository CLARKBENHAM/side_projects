"""Application-answer defaults.

These values are used when job forms ask reusable profile questions. Replace
every placeholder with the applicant's own resume, compensation, profile, and
work-authorization details before running.
"""

# Resume to upload. Keep a real resume outside shared zips.
default_resume_path = "all resumes/default_resume.pdf"

# Experience and work authorization.
years_of_experience = "3"
require_visa = "No"
us_citizenship = ""

# Public profile links.
website = ""
linkedIn = ""

# Compensation. Use numbers only where the original bot expects numbers.
desired_salary = 80000
current_ctc = 0
minimum_annual_salary_usd = 70000
notice_period = 14

# Application text answers.
linkedin_headline = ""
linkedin_summary = ""
cover_letter = ""
recent_employer = ""
confidence_level = "7"

# Safety settings for a new user. Change only after the first few dry runs.
pause_before_submit = True
pause_at_failed_question = True
overwrite_previous_answers = False
