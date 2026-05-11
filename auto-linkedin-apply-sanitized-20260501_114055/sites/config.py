"""Shared configuration for the multi-site job applier."""

from dataclasses import dataclass, field
import os


@dataclass
class ApplicantProfile:
    first_name: str = ""
    middle_name: str = ""
    last_name: str = ""
    email: str = ""
    phone: str = ""
    city: str = ""
    state: str = ""
    zipcode: str = ""
    country: str = "United States"

    resume_path: str = "all resumes/default_resume.pdf"

    linkedin_url: str = ""
    website: str = ""
    github: str = ""

    years_experience: int = 3
    require_visa: str = "No"
    us_citizenship: str = ""
    gender: str = "Decline"
    ethnicity: str = "Decline"
    veteran_status: str = "Decline"
    disability_status: str = "Decline"
    desired_salary: int = 80000

    @property
    def full_name(self) -> str:
        parts = [self.first_name, self.middle_name, self.last_name]
        return " ".join(part for part in parts if part)


@dataclass
class SearchConfig:
    search_terms: list[str] = field(
        default_factory=lambda: [
            "Marketing Manager",
            "Growth Marketing Manager",
            "Product Marketing Manager",
            "Demand Generation Manager",
            "Content Marketing Manager",
        ]
    )
    location: str = "United States"
    radius_miles: int = 25
    salary_min: int = 70000
    job_type: str = "fulltime"
    experience_level: str = "mid"
    remote_ok: bool = True
    date_posted_days: int = 7


CHROME_BETA_PATH = os.environ.get(
    "AUTO_APPLIER_CHROME_PATH",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)

BAD_WORDS: list[str] = []
SKIP_COMPANIES: list[str] = []
