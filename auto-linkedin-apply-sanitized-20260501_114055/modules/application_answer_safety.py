import re


YES_VALUES = {"yes", "y", "true"}
NO_VALUES = {"no", "n", "false"}
AUTHORIZED_TO_WORK_US = True
REQUIRES_SPONSORSHIP = False
WILLING_W2 = True
REQUIRES_C2C = False
WILLING_CONTRACT = True
US_CITIZEN: bool | None = None
GREEN_CARD_HOLDER: bool | None = None
HOME_CITY = "los angeles"
ONSITE_RADIUS_MILES = 30
WILLING_RELOCATE = False
WILLING_5_DAYS_ONSITE = True
MAX_TRAVEL_PERCENT = 75
TRUSTED_AUTO_SUBMIT_SOURCES = {"existing"}
TRUSTED_AUTO_SUBMIT_SOURCE_PREFIXES = (
    "profile:",
    "legal:",
    "preference:",
    "question_bank:",
)
UNSUPPORTED_CLAIM_PATTERNS = (
    r"\bj\.?\s*d\.?\b",
    r"\bjuris doctor\b",
    r"\bdoctor of law\b",
    r"\blaw degree\b",
    r"\bbar admission\b",
    r"\badmitted\b.{0,40}\bbar\b",
    r"\blicensed attorney\b",
    r"\battorney\b",
    r"\blawyer\b",
    r"\bpalantir\b",
    r"\bcsslp\b",
    r"\bcertified secure software lifecycle professional\b",
    r"\bph\.?\s*d\.?\b",
    r"\bdoctorate\b",
)
CLEARANCE_TERMS = (
    "clearance",
    "security clearance",
    "secret clearance",
    "top secret",
    "ts/sci",
    "polygraph",
)
CLEARANCE_WILLING_TERMS = (
    "willing",
    "able",
    "eligible",
    "obtain",
    "apply",
    "undergo",
    "complete",
    "sponsor",
)
CLEARANCE_CURRENT_TERMS = (
    "active",
    "current",
    "existing",
    "hold",
    "have",
    "possess",
    "held",
    "level",
)
UNSUPPORTED_YEAR_TERMS = (
    "c#",
    "c sharp",
    ".net",
    "dotnet",
    "palantir",
    "csslp",
    "transportation",
    "logistics",
    "supply chain",
)
HARD_NO_YEAR_TERMS = (
    "c#",
    "c sharp",
    ".net",
    "dotnet",
    "palantir",
    "csslp",
)
PRIORITY_SKILL_YEAR_RULES = (
    (
        ("architect level ai", "architect level llm"),
        "2",
        "profile:skill_years:architect_ai_llm",
    ),
    (("azure ai foundry",), "1", "profile:skill_years:azure_ai_foundry"),
    (
        ("claude code subagent", "claude code subagents"),
        "1",
        "profile:skill_years:claude_code_subagents",
    ),
    (("java",), "1", "profile:skill_years:java"),
)
SKILL_YEAR_RULES = (
    (("python",), "6", "profile:skill_years:python"),
    (("pytorch",), "4", "profile:skill_years:pytorch"),
    (("deep learning",), "4", "profile:skill_years:deep_learning"),
    (
        ("distributed training", "multi node", "multi gpu"),
        "2",
        "profile:skill_years:distributed_training",
    ),
    (
        ("llm", "large language model", "large language models", "generative ai"),
        "3",
        "profile:skill_years:llm",
    ),
    (("rag", "retrieval augmented generation"), "2", "profile:skill_years:rag"),
    (
        ("agentic ai", "ai agent", "ai agents", "agentic"),
        "1",
        "profile:skill_years:agentic_ai",
    ),
    (("machine learning", " ml "), "4", "profile:skill_years:machine_learning"),
    (("react.js", "reactjs", "react"), "3", "profile:skill_years:react"),
    (("node.js", "nodejs"), "3", "profile:skill_years:node"),
    (("aws", "amazon web services"), "3", "profile:skill_years:aws"),
    (("gcp", "google cloud"), "1", "profile:skill_years:gcp"),
    (("c++", "cpp"), "1", "profile:skill_years:cpp"),
    (("sql", "postgres", "postgresql", "mysql"), "4", "profile:skill_years:sql"),
    (("docker",), "4", "profile:skill_years:docker"),
    (("kubernetes", "k8s"), "1", "profile:skill_years:kubernetes"),
    (("jenkins",), "3", "profile:skill_years:jenkins"),
)
LOCATION_MISMATCH_TERMS = (
    "bay area",
    "redwood city",
    "san francisco",
    "palo alto",
    "menlo park",
    "mountain view",
    "san jose",
    "sunnyvale",
    "new york",
    "nyc",
    "manhattan",
    "brooklyn",
    "seattle",
    "ne corridor",
    "northeast corridor",
    "east coast",
    "boston",
)
LOCAL_COMMUTE_TERMS = (
    "los angeles",
    "la",
    "el segundo",
    "santa monica",
    "culver city",
    "playa vista",
    "marina del rey",
    "manhattan beach",
    "pasadena",
    "burbank",
    "glendale",
    "beverly hills",
    "century city",
)
UNSUPPORTED_DOMAIN_TERMS = (
    "speech",
    "voice",
    "asr",
    "tts",
    "audio model",
    "audio ai",
    "sar",
    "insar",
    "synthetic aperture radar",
    "robotics",
    "robotic",
    "drone",
    "drones",
    "uav",
    "unmanned",
    "medical device",
    "sap",
    "regulated qms",
    "qms",
    "payments",
    "payment",
    "billing",
    "computer vision",
    "defense",
    "edge hardware",
    "embedded",
    "geospatial",
    "government adjacent",
    "government-adjacent",
    "maritime",
    "multimodal inference",
    "on device",
    "on-device",
    "remote sensing",
    "rf data",
    "sensor fusion",
    "signal processing",
)
UNMAPPED_YEAR_TERMS = (
    "agile",
    "analytics engineering",
    "appian",
    "azure",
    "back end web development",
    "backend",
    "bash",
    "bazel",
    "cad",
    "catch2",
    "cdp",
    "clojure",
    "computer vision",
    "data engineering",
    "deepspeed",
    "embedded",
    "etl",
    "flash attention",
    "finance",
    "financial services",
    "fintech",
    "front end",
    "frontend",
    "gdb",
    "gdt",
    "go",
    "golang",
    "hadoop",
    "html",
    "insurance",
    "java",
    "javascript",
    "kafka",
    "large language model operations",
    "llmops",
    "marketing technology",
    "mlops",
    "mocha",
    "nutch",
    "nlp",
    "numpy",
    "ocr",
    "pandas",
    "perl",
    "pybind11",
    "rabbitmq",
    "ray",
    "ruby",
    "rust",
    "spark",
    "sqs",
    "systems integration",
    "tensorflow",
    "transformers",
    "typescript",
    "vision",
    "vllm",
)
COMP_FLOORS = {
    "senior_ai_ml_full_time": 200000,
    "principal_staff_ai": 220000,
    "founding_ai_engineer": 220000,
    "senior_backend_general": 170000,
    "contract_hourly": 90,
}
SPECIFIC_EXPERIENCE_TERMS = UNSUPPORTED_YEAR_TERMS + (
    "claude",
    "c++",
    "cpp",
    "java",
    "kubernetes",
    "terraform",
    "gcp",
    "google cloud",
    "aws",
    "azure",
    "react",
    "node",
    "node.js",
    "sql",
    "docker",
    "jenkins",
    "llm",
    "rag",
    "deep learning",
    "distributed training",
    "banking",
    "finance",
    "financial services",
    "fintech",
    "go",
    "golang",
    "insurance",
    "salesforce",
    "appian",
)


def normalized_label(label: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9+#.]+", " ", label.lower()).split())


def _contains_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _option_value(answer: str, options: list[str] | None) -> str | None:
    if not options:
        return answer
    normalized_answer = answer.strip().lower()
    for option in options:
        if normalized_answer == str(option).strip().lower():
            return option
    for option in options:
        normalized_option = str(option).strip().lower()
        if (
            normalized_answer in normalized_option
            or normalized_option in normalized_answer
        ):
            return option
    return None


def _looks_yes(value: object) -> bool:
    normalized = normalized_label(str(value or ""))
    return normalized in YES_VALUES


def _looks_no(value: object) -> bool:
    normalized = normalized_label(str(value or ""))
    return normalized in NO_VALUES


def _numeric_value(value: object) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    if not match:
        return None
    return float(match.group(0))


def _bool_answer(
    value: bool | None, options: list[str] | None, source: str
) -> tuple[str, str] | None:
    if value is None:
        return None
    answer = _option_or_text("Yes" if value else "No", options)
    return (answer, source) if answer else None


def _extract_percent(label: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", label)
    if match:
        return float(match.group(1))
    if "travel" in label:
        match = re.search(r"\b(\d+(?:\.\d+)?)\b", label)
        if match:
            return float(match.group(1))
    return None


def _parse_money_amount(raw_value: str, suffix: str | None) -> float:
    value = float(raw_value.replace(",", ""))
    if suffix and suffix.lower() == "k":
        value *= 1000
    return value


_COMPENSATION_RANGE_PATTERN = re.compile(
    r"\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?\s*(?:-|to|–|—)\s*"
    r"\$ ?(\d[\d,]*(?:\.\d+)?)\s*(k|K)?",
    re.IGNORECASE,
)


def _compensation_bounds(text: str) -> tuple[float | None, float | None]:
    match = _COMPENSATION_RANGE_PATTERN.search(text.replace("–", "-").replace("—", "-"))
    if not match:
        return None, None
    low = _parse_money_amount(match.group(1), match.group(2))
    high = _parse_money_amount(match.group(3), match.group(4))
    return min(low, high), max(low, high)


def _role_comp_floor(job_context: str) -> tuple[int, str]:
    normalized = normalized_label(job_context)
    if "contract" in normalized or "hourly" in normalized:
        return COMP_FLOORS["contract_hourly"], "preference:comp_floor:contract_hourly"
    if any(term in normalized for term in ("principal", "staff")) and any(
        term in normalized for term in ("ai", "ml", "machine learning")
    ):
        return (
            COMP_FLOORS["principal_staff_ai"],
            "preference:comp_floor:principal_staff_ai",
        )
    if "founding" in normalized and any(
        term in normalized for term in ("ai", "ml", "machine learning")
    ):
        return (
            COMP_FLOORS["founding_ai_engineer"],
            "preference:comp_floor:founding_ai_engineer",
        )
    if any(term in normalized for term in ("senior", "sr")) and any(
        term in normalized for term in ("ai", "ml", "machine learning")
    ):
        return (
            COMP_FLOORS["senior_ai_ml_full_time"],
            "preference:comp_floor:senior_ai_ml_full_time",
        )
    if any(term in normalized for term in ("senior", "sr", "backend")):
        return (
            COMP_FLOORS["senior_backend_general"],
            "preference:comp_floor:senior_backend_general",
        )
    return COMP_FLOORS["senior_ai_ml_full_time"], "preference:comp_floor:default_ai"


def is_rating_scale_label(label: str) -> bool:
    normalized = re.sub(
        r"\s+",
        " ",
        label.lower().replace("–", "-").replace("—", "-"),
    )
    return bool(
        re.search(
            r"\b(?:scale|rating|rate|proficiency)\b.{0,100}\b1\s*(?:-|to)\s*10\b",
            normalized,
        )
        or re.search(
            r"\b1\s*(?:-|to)\s*10\b.{0,100}\b(?:scale|rating|rate|proficiency)\b",
            normalized,
        )
    )


def asks_for_years(label: str) -> bool:
    normalized = normalized_label(label)
    if not re.search(r"\b(years?|yrs?)\b", normalized):
        return False
    return bool(
        "experience" in normalized
        or normalized.startswith(("how many year", "how many yr"))
        or re.search(
            r"\b\d+\s*(?:\+|or more|plus)?\s*(?:years?|yrs?)\b.{0,80}\b"
            r"(?:shipping|building|working|coding|developing|designing|architecting|using|with|in)\b",
            normalized,
        )
        or re.search(
            r"\b(?:years?|yrs?)\b.{0,60}\b(?:of|with|using|in)\b",
            normalized,
        )
    )


def asks_for_experience(label: str) -> bool:
    normalized = normalized_label(label)
    return "experience" in normalized or asks_for_years(normalized)


def is_specific_experience_label(label: str) -> bool:
    normalized = normalized_label(label)
    if not asks_for_experience(normalized):
        return False
    return any(term in normalized for term in SPECIFIC_EXPERIENCE_TERMS) or bool(
        re.search(r"\b(with|using|in)\b.{0,80}\b[A-Za-z0-9+#.]{2,}\b", label)
    )


def _is_yes_no_options(options: list[str] | None) -> bool:
    normalized_options = {normalized_label(str(option)) for option in options or []}
    return bool(normalized_options) and normalized_options.issubset(
        YES_VALUES | NO_VALUES | {"select an option"}
    )


def _asks_yes_no_question(normalized_label_text: str) -> bool:
    return bool(
        re.search(
            r"^(?:do|does|did|have|has|had|are|is|can|could|will|would|should)\b",
            normalized_label_text,
        )
    )


def _option_or_text(answer: str, options: list[str] | None) -> str | None:
    if options:
        return _option_value(answer, options)
    return answer


def _year_option_value(years: str, options: list[str] | None) -> str | None:
    if not options:
        return years
    try:
        numeric_years = float(years)
    except ValueError:
        return _option_value(years, options)

    for option in options:
        normalized_option = normalized_label(str(option))
        range_match = re.search(r"\b(\d+)\s*(?:-|to)\s*(\d+)\b", normalized_option)
        if range_match:
            lower = float(range_match.group(1))
            upper = float(range_match.group(2))
            if lower <= numeric_years <= upper:
                return option
        plus_match = re.search(
            r"\b(\d+)\s*(?:\+|or more|plus)(?:\b|\s|$)", normalized_option
        )
        if plus_match and numeric_years >= float(plus_match.group(1)):
            return option
        exact_match = re.search(r"\b(\d+)\b", normalized_option)
        if exact_match and numeric_years == float(exact_match.group(1)):
            return option
    return _option_value(years, options)


def _year_answer_matches(years: str, answer: object) -> bool:
    raw_answer = str(answer or "").lower().replace("–", "-").replace("—", "-")
    try:
        expected_years = float(years)
    except ValueError:
        return normalized_label(str(answer or "")) == normalized_label(years)

    range_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\b",
        raw_answer,
    )
    if range_match:
        lower = float(range_match.group(1))
        upper = float(range_match.group(2))
        return lower <= expected_years <= upper

    plus_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:\+|or more|plus)(?:\b|\s|$)",
        raw_answer,
    )
    if plus_match:
        return expected_years >= float(plus_match.group(1))

    numeric_answer = _numeric_value(answer)
    return numeric_answer is not None and numeric_answer == expected_years


def _required_year_threshold(label: str) -> float | None:
    normalized = normalized_label(label)
    match = re.search(r"\b(\d+)\s*(?:\+|or more|plus)\s*years?\b", normalized)
    if match:
        return float(match.group(1))
    match = re.search(r"\bat least\s*(\d+)\s*years?\b", normalized)
    if match:
        return float(match.group(1))
    return None


def _normalized_contains_term(normalized_with_spaces: str, term: str) -> bool:
    return f" {normalized_label(term)} " in normalized_with_spaces


def _skill_year_answer(label: str) -> tuple[str, str] | None:
    normalized_label_text = normalized_label(label)
    normalized = f" {normalized_label_text} "
    for terms, answer, source in PRIORITY_SKILL_YEAR_RULES:
        if any(_normalized_contains_term(normalized, term) for term in terms):
            return answer, source
    if any(_normalized_contains_term(normalized, term) for term in UNMAPPED_YEAR_TERMS):
        return None
    for terms, answer, source in SKILL_YEAR_RULES:
        if any(_normalized_contains_term(normalized, term) for term in terms):
            return answer, source
    return None


def _has_unsupported_domain_term(normalized_label_text: str) -> bool:
    return any(term in normalized_label_text for term in UNSUPPORTED_DOMAIN_TERMS)


def _asks_certification_or_license(normalized_label_text: str) -> bool:
    return any(
        term in normalized_label_text
        for term in ("certification", "certified", "certificate", "license", "licensed")
    )


def _has_location_mismatch(normalized_label_text: str) -> bool:
    return (
        any(term in normalized_label_text for term in LOCATION_MISMATCH_TERMS)
        and HOME_CITY not in normalized_label_text
    )


def _has_local_commute_location(normalized_label_text: str) -> bool:
    return any(term in normalized_label_text for term in LOCAL_COMMUTE_TERMS)


def _location_or_travel_answer(
    normalized_label_text: str, options: list[str] | None
) -> tuple[str, str] | None:
    if "travel" in normalized_label_text:
        percent = _extract_percent(normalized_label_text)
        if percent is not None:
            return _bool_answer(
                percent <= MAX_TRAVEL_PERCENT,
                options,
                "preference:max_travel",
            )
        return None

    if any(
        term in normalized_label_text
        for term in (
            "located",
            "near",
            "commute",
            "hybrid",
            "onsite",
            "on site",
            "relocate",
            "relocation",
        )
    ):
        if _has_location_mismatch(normalized_label_text):
            return _bool_answer(False, options, "preference:location_mismatch")
        if _has_local_commute_location(normalized_label_text) and (
            "onsite" not in normalized_label_text
            and "on site" not in normalized_label_text
            or WILLING_5_DAYS_ONSITE
        ):
            return _bool_answer(True, options, "preference:local_onsite_ok")

    if "relocate" in normalized_label_text or "relocation" in normalized_label_text:
        if not WILLING_RELOCATE:
            return _bool_answer(False, options, "preference:no_relocation")

    return None


def recommended_compensation_answer(
    label: str,
    job_context: str = "",
    default_annual: int = 200000,
) -> tuple[str, str] | None:
    normalized = normalized_label(label)
    if not any(
        term in normalized
        for term in ("salary", "compensation", "pay", "ctc", "rate", "hourly")
    ):
        return None
    if "current" in normalized or "present" in normalized:
        return None

    context = f"{label}\n{job_context}"
    floor, source = _role_comp_floor(context)
    if floor == COMP_FLOORS["contract_hourly"] or "hour" in normalized:
        return str(COMP_FLOORS["contract_hourly"]), source

    _low, high = _compensation_bounds(context)
    if high is not None and high >= 300000:
        return str(max(floor, 250000)), "preference:listed_comp_high"
    if high is not None and high >= 200000:
        return str(max(floor, 220000)), "preference:listed_comp_mid"
    return str(max(floor, default_annual)), source


def _has_clearance_term(normalized_label_text: str) -> bool:
    return any(term in normalized_label_text for term in CLEARANCE_TERMS)


def _asks_willing_to_get_clearance(normalized_label_text: str) -> bool:
    return _has_clearance_term(normalized_label_text) and any(
        term in normalized_label_text for term in CLEARANCE_WILLING_TERMS
    )


def _asks_current_clearance(normalized_label_text: str) -> bool:
    return _has_clearance_term(normalized_label_text) and any(
        term in normalized_label_text for term in CLEARANCE_CURRENT_TERMS
    )


def is_contact_or_identity_label(label: str) -> bool:
    normalized = normalized_label(label)
    contact_terms = (
        "name",
        "email",
        "phone",
        "mobile",
        "street",
        "address",
        "city",
        "state",
        "province",
        "postal",
        "postcode",
        "zip",
        "country",
        "linkedin",
        "website",
        "portfolio",
        "headline",
        "signature",
    )
    return any(term in normalized for term in contact_terms)


def manual_review_reason_for_question(label: str) -> str | None:
    normalized = normalized_label(label)
    if "c2c" in normalized or "corp to corp" in normalized or "1099" in normalized:
        return None
    if any(
        term in normalized
        for term in (
            "relocate",
            "relocation",
            "travel",
            "onsite",
            "on site",
            "hybrid",
            "commute",
        )
    ):
        return "location, relocation, onsite, or travel preference needs review"
    if _has_clearance_term(normalized):
        return None
    if _contains_pattern(normalized, UNSUPPORTED_CLAIM_PATTERNS):
        return None
    if (
        "us citizen" in normalized
        or "u s citizen" in normalized
        or "green card" in normalized
        or "permanent resident" in normalized
    ):
        return "citizenship or permanent resident status is not locked"
    if asks_for_years(normalized) and _skill_year_answer(normalized):
        return None
    if asks_for_years(normalized) and any(
        term in normalized for term in HARD_NO_YEAR_TERMS
    ):
        return "unsupported required skill or employer-specific experience"
    if asks_for_years(normalized) and "claude" in normalized:
        return "Claude-specific years need manual review"
    if (
        asks_for_years(normalized)
        and "architect" in normalized
        and ("ai" in normalized or "llm" in normalized)
    ):
        return "architect-level AI years need manual review"
    if asks_for_years(normalized) and _has_unsupported_domain_term(normalized):
        return None
    if asks_for_years(normalized) and is_specific_experience_label(normalized):
        if _skill_year_answer(normalized):
            return None
        return "specific skill years are not in the verified skill map"
    if (
        any(term in normalized for term in ("previously worked", "worked at"))
        or "former employer" in normalized
    ):
        return "previous-employer claims require exact profile evidence"
    if any(term in normalized for term in ("certification", "certified", "license")):
        return "certification or license claims require exact profile evidence"
    return None


def deterministic_answer_for_question(
    label: str,
    field_type: str,
    options: list[str] | None = None,
    confidence_level: str = "8",
    generic_years_of_experience: str = "5",
) -> tuple[str, str] | None:
    normalized = normalized_label(label)
    if _has_clearance_term(normalized):
        if _asks_willing_to_get_clearance(normalized):
            if options:
                answer = _option_value("Yes", options)
                if answer:
                    return answer, "legal:willing_to_obtain_clearance"
            return (
                "I do not currently have a clearance, but I am willing to obtain one.",
                "legal:willing_to_obtain_clearance",
            )
        if options:
            answer = _option_value("No", options)
            if answer:
                return answer, "legal:no_current_clearance"
        return (
            "I do not currently have a clearance, but I am willing to obtain one.",
            "legal:no_current_clearance",
        )

    if _contains_pattern(normalized, UNSUPPORTED_CLAIM_PATTERNS):
        answer = _option_value("No", options)
        return (answer, "legal:verified_no_claim") if answer else None

    if _has_unsupported_domain_term(normalized) and asks_for_years(normalized):
        return "0", "profile:no_experience:domain_specific"

    if _has_unsupported_domain_term(normalized) and (
        _is_yes_no_options(options) or _asks_yes_no_question(normalized)
    ):
        answer = _option_value("No", options)
        return (answer, "profile:no_experience:domain_specific") if answer else None

    if _asks_certification_or_license(normalized) and (
        _is_yes_no_options(options) or _asks_yes_no_question(normalized)
    ):
        answer = _option_value("No", options)
        return (answer, "profile:no_certification_or_license") if answer else None

    if any(
        term in normalized
        for term in ("authorized to work", "legally authorized", "eligible to work")
    ):
        return _bool_answer(AUTHORIZED_TO_WORK_US, options, "legal:work_authorization")

    if (
        "without" in normalized
        and ("sponsorship" in normalized or "visa" in normalized)
    ) or "no sponsorship" in normalized:
        return _bool_answer(
            not REQUIRES_SPONSORSHIP, options, "legal:ok_without_sponsorship"
        )

    if "sponsorship" in normalized or "visa" in normalized:
        return _bool_answer(
            REQUIRES_SPONSORSHIP, options, "legal:no_sponsorship_required"
        )

    if any(term in normalized for term in ("w2", "w-2")) and any(
        term in normalized for term in ("willing", "able", "work", "accept")
    ):
        return _bool_answer(WILLING_W2, options, "preference:w2_ok")

    if any(term in normalized for term in ("c2c", "corp to corp", "1099")):
        return _bool_answer(REQUIRES_C2C, options, "preference:no_c2c_required")

    if (
        "contract" in normalized
        and any(
            term in normalized
            for term in ("comfortable", "willing", "ok", "okay", "open")
        )
        and (_is_yes_no_options(options) or _asks_yes_no_question(normalized))
    ):
        return _bool_answer(WILLING_CONTRACT, options, "preference:contract_ok")

    if (
        ("us citizen" in normalized or "u s citizen" in normalized)
        and "green card" not in normalized
        and "permanent resident" not in normalized
        and "gc" not in normalized
    ):
        return _bool_answer(US_CITIZEN, options, "legal:us_citizen")

    if (
        ("us citizen" in normalized or "u s citizen" in normalized)
        and any(
            term in normalized for term in ("green card", "permanent resident", "gc")
        )
        and _is_yes_no_options(options)
    ):
        if US_CITIZEN is None or GREEN_CARD_HOLDER is None:
            return None
        return _bool_answer(
            US_CITIZEN or GREEN_CARD_HOLDER, options, "legal:citizen_or_gc"
        )

    if location_answer := _location_or_travel_answer(normalized, options):
        return location_answer

    if is_rating_scale_label(label):
        return confidence_level, "profile:rating_scale"

    if asks_for_years(normalized):
        if _is_yes_no_options(options):
            threshold = _required_year_threshold(label)
            if threshold is not None and not is_specific_experience_label(normalized):
                has_years = float(generic_years_of_experience) >= threshold
                return _bool_answer(
                    has_years, options, "profile:generic_years_threshold"
                )
        if "information technology" in normalized or " it " in f" {normalized} ":
            answer = _year_option_value("5", options)
            return (answer, "profile:post_grad_year_cap") if answer else None
        skill_answer = _skill_year_answer(normalized)
        if skill_answer:
            answer = _year_option_value(skill_answer[0], options)
            return (answer, skill_answer[1]) if answer else None
        if any(term in normalized for term in HARD_NO_YEAR_TERMS):
            return None
        if not is_specific_experience_label(normalized):
            answer = _year_option_value(generic_years_of_experience, options)
            return (answer, "profile:generic_years") if answer else None

    return None


def should_override_existing_answer(
    label: str,
    existing_answer: object,
    field_type: str,
    max_post_grad_years: int = 6,
) -> bool:
    normalized = normalized_label(label)
    if not str(existing_answer or "").strip():
        return False
    if is_rating_scale_label(label):
        value = _numeric_value(existing_answer)
        return value is None or value < 1 or value > 10
    if _has_clearance_term(normalized):
        if _asks_willing_to_get_clearance(normalized):
            return _looks_no(existing_answer)
        if _asks_current_clearance(normalized):
            return _looks_yes(existing_answer)
        return _looks_yes(existing_answer)
    if _contains_pattern(normalized, UNSUPPORTED_CLAIM_PATTERNS):
        return _looks_yes(existing_answer) or bool(_numeric_value(existing_answer))
    if any(
        term in normalized
        for term in ("authorized to work", "legally authorized", "eligible to work")
    ):
        return _looks_no(existing_answer)
    if (
        "without" in normalized
        and ("sponsorship" in normalized or "visa" in normalized)
    ) or "no sponsorship" in normalized:
        return _looks_no(existing_answer)
    if "sponsorship" in normalized or "visa" in normalized:
        return _looks_yes(existing_answer)
    if any(term in normalized for term in ("w2", "w-2")) and any(
        term in normalized for term in ("willing", "able", "work", "accept")
    ):
        return _looks_no(existing_answer)
    if any(term in normalized for term in ("c2c", "corp to corp", "1099")):
        return _looks_yes(existing_answer)
    if _asks_certification_or_license(normalized):
        return _looks_yes(existing_answer) or bool(_numeric_value(existing_answer))
    if location_answer := _location_or_travel_answer(normalized, ["Yes", "No"]):
        if location_answer[0] == "Yes":
            return _looks_no(existing_answer)
        if location_answer[0] == "No":
            return _looks_yes(existing_answer)
    if any(term in normalized for term in LOCATION_MISMATCH_TERMS) and _looks_yes(
        existing_answer
    ):
        return True
    if any(
        term in normalized
        for term in (
            "relocate",
            "relocation",
            "travel",
            "onsite",
            "on site",
            "hybrid",
            "commute",
        )
    ) and _looks_yes(existing_answer):
        return True
    if (
        any(term in normalized for term in ("previously worked", "worked at"))
        or "former employer" in normalized
    ) and _looks_yes(existing_answer):
        return True
    if any(
        term in normalized for term in ("certification", "certified", "license")
    ) and _looks_yes(existing_answer):
        return True
    if _has_unsupported_domain_term(normalized) and _looks_yes(existing_answer):
        return True
    if any(
        term in normalized
        for term in (
            "speech",
            "voice",
            "asr",
            "tts",
            "robotics",
            "uav",
            "insar",
            "sar dataset",
            "systems integration",
        )
    ) and _looks_yes(existing_answer):
        return True
    if asks_for_years(normalized):
        value = _numeric_value(existing_answer)
        skill_answer = _skill_year_answer(normalized)
        if skill_answer:
            return not _year_answer_matches(skill_answer[0], existing_answer)
        if _has_unsupported_domain_term(normalized) and value not in {0, None}:
            return True
        if value is not None and value > max_post_grad_years:
            return True
        if any(term in normalized for term in HARD_NO_YEAR_TERMS) and value not in {
            0,
            None,
        }:
            return True
        if is_specific_experience_label(normalized) and not skill_answer:
            return True
    return False


def answer_source_allows_auto_submit(source: object) -> bool:
    source_text = str(source or "").strip()
    if source_text in TRUSTED_AUTO_SUBMIT_SOURCES:
        return True
    return source_text.startswith(TRUSTED_AUTO_SUBMIT_SOURCE_PREFIXES)


def answer_record_manual_review_reason(question_record: tuple) -> str | None:
    if len(question_record) < 5:
        return "Malformed question record requires manual review"

    label, answer, field_type, _previous_answer, answer_source = question_record
    source = str(answer_source or "")
    if source in {"manual", "manual_placeholder"}:
        return f"{field_type}: {label} needs manual input"
    if not answer_source_allows_auto_submit(source):
        return (
            f"{field_type}: {label} used untrusted answer source {source or 'unknown'}"
        )
    if should_override_existing_answer(str(label), answer, str(field_type)):
        return f"{field_type}: {label} has an unsafe saved or generated answer"
    return None
