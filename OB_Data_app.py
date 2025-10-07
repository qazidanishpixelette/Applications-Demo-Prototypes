import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


GENDER_OPTIONS = ["female", "male", "non_binary", "prefer_not_to_say", "other"]
PRIMARY_GOAL_OPTIONS = [
    "raising_capital",
    "hiring",
    "sales_pipeline",
    "partnerships",
    "customer_discovery",
    "other",
]
FUNDING_STAGE_TEXT_OPTIONS = [
    "idea",
    "pre_seed",
    "seed",
    "series_a",
    "series_b",
    "bootstrapped",
    "revenue_generating",
    "other",
]
INTRO_VOLUME_OPTIONS = ["quality_over_quantity", "quantity_over_quality", "balanced"]
COMM_STYLE_OPTIONS = ["formal_direct", "casual_friendly", "concise", "storytelling", "other"]
INTRO_METHOD_OPTIONS = [
    "email_intro",
    "calendar_invite",
    "whatsapp_intro",
    "slack_intro",
    "other",
]


def build_submission(
    user_id: str,
    resume_link: str,
    gender: str,
    age: int,
    multi_opts_bio: str,
    multi_opts_resume: str,
    multi_opts_linkedin: str,
    elevator_pitch: str,
    core_strength_1: str,
    core_strength_2: str,
    core_strength_3: str,
    primary_goal: str,
    funding_stage_numeric: int,
    target_check_size: int,
    key_metrics: str,
    preferred_roles_titles: str,
    industries_of_interest: str,
    funding_stage_text: str,
    deal_breakers: str,
    preferred_intro_volume: str,
    communication_style: str,
    introduction_method: str,
    anything_else: str,
) -> Dict[str, Any]:
    """Builds the submission payload matching onboarding_sample.json structure."""

    questions: List[Dict[str, Any]] = [
        {"prompt": "What is your gender?", "answer": gender},
        {"prompt": "What is your age?", "answer": age},
        {
            "prompt": "Please select any option or all available options.",
            "answer": {
                "bio": multi_opts_bio,
                "resume": multi_opts_resume,
                "linkedIn": multi_opts_linkedin,
            },
        },
        {"prompt": "Your \"Elevator Pitch\"?", "answer": elevator_pitch},
        {
            "prompt": "1. Write about your first core strength",
            "answer": core_strength_1,
        },
        {
            "prompt": "2. Write about your second core strength",
            "answer": core_strength_2,
        },
        {
            "prompt": "3. Write about your third core strength",
            "answer": core_strength_3,
        },
        {
            "prompt": "What is the main objective you want your AI agent to focus on? (Select one primary goal)",
            "answer": primary_goal,
        },
        {"prompt": "What is your funding stage?", "answer": funding_stage_numeric},
        {"prompt": "Target check size?", "answer": target_check_size},
        {
            "prompt": "Key metrics to share with aligned investors?",
            "answer": key_metrics,
        },
        {"prompt": "Preferred roles/titles?", "answer": preferred_roles_titles},
        {"prompt": "Industries of interest:", "answer": industries_of_interest},
        {"prompt": "What is your funding stage?", "answer": funding_stage_text},
        {"prompt": "Deal Breakers:", "answer": deal_breakers},
        {
            "prompt": "Preferred volume of introductions",
            "answer": preferred_intro_volume,
        },
        {
            "prompt": "What communication style do you typically prefer when meeting new people?",
            "answer": communication_style,
        },
        {
            "prompt": "Once you and a potential match both consent, how should the introduction happen?",
            "answer": introduction_method,
        },
        {"prompt": "Share anything else?", "answer": anything_else},
    ]

    submission: Dict[str, Any] = {
        "user_id": user_id,
        "resume_link": resume_link,
        "questions": questions,
    }
    return submission


def append_submission_to_file(submission: Dict[str, Any], file_path: Path) -> None:
    """Appends the submission as one JSON object per line to the onboarding file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(submission, ensure_ascii=False) + "\n")


def main() -> None:
    st.set_page_config(page_title="Onboarding Form", page_icon="📝", layout="centered")
    st.title("Onboarding Form")
    st.caption("Fill in the details below. Submissions are stored to onboarding.txt as JSON lines.")

    # Auto-fill helper
    def apply_autofill() -> None:
        sample = {
            "user_id": "uuid",
            "resume_link": "link_to_s3",
            "gender": "female",
            "age": 24,
            "bio": "Hi, I am a software engineer.",
            "resume_url": "www.s3Bucket.com",
            "linkedin_url": "www.linkedIn.com",
            "elevator_pitch": "this is my elevator pitch",
            "core_strength_1": "this is my core strength # 1",
            "core_strength_2": "this is my core strength # 2",
            "core_strength_3": "this is my core strength # 3",
            "primary_goal": "raising_capital",
            "funding_stage_numeric": 200000,
            "target_check_size": 5000000,
            "key_metrics": "no key metrics",
            "preferred_roles_titles": "Product Managers",
            "industries_of_interest": "AI",
            "funding_stage_text": "seed",
            "deal_breakers": "deal breaker",
            "preferred_intro_volume": "quality_over_quantity",
            "communication_style": "formal_direct",
            "introduction_method": "email_intro",
            "anything_else": "na",
        }
        for k, v in sample.items():
            st.session_state[k] = v

    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("Auto-fill sample data"):
            apply_autofill()
            st.rerun()

    with st.form("onboarding_form", clear_on_submit=False):
        st.subheader("Basic Information")
        user_id = st.text_input(
            "User ID",
            help="Provide a UUID or any unique ID for this user.",
            key="user_id",
            value=st.session_state.get("user_id", ""),
        )
        resume_link = st.text_input(
            "Resume Link (S3 or URL)", key="resume_link", value=st.session_state.get("resume_link", "")
        )

        st.subheader("Profile")
        gender = st.selectbox(
            "What is your gender?",
            options=GENDER_OPTIONS,
            index=(GENDER_OPTIONS.index(st.session_state.get("gender", GENDER_OPTIONS[0]))
                   if st.session_state.get("gender") in GENDER_OPTIONS else 0),
            key="gender",
        )
        age = st.number_input(
            "What is your age?",
            min_value=0,
            max_value=150,
            value=int(st.session_state.get("age", 0)),
            step=1,
            key="age",
        )

        st.markdown("Options: Provide any/all; leave blank if not applicable.")
        multi_opts_bio = st.text_area(
            "Bio", height=100, key="bio", value=st.session_state.get("bio", "")
        )
        multi_opts_resume = st.text_input(
            "Resume URL", key="resume_url", value=st.session_state.get("resume_url", "")
        )
        multi_opts_linkedin = st.text_input(
            "LinkedIn URL", key="linkedin_url", value=st.session_state.get("linkedin_url", "")
        )

        st.subheader("Pitch & Strengths")
        elevator_pitch = st.text_area(
            "Your \"Elevator Pitch\"?",
            height=120,
            key="elevator_pitch",
            value=st.session_state.get("elevator_pitch", ""),
        )
        core_strength_1 = st.text_area(
            "1. Write about your first core strength",
            height=80,
            key="core_strength_1",
            value=st.session_state.get("core_strength_1", ""),
        )
        core_strength_2 = st.text_area(
            "2. Write about your second core strength",
            height=80,
            key="core_strength_2",
            value=st.session_state.get("core_strength_2", ""),
        )
        core_strength_3 = st.text_area(
            "3. Write about your third core strength",
            height=80,
            key="core_strength_3",
            value=st.session_state.get("core_strength_3", ""),
        )

        st.subheader("Objectives & Funding")
        primary_goal = st.selectbox(
            "What is the main objective you want your AI agent to focus on? (Select one primary goal)",
            options=PRIMARY_GOAL_OPTIONS,
            index=(PRIMARY_GOAL_OPTIONS.index(st.session_state.get("primary_goal", PRIMARY_GOAL_OPTIONS[0]))
                   if st.session_state.get("primary_goal") in PRIMARY_GOAL_OPTIONS else 0),
            key="primary_goal",
        )
        funding_stage_numeric = st.number_input(
            "What is your funding stage? (numeric)",
            min_value=0,
            value=int(st.session_state.get("funding_stage_numeric", 0)),
            step=1,
            key="funding_stage_numeric",
        )
        target_check_size = st.number_input(
            "Target check size?",
            min_value=0,
            value=int(st.session_state.get("target_check_size", 0)),
            step=1,
            key="target_check_size",
        )
        key_metrics = st.text_area(
            "Key metrics to share with aligned investors?",
            height=80,
            key="key_metrics",
            value=st.session_state.get("key_metrics", ""),
        )
        preferred_roles_titles = st.text_input(
            "Preferred roles/titles?",
            key="preferred_roles_titles",
            value=st.session_state.get("preferred_roles_titles", ""),
        )
        industries_of_interest = st.text_input(
            "Industries of interest:",
            key="industries_of_interest",
            value=st.session_state.get("industries_of_interest", ""),
        )
        funding_stage_text = st.selectbox(
            "What is your funding stage? (text)",
            options=FUNDING_STAGE_TEXT_OPTIONS,
            index=(FUNDING_STAGE_TEXT_OPTIONS.index(st.session_state.get("funding_stage_text", FUNDING_STAGE_TEXT_OPTIONS[0]))
                   if st.session_state.get("funding_stage_text") in FUNDING_STAGE_TEXT_OPTIONS else 0),
            key="funding_stage_text",
        )

        st.subheader("Preferences")
        deal_breakers = st.text_area(
            "Deal Breakers:", height=80, key="deal_breakers", value=st.session_state.get("deal_breakers", "")
        )
        preferred_intro_volume = st.selectbox(
            "Preferred volume of introductions",
            options=INTRO_VOLUME_OPTIONS,
            index=(INTRO_VOLUME_OPTIONS.index(st.session_state.get("preferred_intro_volume", INTRO_VOLUME_OPTIONS[0]))
                   if st.session_state.get("preferred_intro_volume") in INTRO_VOLUME_OPTIONS else 0),
            key="preferred_intro_volume",
        )
        communication_style = st.selectbox(
            "What communication style do you typically prefer when meeting new people?",
            options=COMM_STYLE_OPTIONS,
            index=(COMM_STYLE_OPTIONS.index(st.session_state.get("communication_style", COMM_STYLE_OPTIONS[0]))
                   if st.session_state.get("communication_style") in COMM_STYLE_OPTIONS else 0),
            key="communication_style",
        )
        introduction_method = st.selectbox(
            "Once you and a potential match both consent, how should the introduction happen?",
            options=INTRO_METHOD_OPTIONS,
            index=(INTRO_METHOD_OPTIONS.index(st.session_state.get("introduction_method", INTRO_METHOD_OPTIONS[0]))
                   if st.session_state.get("introduction_method") in INTRO_METHOD_OPTIONS else 0),
            key="introduction_method",
        )
        anything_else = st.text_area(
            "Share anything else?", height=80, key="anything_else", value=st.session_state.get("anything_else", "")
        )

        submitted = st.form_submit_button("Submit")

    if submitted:
        submission = build_submission(
            user_id=user_id,
            resume_link=resume_link,
            gender=gender,
            age=int(age),
            multi_opts_bio=multi_opts_bio,
            multi_opts_resume=multi_opts_resume,
            multi_opts_linkedin=multi_opts_linkedin,
            elevator_pitch=elevator_pitch,
            core_strength_1=core_strength_1,
            core_strength_2=core_strength_2,
            core_strength_3=core_strength_3,
            primary_goal=primary_goal,
            funding_stage_numeric=int(funding_stage_numeric),
            target_check_size=int(target_check_size),
            key_metrics=key_metrics,
            preferred_roles_titles=preferred_roles_titles,
            industries_of_interest=industries_of_interest,
            funding_stage_text=funding_stage_text,
            deal_breakers=deal_breakers,
            preferred_intro_volume=preferred_intro_volume,
            communication_style=communication_style,
            introduction_method=introduction_method,
            anything_else=anything_else,
        )

        output_file = Path(__file__).resolve().parent / "onboarding.txt"
        append_submission_to_file(submission, output_file)

        st.success("Submission saved to onboarding.txt")
        with st.expander("View saved JSON"):
            st.code(json.dumps(submission, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


