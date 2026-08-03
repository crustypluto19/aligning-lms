"""Background page - demographic questions."""

import streamlit as st
from state.session import initialize_session
from services.persistence import save_background, save_draft, is_submission_complete
from db import get_database, get_collections
from assignment import assign_survey_questions

st.set_page_config(
    page_title="Background - Survey",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hide specific toolbar and profile info
st.markdown(
    """
    <style>
    div._profileContainer_gzau3_53,
    ._profileContainer_gzau3_53 {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def _load_background_from_db(user_id: str):
    """Load existing background data from MongoDB if it exists (cached)."""
    try:
        db = get_database()
        _, _, respondents_collection = get_collections(db)

        respondent_doc = respondents_collection.find_one({"user_id": user_id})
        if respondent_doc:
            return respondent_doc.get("background", {}), respondent_doc.get("participant_type")
    except Exception as e:
        print(f"Error loading background data: {e}")
    return None, None


def _load_existing_background(session):
    """Load existing background data from MongoDB if it exists."""
    try:
        bg_data, participant_type = _load_background_from_db(session.user_id)

        if bg_data:
            session.background = bg_data
            session.background_submitted = True
        if participant_type and not session.participant_type:
            session.participant_type = participant_type
    except Exception as e:
        print(f"Error loading existing background: {e}")


@st.cache_data(ttl=600)  # Cache for 10 minutes
def _load_questions_from_db(variant_ids: tuple):
    """Load questions from database by variant IDs (cached)."""
    db = get_database()
    questions_collection, _, _ = get_collections(db)
    questions = list(
        questions_collection.find(
            {"variant_id": {"$in": list(variant_ids)}}
        )
    )
    return {q["variant_id"]: q for q in questions}


def _load_or_assign_questions(session):
    """Load existing assigned questions from draft or assign new ones."""
    db = get_database()
    questions_collection, responses_collection, _ = get_collections(db)

    if session.assigned_variant_ids:
        # Load those specific questions using cached function
        questions_dict = _load_questions_from_db(tuple(session.assigned_variant_ids))
        # Restore original order
        session.questions = [
            questions_dict[vid]
            for vid in session.assigned_variant_ids
            if vid in questions_dict
        ]
    else:
        session.questions = assign_survey_questions(
            questions_collection, session.user_id
        )
        session.assigned_variant_ids = [q["variant_id"] for q in session.questions]
        save_draft(session)


def go_to_welcome():
    """Callback to save background and go to welcome page."""
    session = st.session_state.survey
    
    # Store all responses
    session.background["age"] = st.session_state.get("bg_age")
    session.background["gender"] = st.session_state.get("bg_gender")
    session.background["country"] = st.session_state.get("bg_country")
    session.background["education"] = st.session_state.get("bg_education")
    session.background["employment"] = st.session_state.get("bg_employment")
    
    save_background(session)
    session.current_page = "welcome"
    st.session_state.navigate_to = "pages/1_welcome.py"


def go_to_questions():
    """Callback to save background and go to questions page."""
    session = st.session_state.survey
    
    # Store all responses
    session.background["age"] = st.session_state.get("bg_age")
    session.background["gender"] = st.session_state.get("bg_gender")
    session.background["country"] = st.session_state.get("bg_country")
    session.background["education"] = st.session_state.get("bg_education")
    session.background["employment"] = st.session_state.get("bg_employment")
    
    session.background_submitted = True
    save_background(session)
    
    if not session.questions and "questions_assigned" not in st.session_state:
        st.session_state.prepare_questions = True
    
    session.current_page = "questions"
    st.session_state.navigate_to = "pages/3_questions.py"


session = initialize_session()

st.title("📝 Private Text Generation Survey")

if not session.user_id:
    st.warning("⚠️ Please start from the beginning.")
    session.current_page = "welcome"
    st.switch_page("pages/1_welcome.py")

# Check if survey already completed (cached)
if "submission_checked_background" not in st.session_state:
    st.session_state.submission_checked_background = is_submission_complete(
        session.user_id
    )

if st.session_state.submission_checked_background:
    st.info(
        "✅ You have already completed this survey. Thank you for your participation!"
    )
    st.stop()

# Load background data
if "background_loaded" not in st.session_state:
    if not session.background:
        with st.spinner("Loading..."):
            _load_existing_background(session)
    st.session_state.background_loaded = True

st.header("Background", divider=True)
st.text("Below are some background questions to help us get to know you better.")

# Define options
age_options = ["18-24", "25-34", "35-44", "45-54", "55+"]
gender_options = ["Female", "Male", "Non-binary", "Other", "I prefer not to say"]
education_options = [
    "High School",
    "Associate's Degree or equivalent",
    "Bachelor's Degree or equivalent",
    "Master's Degree or equivalent",
    "Doctorate or equivalent",
]
employment_options = ["Full-time", "Part-time", "Student", "Unemployed", "Other"]

# Batch inputs via form to prevent reruns
with st.form("background_form", clear_on_submit=False):
    age = st.radio(
        "What is your age?",
        options=age_options,
        index=(
            age_options.index(session.background["age"])
            if "age" in session.background and session.background["age"] in age_options
            else None
        ),
        key="bg_age",
    )

    gender = st.radio(
        "Which best describes your gender?",
        options=gender_options,
        index=(
            gender_options.index(session.background["gender"])
            if "gender" in session.background
            and session.background["gender"] in gender_options
            else None
        ),
        key="bg_gender",
    )

    country = st.selectbox(
        "What is your home country?",
        options=session.countries,
        index=(
            session.countries.index(session.background["country"])
            if "country" in session.background
            and session.background["country"] in session.countries
            else None
        ),
        key="bg_country",
    )

    education = st.radio(
        "What is your highest attained level of education?",
        options=education_options,
        index=(
            education_options.index(session.background["education"])
            if "education" in session.background
            and session.background["education"] in education_options
            else None
        ),
        key="bg_education",
    )

    employment = st.radio(
        "Please select the option that best describes your current employment status.",
        options=employment_options,
        index=(
            employment_options.index(session.background["employment"])
            if "employment" in session.background
            and session.background["employment"] in employment_options
            else None
        ),
        key="bg_employment",
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        previous_clicked = st.form_submit_button("⏮️ Previous", type="secondary")

    with col2:
        next_clicked = st.form_submit_button(
            "Next ⏭️", type="primary", use_container_width=True
        )

if previous_clicked:
    session.current_page = "welcome"
    session.background["age"] = age  # type: ignore
    session.background["gender"] = gender  # type: ignore
    session.background["country"] = country  # type: ignore
    session.background["education"] = education  # type: ignore
    session.background["employment"] = employment  # type: ignore

    save_background(session)
    st.switch_page("pages/1_welcome.py")

if next_clicked:
    session.background["age"] = age  # type: ignore
    session.background["gender"] = gender  # type: ignore
    session.background["country"] = country  # type: ignore
    session.background["education"] = education  # type: ignore
    session.background["employment"] = employment  # type: ignore

    session.background_submitted = True
    save_background(session)

    if not session.questions and "questions_assigned" not in st.session_state:
        with st.status("Preparing your survey questions...", expanded=True) as status:
            st.write("🔍 Checking existing assignments...")
            _load_or_assign_questions(session)
            st.write("✅ Questions loaded successfully!")
            status.update(label="Questions ready!", state="complete")
            st.session_state.questions_assigned = True

    session.current_page = "questions"
    st.switch_page("pages/3_questions.py")
