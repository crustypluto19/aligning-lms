"""Welcome page - participant type selection and ID entry."""

import uuid
import streamlit as st
from datetime import datetime, timezone
from state.session import initialize_session
from services.persistence import check_duplicate_submission, is_submission_complete

st.set_page_config(
    page_title="Welcome - Survey",
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


def start_survey():
    """Callback to validate and start survey."""
    
    session = st.session_state.survey
    user_id_input = st.session_state.get("user_id_input", "")
    if not participant_type:
        st.session_state.start_error = "❌ Please select your participant type"
        return

    if session.user_id and session.participant_type == participant_type:
        user_id = session.user_id
    elif participant_type == "Prolific":
        if not user_id_input.strip():
            st.session_state.start_error = (
                "❌ Prolific ID is required for Prolific participants"
            )
            return
        user_id = user_id_input.strip()
    else:

        user_id = str(uuid.uuid4())[:8]

    if check_duplicate_submission(user_id):
        st.session_state.start_error = f"❌ This ID ({user_id}) has already completed the survey. Each participant can only submit once."
        return

    session.user_id = user_id
    session.participant_type = participant_type
    
    # Set start time if not already set for duration tracking
    if session.start_time is None:
        session.start_time = datetime.now(tz=timezone.utc)
    
    session.current_page = "background"
    st.session_state.start_error = None
    st.session_state.navigate_to = "pages/2_background.py"


session = initialize_session()

st.title("📝 Private Text Generation Survey")

# Check if survey already completed (cached)
if session.user_id:
    if "submission_checked_welcome" not in st.session_state:
        st.session_state.submission_checked_welcome = is_submission_complete(
            session.user_id
        )

    if st.session_state.submission_checked_welcome:
        st.info(
            "✅ You have already completed this survey. Thank you for your participation!"
        )
        st.stop()

st.markdown(
    """
### Welcome!

Thank you for participating in this survey. Our goal is to evaluate how **private text generation methods** 
affect the quality and acceptability of generated text. We use a technique to protect sensitive information 
by adding controlled noise to data. In this survey, you will compare original texts with privacy-protected alternatives.

**What you'll do:**
- Identify your participant type
- Answer some background questions
- Evaluate 17 questions comparing original text with privacy-protected alternatives
- Provide optional feedback

**Estimated completion time:** ~10 minutes

**CAUTION**: Some of the texts you see in this survey may contain emotional, offensive, vulgar, 
or uncomfortable language. If you feel that these may be upsetting to you, please consider not taking the survey.

**Extra important note**: At no point during the survey should you refresh your browser. 
This will reset the survey and invalidate your ongoing response! In between pages, please use the provided navigation buttons. This will ensure your responses are saved correctly.
"""
)

st.divider()

st.subheader("Step 1: Participant Type")

participant_type = st.radio(
    "Which type of participant are you?",
    options=["Prolific", "Friends and Family"],
    key="welcome_participant_type",
    horizontal=True,
)

if participant_type == "Prolific":
    st.subheader("Step 2: Prolific ID")
    # Pre-fill with existing user_id if returning to this page
    default_value = (
        session.user_id
        if session.user_id and session.participant_type == "Prolific"
        else ""
    )
    user_id_input = st.text_input(
        "Enter your Prolific ID:",
        value=default_value,
        placeholder="Required for Prolific participants",
        key="user_id_input",
    )
else:
    user_id_input = ""

# Show any validation errors
if "start_error" in st.session_state and st.session_state.start_error:
    st.error(st.session_state.start_error)

with st.form("welcome_form", clear_on_submit=False):
    st.form_submit_button(
        "🚀 Start Survey",
        type="primary",
        use_container_width=True,
        on_click=start_survey,
    )
if "navigate_to" in st.session_state and st.session_state.navigate_to:
    target = st.session_state.navigate_to
    st.session_state.navigate_to = None
    with st.spinner("📦 Loading survey..."):
        st.switch_page(target)
