"""Optional open-ended feedback page."""

import streamlit as st
from streamlit_survey import StreamlitSurvey
from state.session import initialize_session
from services.persistence import submit_response, is_submission_complete

st.set_page_config(
    page_title="Feedback - Survey",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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


def _submit_survey(session):
    """Submits the final survey response."""
    with st.status("📤 Submitting your responses...", expanded=True) as status:
        try:
            st.write("📝 Collecting feedback...")
            feedback_text = survey.data.get("feedback", {}).get("value", "")

            st.write("💾 Saving to database...")
            submit_response(session, feedback_text)

            st.write("✅ Successfully submitted!")
            status.update(label="Submission complete!", state="complete")

            session.current_page = "thank_you"
            st.switch_page("pages/5_thank_you.py")

        except Exception as e:
            status.update(label="Submission failed!", state="error")
            st.error(f"❌ Submission failed: {str(e)}")
            st.error("Please try again or contact support if the problem persists.")


def go_back_to_questions():
    """Navigates back to the questions page."""
    session = st.session_state.survey
    session.current_page = "questions"
    st.session_state.navigate_to = "pages/3_questions.py"


def submit_survey_callback():
    """Triggers the final survey submission."""
    session = st.session_state.survey
    st.session_state.submit_survey = True


survey = StreamlitSurvey()

session = initialize_session()

st.title("📝 Private Text Generation Survey")

st.title("📝 Private Text Generation Survey")

if not session.user_id:
    st.warning("⚠️ Please start from the beginning.")
    session.current_page = "welcome"
    st.switch_page("pages/1_welcome.py")

    st.switch_page("pages/1_welcome.py")

if "submission_checked_feedback" not in st.session_state:
    st.session_state.submission_checked_feedback = is_submission_complete(
        session.user_id
    )

if st.session_state.submission_checked_feedback:
    st.info(
        "✅ You have already completed this survey. Thank you for your participation!"
    )
    st.stop()

st.header("Feedback", divider=True)
st.markdown(
    "Thank you for your answers so far! We just have one more open-ended question, "
    "and it is **completely optional**."
)

survey.text_area(
    "In general, what were your reasoning patterns behind answering the previous questions? "
    "In other words, how did you decide which texts seemed most acceptable to you? What did you consider when determining whether texts were private or not?",
    id="feedback",
    height=300,
    placeholder="Type your answer here...",
)

with st.form("navigation_form"):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.form_submit_button(
            "⏮️ Previous", 
            type="secondary",
            on_click=go_back_to_questions
        )

    with col2:
        st.form_submit_button(
            "Submit Survey 🎯", 
            type="primary", 
            use_container_width=True,
            on_click=submit_survey_callback
        )

if "navigate_to" in st.session_state and st.session_state.navigate_to:
    target = st.session_state.navigate_to
    st.session_state.navigate_to = None
    st.switch_page(target)

if st.session_state.get("submit_survey"):
    st.session_state.submit_survey = False
    _submit_survey(session)
