"""Completion confirmation page."""

import streamlit as st
from state.session import initialize_session

st.set_page_config(
    page_title="Thank You - Survey",
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

session = initialize_session()

st.title("📝 Private Text Generation Survey")

if not session.user_id:
    st.warning("⚠️ Please start from the beginning.")
    session.current_page = "welcome"
    st.switch_page("pages/1_welcome.py")

st.header("✅ Thank You!", divider=True)

if session.participant_type == "Prolific":
    st.success(
        f"""
### Your responses have been recorded successfully! It is now safe to close this tab.

**Participant ID:** `{session.user_id}`

Your contribution helps advance research in privacy-preserving technologies.
Thank you for your time and thoughtful responses!
"""
    )
else:
    st.success(
        """
### Your responses have been recorded successfully! It is now safe to close this tab.

Your contribution helps advance research in privacy-preserving 
technologies.
Thank you for your time and thoughtful responses!
"""
    )

st.balloons()

st.divider()

if session.participant_type == "Prolific":
    try:
        checks_passed = 0
        if not session.attention_check_failed:
            checks_passed += 1
        if not session.attention_check_2_failed:
            checks_passed += 1
        attention_check_passed = checks_passed >= 1

        if attention_check_passed:
            completion_url = st.secrets["PROLIFIC_PASSED"]
            completion_code = st.secrets["PROLIFIC_PASSED_CODE"]
        else:
            completion_url = st.secrets["PROLIFIC_FAILED"]
            completion_code = st.secrets["PROLIFIC_FAILED_CODE"]

        st.info(
            f"Please head to the following completion link: [{completion_url}]({completion_url}). "
            f"Alternatively, use the completion code: **{completion_code}**"
        )
    except Exception:
        st.markdown("Please return to Prolific to complete your submission.")

st.text("Thank you once again! Have a great rest of your day.")
