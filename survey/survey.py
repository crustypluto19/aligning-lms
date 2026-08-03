import streamlit as st
from state.session import initialize_session

st.set_page_config(
    page_title="Private Text Generation Evaluation Survey",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def main():
    session = initialize_session()
    
    
    page_map = {
        "welcome": "pages/1_welcome.py",
        "background": "pages/2_background.py",
        "questions": "pages/3_questions.py",
        "feedback": "pages/4_feedback.py",
        "thank_you": "pages/5_thank_you.py",
    }
    
    target_page = page_map.get(session.current_page, "pages/1_welcome.py")
    st.switch_page(target_page)

if __name__ == "__main__":
    main()
