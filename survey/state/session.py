"""Session state manager - single entry point for accessing survey session."""
import streamlit as st
import os
from models.session import SurveySession
from services.persistence import load_draft
from assignment import cleanup_stale_reservations
from db import get_database, get_collections


@st.cache_data
def _load_countries() -> list:
    """Load countries from data/countries.txt (cached)."""
    try:
        base = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base, "data", "countries.txt")
        with open(path, "r", encoding="utf-8") as f:
            countries = [line.strip() for line in f if line.strip()]
            return countries if countries else [""]
    except Exception:
        return [""]


def get_session() -> SurveySession:
    """Get or create the survey session."""
    if "survey" not in st.session_state:
        st.session_state.survey = SurveySession()
        st.session_state.survey.countries = _load_countries()
    
    return st.session_state.survey


def restore_from_query_params(session: SurveySession):
    """Restore session state from URL query parameters."""
    query_params = st.query_params
    
    qp_user = query_params.get("user_id")
    if qp_user:
        if isinstance(qp_user, (list, tuple)):
            qp_user = qp_user[0] if qp_user else None
        if qp_user and not session.user_id:
            session.user_id = qp_user


def sync_to_query_params(session: SurveySession):
    """Sync session state to URL query parameters."""
    if session.user_id:
        st.query_params["user_id"] = session.user_id


def initialize_session() -> SurveySession:
    """Initialize session with all necessary setup."""
    session = get_session()
    
    restore_from_query_params(session)
    
    if session.user_id and not session.draft_loaded:
        load_draft(session)
    
    if not session.cleanup_done:
        _cleanup_stale_reservations(session)
    
    sync_to_query_params(session)
    
    return session


def _cleanup_stale_reservations(session: SurveySession):
    """Run cleanup of stale question reservations once per session."""
    try:
        db = get_database()
        questions_collection, _, _ = get_collections(db)
        cleanup_stale_reservations(questions_collection)
        session.cleanup_done = True
    except Exception as e:
        print(f"Cleanup error (non-critical): {e}")
