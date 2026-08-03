"""Centralized MongoDB operations for survey sessions."""
import streamlit as st
from datetime import datetime, timezone
from typing import Optional
from models.session import SurveySession
from db import get_database, get_collections
from assignment import release_questions


@st.cache_data(ttl=60)  # Cache for 60 seconds to reduce DB queries
def is_submission_complete(user_id: str) -> bool:
    """Check if user_id has a completed submission."""
    try:
        db = get_database()
        _, responses_collection, _ = get_collections(db)
        existing = responses_collection.find_one({
            "user_id": user_id,
            "is_complete": True
        })
        return existing is not None
    except Exception:
        return False


def check_duplicate_submission(user_id: str) -> bool:
    """Check if user_id already has a completed submission."""
    return is_submission_complete(user_id)


def save_draft(session: SurveySession):
    """Save session as draft to MongoDB."""
    if not session.user_id:
        return  # Can't save without user_id
    
    try:
        db = get_database()
        _, responses_collection, _ = get_collections(db)
        
        draft_doc = {
            "user_id": session.user_id,
            "is_complete": False,
            "current_page": session.current_page,
            "start_time": session.start_time,
            "last_updated": datetime.now(tz=timezone.utc),
            "answers": session.answers,
            "attention_check_answered": session.attention_check_answered,
            "attention_check_failed": session.attention_check_failed,
            "attention_check_target": session.attention_check_target,
            "attention_check_position": session.attention_check_position,
            "attention_check_2_answered": session.attention_check_2_answered,
            "attention_check_2_failed": session.attention_check_2_failed,
            "attention_check_2_target": session.attention_check_2_target,
            "attention_check_2_position": session.attention_check_2_position,
            "assigned_questions": session.assigned_variant_ids,
        }
        
        responses_collection.update_one(
            {"user_id": session.user_id, "is_complete": False},
            {"$set": draft_doc},
            upsert=True
        )
        
        # Clear persist flag
        session.needs_persist = False
        
    except Exception as e:
        st.warning(f"Could not save draft: {str(e)}")


def load_draft(session: SurveySession):
    """Load draft data from MongoDB and update session."""
    if not session.user_id:
        return
    
    try:
        db = get_database()
        _, responses_collection, respondents_collection = get_collections(db)
        
        # Load response draft
        draft = responses_collection.find_one({
            "user_id": session.user_id,
            "is_complete": False
        })
        
        if draft:
            session.answers = draft.get("answers", {})
            # Ensure start_time is timezone-aware (MongoDB returns naive datetimes)
            start_time = draft.get("start_time")
            if start_time and start_time.tzinfo is None:
                session.start_time = start_time.replace(tzinfo=timezone.utc)
            else:
                session.start_time = start_time
            session.attention_check_answered = draft.get("attention_check_answered", False)
            session.attention_check_failed = draft.get("attention_check_failed", False)
            session.attention_check_target = draft.get("attention_check_target")
            session.attention_check_position = draft.get("attention_check_position")
            session.assigned_variant_ids = draft.get("assigned_questions")
            session.draft_loaded = True
        
        # Load background data
        respondent_doc = respondents_collection.find_one({"user_id": session.user_id})
        if respondent_doc:
            bg_data = respondent_doc.get("background", {})
            if bg_data:
                session.background = bg_data
                session.background_submitted = True
            
            # Restore participant type if available
            if "participant_type" in respondent_doc and not session.participant_type:
                session.participant_type = respondent_doc["participant_type"]
    
    except Exception as e:
        print(f"Error loading draft: {e}")


def save_background(session: SurveySession):
    """Persist background demographic data."""
    if not session.user_id:
        return
    
    try:
        db = get_database()
        _, _, respondents_collection = get_collections(db)
        
        respondents_collection.update_one(
            {"user_id": session.user_id},
            {
                "$set": {
                    "participant_type": session.participant_type,
                    "background": session.background,
                    "last_updated": datetime.now(tz=timezone.utc),
                },
                "$setOnInsert": {
                    "timestamp": datetime.now(tz=timezone.utc),  # Only set on first insert
                }
            },
            upsert=True
        )
    
    except Exception as e:
        st.warning(f"Could not persist background data: {e}")


def submit_response(session: SurveySession, feedback: Optional[str] = None):
    """Submit final survey response."""
    if not session.user_id:
        raise ValueError("Cannot submit without user_id")
    
    db = get_database()
    questions_collection, responses_collection, respondents_collection = get_collections(db)

    # Idempotency guard: if a completed submission already exists,
    # do not apply counter updates again.
    existing_complete = responses_collection.find_one(
        {"user_id": session.user_id, "is_complete": True},
        {"_id": 1},
    )
    if existing_complete:
        return

    # 1. CALCULATE METADATA
    checks_passed = 0
    if not session.attention_check_failed:
        checks_passed += 1
    if not session.attention_check_2_failed:
        checks_passed += 1
    attention_check_passed = checks_passed >= 1
    
    # Calculate survey completion duration
    submitted_at = datetime.now(tz=timezone.utc)
    duration_seconds = None
    if session.start_time:
        duration_seconds = (submitted_at - session.start_time).total_seconds()
    
    # Prepare final response document
    response_doc = {
        "user_id": session.user_id,
        "is_complete": True,
        "start_time": session.start_time,
        "submitted_at": submitted_at,
        "duration_seconds": duration_seconds,
        "answers": session.answers,
        "attention_check_1_passed": not session.attention_check_failed,
        "attention_check_2_passed": not session.attention_check_2_failed,
        "attention_check_passed": attention_check_passed,
        "assigned_questions": session.assigned_variant_ids,
        "feedback": feedback,
    }

    # If insert fails, function aborts and user can retry safely.
    responses_collection.insert_one(response_doc)

    # 3. INCREMENT ANSWERED COUNTS
    submitted_variant_ids = list(session.answers.keys())
    if submitted_variant_ids:
        try:
            questions_collection.update_many(
                {"variant_id": {"$in": submitted_variant_ids}},
                {"$inc": {"answered_count": 1}},
            )
        except Exception as e:
            print(f"Warning: Error updating answered counts: {e}")

    # 4. CLEANUP: RELEASE LOCKS
    # Use session-state assignment snapshot and rely on idempotent release.
    if session.assigned_variant_ids:
        try:
            release_questions(
                questions_collection,
                session.assigned_variant_ids,
                session.user_id,
            )
        except Exception as e:
            print(f"Warning: Non-fatal error releasing reservations: {e}")

    # 5. CLEANUP: DELETE DRAFT & SAVE PROFILE
    try:
        responses_collection.delete_many({
            "user_id": session.user_id,
            "is_complete": False,
        })

        respondents_collection.update_one(
            {"user_id": session.user_id},
            {
                "$set": {
                    "participant_type": session.participant_type,
                    "background": session.background,
                    "last_updated": datetime.now(tz=timezone.utc),
                },
                "$setOnInsert": {
                    "timestamp": datetime.now(tz=timezone.utc),
                },
            },
            upsert=True,
        )
    except Exception as e:
        print(f"Warning: Non-fatal error during final cleanup: {e}")
