"""Questions page - main survey with attention check."""

import streamlit as st
import random
import os
import toml
from state.session import initialize_session
from services.persistence import save_draft, is_submission_complete, load_draft
from db import get_database, get_collections

st.set_page_config(
    page_title="Questions - Survey",
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


@st.cache_data(ttl=600)  # Cache for 10 minutes
def _fetch_questions_by_variant_ids(variant_ids: tuple):
    """Fetch questions from database by variant IDs (cached)."""
    db = get_database()
    questions_collection, _, _ = get_collections(db)
    questions = list(
        questions_collection.find({"variant_id": {"$in": list(variant_ids)}})
    )
    return {q["variant_id"]: q for q in questions}


def _ensure_questions_loaded(session):
    """Ensure questions are loaded from assignment or draft."""
    if not session.user_id:
        st.error("❌ Session expired. Please start from the beginning.")
        session.current_page = "welcome"
        st.switch_page("pages/1_welcome.py")

    if session.assigned_variant_ids and not session.questions:
        if "questions_loaded_from_db" not in st.session_state:
            questions_dict = _fetch_questions_by_variant_ids(
                tuple(session.assigned_variant_ids)
            )
            session.questions = [
                questions_dict[vid]
                for vid in session.assigned_variant_ids
                if vid in questions_dict
            ]
            st.session_state.questions_loaded_from_db = True

    if not session.questions:
        st.error("❌ Could not load questions. Please go back and try again.")
        session.current_page = "background"
        st.switch_page("pages/2_background.py")


def _render_attention_check(session, question_num: int, check_number: int = 1):
    """Render attention check question."""
    st.markdown(f"### Question {question_num} of 17")
    st.markdown("This is an attention check to make sure that you're still awake! 😴")

    attention_options = [
        "This response demonstrates careful reading and understanding of the instructions provided above.",
        "The participant has reviewed the survey carefully and is paying close attention to detail.",
        "An attentive reader would select this option after reading the text above thoroughly.",
    ]

    option_labels = ["Option 1", "Option 2", "Option 3"]

    if check_number == 1:
        target_attr = "attention_check_target"
        answered_attr = "attention_check_answered"
        failed_attr = "attention_check_failed"
        radio_key = "attention_check"
    else:
        target_attr = "attention_check_2_target"
        answered_attr = "attention_check_2_answered"
        failed_attr = "attention_check_2_failed"
        radio_key = "attention_check_2"

    if getattr(session, target_attr) is None:
        random.seed()
        setattr(session, target_attr, random.randint(0, 2))

    target_value = getattr(session, target_attr)
    target_option = option_labels[target_value]
    st.markdown(f"Please select **{target_option}** below to continue.")

    default_index = target_value if getattr(session, answered_attr) else None

    selected = st.radio(
        "Your selection:",
        options=range(len(attention_options)),
        format_func=lambda i: f"**{option_labels[i]}:**\n\n{attention_options[i]}\n",
        key=radio_key,
        index=default_index,
    )

    if selected is not None:
        setattr(session, answered_attr, True)
        if selected != target_value:
            setattr(session, failed_attr, True)
        else:
            setattr(session, failed_attr, False)


def _render_question(session, question_data: dict, question_num: int):
    """Render a single survey question."""
    is_dev = _get_environment_mode() == "development"

    st.markdown(f"### Question {question_num} of 17")

    if is_dev:
        with st.expander("ℹ️ Question metadata (for reference)"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.caption(f"**Dataset:** {question_data.get('dataset', 'N/A')}")
            with col2:
                st.caption(f"**Mechanism:** {question_data.get('mechanism', 'N/A')}")
            with col3:
                st.caption(f"**Epsilon:** {question_data.get('epsilon', 'N/A')}")
            with col4:
                st.caption(f"**Variant:** {question_data.get('variant_id', 'N/A')}")

    st.markdown("##### 📄 Original Text:")
    st.info(question_data["reference_text"])

    st.markdown("###### Choose the single MOST ACCEPTABLE alternative:")
    st.caption(
        "For each original text above, select the one alternative that you believe best preserves the original meaning while also being: most privacy-preserving (i.e., in your view reveals the least sensitive information), clearest and easiest to understand, most coherent and fluent, and simplest to read."
    )

    # Shuffle options deterministically
    variant_id = question_data["variant_id"]
    rng = random.Random(variant_id)
    shuffled_options = question_data["options"].copy()
    rng.shuffle(shuffled_options)

    options_text = []
    for display_idx, opt in enumerate(shuffled_options):
        options_text.append(
            f"**Option {display_idx + 1}:**\n\n{opt['privatized_text']}\n"
        )

    existing_answer = session.answers.get(variant_id)
    default_index = None
    if existing_answer is not None:
        for idx, opt in enumerate(shuffled_options):
            if opt["tertile"] == existing_answer:
                default_index = idx
                break

    selected = st.radio(
        "Your selection:",
        options=range(len(options_text)),
        format_func=lambda i: options_text[i],
        key=f"q_{question_num}_{variant_id}",
        index=default_index,
    )

    if selected is not None:
        return shuffled_options[selected]["tertile"]
    return None


def _get_environment_mode() -> str:
    """Get environment mode from config."""
    try:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), ".streamlit", "config.toml"
        )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = toml.load(f)
                return config.get("server", {}).get("environment", "production")
    except Exception:
        pass
    return os.getenv("ENVIRONMENT", "production")


def go_to_background():
    """Callback to save draft and return to background page."""
    session = st.session_state.survey
    save_draft(session)
    session.current_page = "background"
    st.session_state.navigate_to = "pages/2_background.py"


def go_to_feedback():
    """Callback to save draft and proceed to feedback page."""
    session = st.session_state.survey
    answered_count = (
        len(session.answers)
        + (1 if session.attention_check_answered else 0)
        + (1 if session.attention_check_2_answered else 0)
    )

    if answered_count < 17:
        st.session_state.questions_error = (
            f"❌ Please answer all questions ({answered_count}/17 completed)"
        )
        return

    save_draft(session)
    session.current_page = "feedback"
    st.session_state.questions_error = None
    st.session_state.navigate_to = "pages/4_feedback.py"


session = initialize_session()

st.title("📝 Private Text Generation Survey")

if not session.user_id:
    st.warning("⚠️ Please start from the beginning.")
    session.current_page = "welcome"
    st.switch_page("pages/1_welcome.py")

# Check if survey completed (cached)
if "submission_checked_questions" not in st.session_state:
    st.session_state.submission_checked_questions = is_submission_complete(
        session.user_id
    )

if st.session_state.submission_checked_questions:
    st.info(
        "✅ You have already completed this survey. Thank you for your participation!"
    )
    st.stop()

# Load questions
if "questions_ensured" not in st.session_state:
    _ensure_questions_loaded(session)
    st.session_state.questions_ensured = True

if not session.questions:
    st.error("Unable to load questions. Please try again.")
    st.stop()

if session.attention_check_position is None:
    random.seed()
    session.attention_check_position = random.randint(5, 8)

if session.attention_check_2_position is None:
    random.seed()
    session.attention_check_2_position = random.randint(10, 14)

progress = session.get_progress()
st.progress(
    progress, text=f"Answer all 17 questions below (including 2 attention checks)"
)
st.warning(
    """**IMPORTANT:** Focus on the *general quality* of the alternative texts, you do **NOT** need to analyze every detail of each option — pick the option that feels best overall without getting bogged down in content specifics."""
)

st.markdown(
    f"""
**Participant ID:** `{session.user_id}`

**Instructions:**
1. For each question you will see the **original text** followed by three alternative rephrasals (options).
2. Choose the single alternative you find **MOST ACCEPTABLE** — the one that **best preserves the meaning of the original text** while being: 
    - most private (i.e., least revealing of sensitive information in your opinion),
    - easiest to understand (clearest in topic), 
    - most coherent and fluent, and
    - easiest to read.
3. Answer all 17 questions (including 2 attention checks). You cannot change your answers after submission.

Please use your own understanding of “privacy” — we do not provide a definition because we want your natural judgment about what feels private.

**Estimated time remaining:** 5–10 minutes
"""
)

st.divider()

display_num = 1
for i, question_data in enumerate(session.questions, 1):  # type: ignore
    if display_num == session.attention_check_position:
        with st.container(border=True):
            _render_attention_check(session, display_num, check_number=1)
        display_num += 1

    if display_num == session.attention_check_2_position:
        with st.container(border=True):
            _render_attention_check(session, display_num, check_number=2)
        display_num += 1

    with st.container(border=True):
        selected = _render_question(session, question_data, display_num)
        if selected is not None:
            session.answers[question_data["variant_id"]] = selected
    display_num += 1

answered_count = (
    len(session.answers)
    + (1 if session.attention_check_answered else 0)
    + (1 if session.attention_check_2_answered else 0)
)
st.progress(
    min(answered_count / 17, 1.0),
    text=f"Progress: {answered_count}/17 questions answered",
)

if "questions_error" in st.session_state and st.session_state.questions_error:
    st.error(st.session_state.questions_error)

with st.form("navigation_form"):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.form_submit_button("⏮️ Previous", type="secondary", on_click=go_to_background)

    with col2:
        st.form_submit_button(
            "Next ⏭️", type="primary", use_container_width=True, on_click=go_to_feedback
        )

if "navigate_to" in st.session_state and st.session_state.navigate_to:
    target = st.session_state.navigate_to
    st.session_state.navigate_to = None
    st.switch_page(target)
