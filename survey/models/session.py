"""Survey session model - single source of truth for session state."""
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from datetime import datetime


@dataclass
class SurveySession:
    """Canonical representation of survey session state."""
    
    # Identity
    user_id: Optional[str] = None
    participant_type: Optional[str] = None
    
    # Navigation
    current_page: str = "welcome"
    
    # Timing
    start_time: Optional[datetime] = None
    
    # Answers (keyed by variant_id)
    answers: Dict[str, int] = field(default_factory=dict)
    
    # Background demographics
    background: Dict[str, str] = field(default_factory=dict)
    background_submitted: bool = False
    
    # Question assignment
    assigned_variant_ids: Optional[List[str]] = None
    questions: Optional[List[Dict[str, Any]]] = None  # Loaded question documents
    
    # Attention checks (2 checks total)
    attention_check_position: Optional[int] = None
    attention_check_target: Optional[int] = None
    attention_check_answered: bool = False
    attention_check_failed: bool = False
    
    attention_check_2_position: Optional[int] = None
    attention_check_2_target: Optional[int] = None
    attention_check_2_answered: bool = False
    attention_check_2_failed: bool = False
    
    # Metadata
    countries: List[str] = field(default_factory=list)
    cleanup_done: bool = False
    draft_loaded: bool = False
    
    # Flags
    needs_persist: bool = False  # Signal that draft should be saved
    passed: bool = False  # Attention check passed
    
    def is_ready_to_submit(self) -> bool:
        """Check if all required data is present for submission."""
        return (
            self.user_id is not None
            and self.background_submitted
            and len(self.answers) >= 15
            and self.attention_check_answered
            and self.attention_check_2_answered
        )
    
    def get_progress(self) -> float:
        """Calculate completion progress (0.0 to 1.0)."""
        total_questions = 17  # 15 + 2 attention checks
        answered = len(self.answers) + (1 if self.attention_check_answered else 0) + (1 if self.attention_check_2_answered else 0)
        return answered / total_questions
    
    def to_dict(self) -> dict:
        """Convert to dictionary for MongoDB persistence."""
        return {
            "user_id": self.user_id,
            "participant_type": self.participant_type,
            "current_page": self.current_page,
            "start_time": self.start_time,
            "answers": self.answers,
            "background": self.background,
            "background_submitted": self.background_submitted,
            "assigned_variant_ids": self.assigned_variant_ids,
            "attention_check_position": self.attention_check_position,
            "attention_check_target": self.attention_check_target,
            "attention_check_answered": self.attention_check_answered,
            "attention_check_failed": self.attention_check_failed,
            "attention_check_2_position": self.attention_check_2_position,
            "attention_check_2_target": self.attention_check_2_target,
            "attention_check_2_answered": self.attention_check_2_answered,
            "attention_check_2_failed": self.attention_check_2_failed,
            "passed": self.passed,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SurveySession":
        """Reconstruct session from MongoDB document."""
        return cls(
            user_id=data.get("user_id"),
            participant_type=data.get("participant_type"),
            current_page=data.get("current_page", "welcome"),
            start_time=data.get("start_time"),
            answers=data.get("answers", {}),
            background=data.get("background", {}),
            background_submitted=data.get("background_submitted", False),
            assigned_variant_ids=data.get("assigned_variant_ids"),
            attention_check_position=data.get("attention_check_position"),
            attention_check_target=data.get("attention_check_target"),
            attention_check_answered=data.get("attention_check_answered", False),
            attention_check_failed=data.get("attention_check_failed", False),
            attention_check_2_position=data.get("attention_check_2_position"),
            attention_check_2_target=data.get("attention_check_2_target"),
            attention_check_2_answered=data.get("attention_check_2_answered", False),
            attention_check_2_failed=data.get("attention_check_2_failed", False),
            passed=data.get("passed", False),
            draft_loaded=True,  # Mark that this came from persistence
        )
