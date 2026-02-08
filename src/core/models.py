from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
import json

class BaseCCModel(BaseModel):
    """Base model for Creator Catalyst with common configuration."""
    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_row(cls, row):
        """Helper to create a model from a sqlite3.Row object."""
        if row is None:
            return None
        return cls.model_validate(dict(row))

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for backward compatibility."""
        return self.model_dump()

class TranscriptSegment(BaseCCModel):
    """Represents a single SRT caption segment with timing."""
    index: int
    start_time: str
    end_time: str
    text: str

    def to_seconds(self, time_str: Optional[str] = None) -> float:
        """Convert SRT timestamp to seconds."""
        t = time_str or self.start_time
        # Format: HH:MM:SS,mmm or MM:SS,mmm
        t = t.replace(',', '.')
        parts = t.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return 0.0

class TitleSuggestion(BaseCCModel):
    """Represents a single title suggestion with metadata."""
    title: str
    style: str  # e.g., "curiosity", "listicle", "how-to", "secret"
    hook_type: str  # e.g., "question", "number", "power_word", "mystery"
    estimated_ctr: str  # "high", "medium", "low"

class TitleGenerationResult(BaseCCModel):
    """Contains all generated titles for a video/short."""
    original_titles: List[TitleSuggestion] = Field(default_factory=list)
    shorts_titles: Dict[int, List[TitleSuggestion]] = Field(default_factory=dict)
    selected_original_title: Optional[str] = None
    selected_shorts_titles: Optional[Dict[int, str]] = None

class Video(BaseCCModel):
    """Represents a processed video."""
    id: Optional[int] = None
    filename: str = ""
    file_path: str = ""
    file_size_mb: float = 0.0
    duration_seconds: Optional[int] = None
    uploaded_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    platform: str = "General"
    grounding_enabled: bool = True
    processing_status: str = "pending"  # pending, processing, completed, failed
    searchable_text: str = ""

class ContentOutput(BaseCCModel):
    """Represents a generated content output."""
    id: Optional[int] = None
    video_id: int = 0
    content_type: str = ""  # captions, blog_post, social_post, shorts_idea, thumbnail_idea
    content: str = ""
    metadata: Union[Dict[str, Any], str] = Field(default_factory=dict)
    version: int = 1
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    grounding_rate: Optional[float] = None
    validation_status: Optional[str] = None

    @field_validator('metadata', mode='before')
    @classmethod
    def parse_metadata(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return {}
        return v

class GroundingReport(BaseCCModel):
    """Represents a fact-grounding validation report."""
    id: Optional[int] = None
    video_id: int = 0
    blog_grounding_rate: float = 0.0
    social_grounding_rate: float = 0.0
    shorts_verification_rate: float = 0.0
    total_claims: int = 0
    verified_claims: int = 0
    unverified_claims: int = 0
    full_report: Union[Dict[str, Any], str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @field_validator('full_report', mode='before')
    @classmethod
    def parse_full_report(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return {}
        return v

class AIRequest(BaseCCModel):
    """Represents a single AI API request."""
    id: Optional[int] = None
    user_id: str = "default_user"
    endpoint: str = ""
    provider: str = ""
    operation_type: str = ""
    tokens_used: int = 0
    cost_credits: float = 0.0
    cost_usd: float = 0.0
    response_time_ms: int = 0
    success: bool = True
    error_message: Optional[str] = None
    request_metadata: Union[Dict[str, Any], str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @field_validator('request_metadata', mode='before')
    @classmethod
    def parse_request_metadata(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except:
                return {}
        return v

class UserCredits(BaseCCModel):
    """Represents user credit balance."""
    id: Optional[int] = None
    user_id: str = "default_user"
    current_balance: int = 0
    total_earned: int = 0
    total_spent: int = 0
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())

class CreditTransaction(BaseCCModel):
    """Represents a single credit transaction."""
    id: Optional[int] = None
    user_id: str = "default_user"
    transaction_type: str = ""
    amount: int = 0
    balance_after: int = 0
    operation_type: Optional[str] = None
    description: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())

class ShortsIdea(BaseCCModel):
    """Represents a generated shorts idea."""
    topic: str = ""
    start_time: str = ""
    end_time: str = ""
    summary: str = ""
    hook: str = ""
    script_outline: str = ""
    visual_cues: str = ""
    estimated_duration: str = ""
    validation_status: Optional[str] = None
    validation_badge: Optional[str] = None

class EngagementScore(BaseCCModel):
    """Represents an engagement score with detailed breakdown."""
    overall_score: int  # 0-100
    platform_scores: Dict[str, int] = Field(default_factory=dict)
    recommended_platform: str = ""
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    engagement_factors: Dict[str, float] = Field(default_factory=dict)
    sentiment: Dict[str, float] = Field(default_factory=dict)
    readability_score: float = 0.0
    virality_score: float = 0.0
    optimal_posting_time: Optional[str] = None
