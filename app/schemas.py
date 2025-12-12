from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# ==========================================
# 1. CHAT & ROUTING SYSTEM
# ==========================================

class ChatMessage(BaseModel):
    """Format pesan tunggal untuk riwayat chat (History)."""
    role: str       # "user" atau "assistant"
    content: str

class ChatRequest(BaseModel):
    """
    Payload utama yang dikirim Frontend saat user chatting.
    Backend butuh 'current_skills' dan 'role' karena Backend tidak punya Database.
    """
    message: str                        # Pesan user saat ini
    role: str                           # Contoh: "AI Engineer"
    history: List[ChatMessage] = []     # 5 pesan terakhir untuk konteks percakapan
    current_skills: Dict[str, str] = {} # Contoh: {"python": "intermediate", "nlp": "beginner"}

class ChatResponse(BaseModel):
    """Balasan dari Backend ke Frontend."""
    reply: str                          # Teks balasan bot
    action_type: str                    # "START_EXAM", "GET_RECOMMENDATION", "CASUAL_CHAT", "START_PSYCH_TEST"
    data: Optional[Dict[str, Any]] = None # Data tambahan (Soal ujian / List Rekomendasi)

# ==========================================
# 2. EXAM SYSTEM (UJIAN)
# ==========================================

class QuestionResponse(BaseModel):
    """Output soal dari LLM."""
    question_text: str
    question_context: Dict[str, Any]    # Kunci jawaban/Rubrik (Frontend wajib simpan ini)
    skill_id: str

class AnswerSubmission(BaseModel):
    """Payload saat user mengirim jawaban ujian."""
    user_answer: str
    question_context: Dict[str, Any]    # Kunci jawaban yang dikirim balik oleh Frontend

class EvaluationResponse(BaseModel):
    """Hasil penilaian AI Judge."""
    is_correct: bool
    score: int
    feedback: str
    passed: bool                        # True jika score >= 70
    suggested_new_level: Optional[str] = None # Saran level baru (misal: "intermediate")

# ==========================================
# 3. RECOMMENDATION SYSTEM (ML POWERED)
# ==========================================

class SkillGap(BaseModel):
    skill_name: str      # Contoh: "SQL"
    target_level: str    # Contoh: "Pemula"

class UserProfile(BaseModel):
    name: str            # Contoh: "Siti Adaptive"
    active_path: str     # Contoh: "Data Scientist"
    missing_skills: List[SkillGap] # List of objects
    completed_courses: List[int] = []

class RecommendationItem(BaseModel):
    skill: str
    current_level: str
    course_to_take: str
    chapters: List[str]
    match_score: float
    badge: str
# ==========================================
# 4. PROGRESS SYSTEM
# ==========================================

class ProgressRequest(BaseModel):
    """Request untuk hitung progress bar."""
    role: str
    current_skills: Dict[str, str]

class ProgressItem(BaseModel):
    """Format satu item progress skill."""
    skill_name: str
    current_level: str
    progress_percent: int               # 0 - 100
    remaining_tutorials: int

class ProgressData(BaseModel):
    user_name: str
    active_courses: List[Dict[str, Any]] # Isinya: course_name, progress_percent, dll
    skill_updates: List[Dict[str, Any]]  # Isinya data dari skill_history
    current_skills: List[Dict[str, Any]]

# ==========================================
# 5. PSYCH TEST SYSTEM (Fitur Baru)
# ==========================================

class PsychQuestionItem(BaseModel):
    id: int
    question: str
    options: Dict[str, str] # {"A": "...", "B": "..."}

class PsychSubmitRequest(BaseModel):
    """Format jawaban yang dikirim user. Key = ID Soal, Value = Pilihan (A/B)"""
    answers: Dict[int, str]  # Contoh: {1: "A", 2: "B", 3: "A", ...}

class PsychResultResponse(BaseModel):
    suggested_role: str
    analysis: str            # Penjelasan dari LLM
    scores: Dict[str, int]   # Skor detail (misal: AI=3, Frontend=2)