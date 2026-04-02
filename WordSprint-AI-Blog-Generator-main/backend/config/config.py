from typing import Dict
from pathlib import Path

DOMAIN_MAP: Dict[str, str] = {
    "machine_learning.json": "ml",
    "deep_learning.json": "dl",
    "statistics.json": "statistics",
    "natural_language_processing.json": "nlp",
    "computer_vision.json": "cv",
    "generative_ai.json": "genai"
    }

CATEGORY_META: Dict[str, Dict[str, str]] = {
    "ml": {"label": "Machine Learning", "shortLabel": "ML"},
    "dl": {"label": "Deep Learning", "shortLabel": "DL"},
    "nlp": {"label": "Natural Language Processing", "shortLabel": "NLP"},
    "cv": {"label": "Computer Vision", "shortLabel": "CV"},
    "genai": {"label": "Generative AI", "shortLabel": "Gen AI"},
    "ainews": {"label": "AI News", "shortLabel": "AI News"},
    "statistics": {"label": "Statistics for AI", "shortLabel": "Stats"}
}

MODEL: str = "llama-3.3-70b-versatile"
TEMPERATURE: float = 1.0
MAX_TOKENS:  int   = 4096
WORDS_PER_MINUTE: int = 200

BACKEND_DIR: Path = Path(__file__).parent.parent
ROOT_DIR:    Path = BACKEND_DIR.parent
INPUT_DIR:     Path = BACKEND_DIR / "data" / "input"
SCHEDULE_FILE: Path = BACKEND_DIR / "schedule.json"
PROMPT_DIRECTORY: Path = BACKEND_DIR / "prompts"
BLOGS_DIR: Path = ROOT_DIR / "frontend" / "blogs"