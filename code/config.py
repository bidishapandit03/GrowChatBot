"""Allowlist, data paths, and non-secret settings. Secrets stay in the environment."""

from pathlib import Path
from urllib.parse import urlparse, urlunparse

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_HTML_DIR = DATA_DIR / "raw" / "html"
RAW_DOCUMENTS_DIR = DATA_DIR / "raw" / "documents"
CHUNKS_DIR = DATA_DIR / "chunks"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
CHROMA_DIR = DATA_DIR / "chroma"

HTTP_TIMEOUT_SECONDS = 30
HTTP_MAX_RETRIES = 2
HTTP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

LOAD_STATUS_SUCCESSFUL = "successful"
LOAD_STATUS_UNAVAILABLE = "unavailable"
LOAD_STATUS_VALIDATION_FAILED = "validation_failed"

APPROVED_SOURCES = [
    {
        "fund_category": "large-cap",
        "fund_name": "HDFC Large Cap Fund Direct Growth",
        "canonical_url": "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    },
    {
        "fund_category": "flexi-cap",
        "fund_name": "HDFC Flexi Cap Fund Direct Growth",
        "canonical_url": "https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth",
    },
    {
        "fund_category": "elss",
        "fund_name": "HDFC ELSS Tax Saver Fund Direct Plan Growth",
        "canonical_url": "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth",
    },
    {
        "fund_category": "small-cap",
        "fund_name": "HDFC Small Cap Fund Direct Growth",
        "canonical_url": "https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth",
    },
    {
        "fund_category": "hybrid",
        "fund_name": "HDFC Balanced Advantage Fund Direct Growth",
        "canonical_url": "https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth",
    },
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE_TOKENS = (300, 500)
CHUNK_OVERLAP_TOKENS = (50, 75)
TOP_K = 4
# Working default until Phase 6 calibration on the labelled retrieval set. Cosine
# distance (0 = identical); known-fact matches cluster ~0.20-0.55 for this corpus.
RELEVANCE_THRESHOLD = 0.55
CHROMA_COLLECTION_NAME = "hdfc_groww_funds"

MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_GENERATION_TIMEOUT_SECONDS = 90
MISTRAL_MAX_TOKENS = 300
MISTRAL_TEMPERATURE = 0.0
MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
MAX_ANSWER_SENTENCES = 3


def source_slug(canonical_url: str) -> str:
    path = urlparse(canonical_url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1]
    if not slug:
        raise ValueError(f"Cannot derive slug from URL: {canonical_url}")
    return slug


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((scheme, netloc, path, "", "", ""))


def is_corresponding_approved_url(final_url: str, canonical_url: str) -> bool:
    """Redirects are valid only when they resolve to this source's approved page."""
    return normalize_url(final_url) == normalize_url(canonical_url)
