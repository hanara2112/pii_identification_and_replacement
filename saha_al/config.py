"""
SAHA-AL Configuration
Central source of truth for all paths, thresholds, entity types, and settings.
Nothing should be hardcoded elsewhere — import from here.
"""

import os

# ─── BASE DIRECTORY ───
# All paths are relative to the saha_al/ directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── DATA PATHS ───
ORIGINAL_DATA_PATH = os.path.join(BASE_DIR, "data", "original_data.jsonl")
PRE_ANNOTATED_PATH = os.path.join(BASE_DIR, "data", "pre_annotated.jsonl")
GREEN_QUEUE_PATH = os.path.join(BASE_DIR, "data", "queues", "green_queue.jsonl")
YELLOW_QUEUE_PATH = os.path.join(BASE_DIR, "data", "queues", "yellow_queue.jsonl")
RED_QUEUE_PATH = os.path.join(BASE_DIR, "data", "queues", "red_queue.jsonl")
GOLD_STANDARD_PATH = os.path.join(BASE_DIR, "data", "gold_standard.jsonl")
SKIPPED_PATH = os.path.join(BASE_DIR, "data", "skipped.jsonl")
FLAGGED_PATH = os.path.join(BASE_DIR, "data", "flagged.jsonl")
BACKUP_DIR = os.path.join(BASE_DIR, "data", "backups")

# ─── LOG PATHS ───
ANNOTATION_LOG_PATH = os.path.join(BASE_DIR, "logs", "annotation_log.jsonl")
MODEL_VERSIONS_PATH = os.path.join(BASE_DIR, "logs", "model_versions.jsonl")

# ─── MODEL PATHS ───
BOOTSTRAP_MODEL_DIR = os.path.join(BASE_DIR, "models", "bootstrap")

# ─── SPACY MODEL ───
# Use "en_core_web_trf" if GPU available, "en_core_web_lg" otherwise
SPACY_MODEL = "en_core_web_lg"

# ─── ENTITY TYPE TAXONOMY (authoritative list) ───
ENTITY_TYPES = [
    "FULLNAME",
    "FIRST_NAME",
    "LAST_NAME",
    "ID_NUMBER",
    "PASSPORT",
    "SSN",
    "PHONE",
    "EMAIL",
    "ADDRESS",
    "DATE",
    "TIME",
    "LOCATION",
    "ORGANIZATION",
    "ACCOUNT_NUMBER",
    "CREDIT_CARD",
    "ZIPCODE",
    "TITLE",
    "GENDER",
    "NUMBER",
    "OTHER_PII",
    "UNKNOWN",
]

# ─── SPACY LABEL → SAHA TAXONOMY MAPPING ───
SPACY_TO_SAHA = {
    "PERSON": "FULLNAME",
    "GPE": "LOCATION",
    "ORG": "ORGANIZATION",
    "DATE": "DATE",
    "TIME": "TIME",
    "NORP": "OTHER_PII",
    "LOC": "LOCATION",
    "FAC": "ADDRESS",
    # Skip non-PII types
    "CARDINAL": None,
    "ORDINAL": None,
    "MONEY": None,
    "PERCENT": None,
    "QUANTITY": None,
    "EVENT": None,
    "WORK_OF_ART": None,
    "LAW": None,
    "LANGUAGE": None,
    "PRODUCT": None,
}

# ─── ROUTING THRESHOLDS ───
CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.50

# ─── ACTIVE LEARNING ───
BOOTSTRAP_INITIAL_SIZE = 200   # minimum gold entries before first bootstrap train
RETRAIN_EVERY_N = 500          # retrain bootstrap every N new gold entries
UNCERTAINTY_THRESHOLD = 0.40

# ─── QUALITY CHECK THRESHOLDS ───
MAX_LENGTH_RATIO = 1.5   # anonymized text shouldn't be 1.5x longer than original
MIN_LENGTH_RATIO = 0.5   # or less than half

# ─── TITLE AND GENDER CONSTANTS ───
TITLES = {
    "Mr", "Mrs", "Ms", "Miss", "Dr", "Prof",
    "Mister", "Madame", "Master", "Mstr",
    "Sir", "Mayoress", "Mayor", "Principal",
    "Solicitor", "Senior", "Junior",
}

GENDERS = {"Male", "Female", "Non-binary", "Other", "M", "F"}

# ─── AUGMENTATION PATHS ───
AUGMENTED_DATA_PATH = os.path.join(BASE_DIR, "data", "augmented_data.jsonl")
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "data", "training_data.jsonl")

# ─── AUGMENTATION DEFAULTS ───
AUG_ENTITY_SWAP_MULTIPLIER = 4     # Dai & Adel, COLING 2020
AUG_TEMPLATE_FILL_COUNT = 5000     # Anaby-Tavor et al., AAAI 2020
AUG_EDA_MULTIPLIER = 3             # Wei & Zou, EMNLP 2019
AUG_EDA_ALPHA = 0.1                # Fraction of words changed per EDA operation
