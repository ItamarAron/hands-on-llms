from enum import Enum
from pathlib import Path

from transformers import TrainingArguments

class Scope(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "test"


# TODO: Use Hydra as a configuration management tool.
# TODO: Configure this path instead of hardcoding it.
ROOT_DIR = Path("/home/pauliusztin/Documents/projects/hands-on-llms/modules/training")
ROOT_DATASET_DIR_DEFAULT = ROOT_DIR / "dataset"
MODEL_ID_DEFAULT = "tiiuae/falcon-7b-instruct"

RESULT_DIR_DEFAULT = ROOT_DIR / "results"
LOGGING_DIR_DEFAULT = ROOT_DIR / "logs"
DEFAULT_TRAINING_ARGUMENTS = TrainingArguments(
        output_dir=str(RESULT_DIR_DEFAULT),
        logging_dir=str(LOGGING_DIR_DEFAULT),
        per_device_train_batch_size=1,  # increase this value if you have more VRAM
        gradient_accumulation_steps=3,
        per_device_eval_batch_size=1,  # increase this value if you have more VRAM
        optim="paged_adamw_32bit",  # This parameter activate QLoRa's pagination # TODO: Should we use paged_adamw_8bit instead?
        save_steps=40,
        logging_steps=1,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        evaluation_strategy="steps",
        eval_steps=3,
        report_to="comet_ml",
        seed=42,
    )