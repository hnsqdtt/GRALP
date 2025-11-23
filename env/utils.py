import json
import re
import logging
import json
import logging
import os
import re
from typing import Any, Dict


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("rl_env")
    if not logger.handlers:
        level = os.environ.get("RL_ENV_LOG", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        ch = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


LOGGER = _build_logger()


def _strip_json_comments_and_trailing_commas(text: str) -> str:
    # Strip // and /* */ comments and trailing commas.
    text = re.sub(r"(^|[^\\])//.*", r"\1", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def load_json_config(path: str, *, strict: bool = False) -> Dict[str, Any]:
    """Load a JSON config file.
    - strict=False (default): missing file returns an empty dict
    - strict=True: missing file raises FileNotFoundError
    Parse errors always raise an exception.
    """
    if not os.path.exists(path):
        if strict:
            raise FileNotFoundError(f"Config not found: {path}")
        LOGGER.warning("Config not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        try:
            cleaned = _strip_json_comments_and_trailing_commas(data)
            return json.loads(cleaned)
        except Exception:
            raise
