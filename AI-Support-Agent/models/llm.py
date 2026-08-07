"""
models/llm.py
=============

Local, offline language model wrapper using Hugging Face Transformers.

This module loads a small Seq2Seq model (``google/flan-t5-small`` by default)
and provides a single public function, ``generate_answer``, that turns a fully
formed prompt into a generated answer string.

Design goals:

    * Fully offline: models come from the local Hugging Face cache only.
    * Lazily loaded singletons: the tokenizer and model are loaded once per
      process and reused.
    * Deterministic-ish generation: low temperature and ``do_sample=False`` by
      default to keep answers grounded and stable.
    * Clean output: special tokens are stripped and whitespace is normalised.

The model is intentionally small and runs on CPU, trading a little raw quality
for near-instant startup and complete privacy.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import Models
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons (lazily initialised)
# ---------------------------------------------------------------------------
_tokenizer: AutoTokenizer | None = None
_model: AutoModelForSeq2SeqLM | None = None


def _load_llm() -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """
    Load the tokenizer and model once, returning cached instances.

    Returns
    -------
    tuple[AutoTokenizer, AutoModelForSeq2SeqLM]
        The cached tokenizer and model.
    """
    global _tokenizer, _model  # noqa: PLW0603
    if _tokenizer is None or _model is None:
        logger.info(
            "Loading local LLM '%s' (this may take a moment)...",
            Models.LLM_NAME,
        )
        _tokenizer = AutoTokenizer.from_pretrained(Models.LLM_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(Models.LLM_NAME)
        _model.eval()  # Inference mode.
        logger.info("LLM loaded successfully.")
    return _tokenizer, _model


def generate_answer(prompt: str) -> str:
    """
    Generate an answer for a fully formed prompt using the local LLM.

    Parameters
    ----------
    prompt : str
        The complete prompt to send to the model (e.g. from
        ``utils.prompts.build_generation_prompt``).

    Returns
    -------
    str
        The generated answer with special tokens and excess whitespace removed.

    Raises
    ------
    RuntimeError
        If the model fails to load or generate (e.g. out-of-memory).
    """
    tokenizer, model = _load_llm()

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=Models.MAX_NEW_TOKENS,
                do_sample=False,
                temperature=Models.TEMPERATURE,
                top_p=Models.TOP_P,
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Normalise whitespace for a clean final string.
        answer = " ".join(answer.split())
        logger.debug("Generated answer (%d chars).", len(answer))
        return answer

    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM generation failed: %s", exc)
        raise RuntimeError("Local LLM generation failed.") from exc


def release_llm() -> None:
    """
    Free the LLM from memory.

    Useful in tests or memory-constrained environments. The model is lazily
    re-loaded on the next ``generate_answer`` call.
    """
    global _tokenizer, _model  # noqa: PLW0603
    _tokenizer = None
    _model = None
    logger.info("LLM released from memory.")
