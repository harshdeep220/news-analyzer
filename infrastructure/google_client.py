"""
Google AI Studio Client — Thin wrapper over google-genai SDK.

Two methods:
  - generate(prompt, model, system) → str
  - embed(text) → list[float]

Both wrapped in tenacity exponential backoff. No model strings here — 
all model IDs come from config.py.
"""

import logging
import time

from google import genai
from google.genai import types as genai_types
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from config import (
    GOOGLE_API_KEY,
    FAST_MODEL,
    EMBED_MODEL,
    EMBED_DIM,
    EMBED_MAX_TOKENS,
    EMBED_OUTPUT_DIM,
    RETRY_MIN_WAIT,
    RETRY_MAX_WAIT,
    RETRY_MAX_ATTEMPTS,
)

logger = logging.getLogger(__name__)

# ─── Module-level client ──────────────────────────────────────────────────────
_client = None


def _get_client():
    """Lazy-init the google-genai client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


# ─── Retry decorator ─────────────────────────────────────────────────────────
_retry_decorator = retry(
    wait=wait_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    retry=retry_if_exception_type(Exception),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class GoogleClient:
    """Thin wrapper over google-genai SDK for LLM generation and embedding."""

    def __init__(self):
        self.client = _get_client()

    @_retry_decorator
    def generate(self, prompt: str, model: str = None, system: str = None) -> str:
        """
        Generate text completion from a Gemini model.

        Args:
            prompt: The user prompt / input text.
            model: Model ID (default: FAST_MODEL from config).
            system: Optional system instruction.

        Returns:
            The generated text response as a string.
        """
        model = model or FAST_MODEL
        start_time = time.time()

        config = None
        if system:
            config = genai_types.GenerateContentConfig(
                system_instruction=system,
            )

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        elapsed = time.time() - start_time
        logger.info(f"generate() [{model}] completed in {elapsed:.2f}s")

        return response.text

    @_retry_decorator
    def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for text using gemini-embedding-2.

        Uses Matryoshka Representation Learning (MRL) to output 768-dim
        vectors for compatibility with our ChromaDB collections.

        Args:
            text: Input text to embed.

        Returns:
            List of floats (768-dimensional vector).
        """
        # Safety cap — model supports 8192 tokens, we keep conservative
        word_count = len(text.split())
        if word_count > EMBED_MAX_TOKENS:
            logger.warning(
                f"embed() input has {word_count} words, exceeds {EMBED_MAX_TOKENS} cap. Truncating."
            )
            text = " ".join(text.split()[:EMBED_MAX_TOKENS])

        response = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=genai_types.EmbedContentConfig(
                output_dimensionality=EMBED_OUTPUT_DIM,
            ),
        )

        vector = response.embeddings[0].values
        assert len(vector) == EMBED_DIM, (
            f"Expected {EMBED_DIM}-dim embedding, got {len(vector)}"
        )

        return list(vector)
