import re
import logging
import asyncio
from typing import Tuple, Dict, Any
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import APIError
from app.config import settings
from app.models.schemas import FigureExtraction, VectorPageExtraction

logger = logging.getLogger("papervision")

RASTER_FIGURE_PROMPT = """
You are an expert AI figure parser called PaperVision.
Analyze the provided image, which is a figure extracted from a research paper, and extract structured information from it.

CRITICAL: Do NOT extract logos, publisher brandings, cover branding, watermarks, or decorative layout icons. 
If the provided image is a publisher logo (e.g. BMJ, Elsevier), journal branding (e.g. "BMJ Open"), editorial banner, watermark, or submission site banner (e.g. "SCHOLARONE Manuscripts"), you must classify its type as 'unknown' and set the confidence score strictly to 0.0.

Otherwise, first classify the figure type into one of these categories:
- chart
- graph
- table_image
- flowchart
- scientific_diagram
- equation_image
- unknown

Based on your classification, populate the appropriate sub-field in the schema:
- If 'chart' or 'graph': populate 'chart_graph_data' with the title, X-axis label and scale/values, Y-axis label and scale/values, legends/series keys, and visible annotations.
- If 'table_image': populate 'table_data' with column headers and rows of cell values.
- If 'flowchart' or 'scientific_diagram': populate 'diagram_data' with node labels, edge/connector labels, and semantic relationships.
- If 'equation_image': populate 'equation_data' with the LaTeX string representing the formula.
- If 'unknown' or too complex: populate 'general_description' with a textual summary of everything visible in the image.

Ensure that only the matching sub-field is populated, while the others remain null. Return the JSON object matching the requested schema exactly.
"""

VECTOR_PAGE_PROMPT = """
You are an expert AI figure parser called PaperVision.
The provided image is a rendered full page of a research paper PDF containing vector-drawn elements.

Your task is to:
1. Scan the entire page to locate and isolate any meaningful vector figures (e.g. vector charts, line graphs, schematics, vector diagrams, tables, or highlighted equations).
2. Ignore all surrounding body text columns, paragraph text, running headers, running footers, page numbers, references, or publisher branding watermarks.
3. CRITICAL: Do NOT extract or return logos, publisher brandings, cover branding, watermarks, or decorative layout icons. If you identify a logo (e.g. BMJ) or branding banner on the page, ignore it completely and do not return it in the list.
4. For each isolated figure you identify:
   a. Classify its type: chart, graph, table_image, flowchart, scientific_diagram, equation_image, or unknown.
   b. Extract the structured details according to its type (e.g., axes, legends, headers, rows, node labels, latex).
   c. Set a confidence score and explain in 'reasoning' why this area was selected and where it is located.

Return a list of all identified figures in the 'figures' array field matching the VectorPageExtraction schema. Ignore general text.
"""


def _parse_retry_delay(error: APIError) -> float | None:
    """
    Attempts to extract the server-recommended retry delay (in seconds) from a
    429 APIError response. Gemini includes this in the error message as
    'retryDelay: Xs'. Returns None if it cannot be parsed.
    """
    try:
        error_str = str(error)
        # Matches patterns like "retryDelay: '33s'" or 'retryDelay: "28.5s"' or "retryDelay: 10s"
        match = re.search(r"retryDelay['\"\s:]+([0-9.]+)s", error_str)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


class GeminiClient:
    """
    Dedicated client class for interfacing with Google's Gemini Vision API.
    Handles image uploads, structured prompts, schema validation, and token usage tracking.
    """
    def __init__(self):
        self._client = None

    @property
    def client(self) -> genai.Client:
        """Lazily instantiates the Google GenAI SDK Client."""
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not configured. "
                "Please add a valid Google Gemini API key to your .env file."
            )
        if self._client is None:
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    def _call_gemini_sync(
        self, image: Image.Image, prompt: str, response_schema: Any
    ) -> Tuple[Any, Dict[str, int]]:
        """
        Synchronous helper that executes the generate_content call against the Gemini API.

        Retry behaviour:
        - 429 RESOURCE_EXHAUSTED: reads the server-recommended retryDelay from the
          error response and waits that long before retrying. If no delay is found,
          falls back to exponential backoff. After max_retries, moves to the next model.
        - 503 UNAVAILABLE: retries with exponential backoff.
        - 404 NOT_FOUND: unretryable — skips immediately to the next model without
          wasting retry attempts.
        - Any other error: skips immediately to the next model.
        """
        import time

        models_to_try = [settings.GEMINI_MODEL] + [
            m for m in settings.GEMINI_FALLBACK_MODELS
            if m != settings.GEMINI_MODEL
        ]

        max_retries = 3
        base_delay = 2.0  # seconds — used only when server provides no retryDelay

        last_error = None

        for model_name in models_to_try:
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(
                        f"Calling Gemini API using model '{model_name}' "
                        f"(Attempt {attempt}/{max_retries})..."
                    )

                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=[image, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=response_schema,
                            temperature=0.1,
                        ),
                    )

                    # Extract token usage from response metadata
                    prompt_tokens = 0
                    candidate_tokens = 0
                    total_tokens = 0

                    if response.usage_metadata:
                        prompt_tokens = (
                            getattr(response.usage_metadata, "prompt_token_count", 0)
                            or getattr(response.usage_metadata, "promptTokenCount", 0)
                            or 0
                        )
                        candidate_tokens = (
                            getattr(response.usage_metadata, "candidates_token_count", 0)
                            or getattr(response.usage_metadata, "candidatesTokenCount", 0)
                            or 0
                        )
                        total_tokens = (
                            getattr(response.usage_metadata, "total_token_count", 0)
                            or getattr(response.usage_metadata, "totalTokenCount", 0)
                            or 0
                        )

                    token_usage = {
                        "prompt": prompt_tokens,
                        "completion": candidate_tokens,
                        "total": total_tokens,
                    }

                    logger.info(
                        f"Gemini API call succeeded with model '{model_name}'. "
                        f"Token usage: {token_usage}"
                    )

                    parsed_response = response_schema.model_validate_json(response.text)
                    return parsed_response, token_usage

                except APIError as ae:
                    last_error = ae
                    status_code = (
                        getattr(ae, "code", 0)
                        or getattr(ae, "status_code", 0)
                        or 0
                    )
                    error_msg = str(ae).lower()

                    is_rate_limit = (
                        status_code == 429
                        or "429" in error_msg
                        or "resource_exhausted" in error_msg
                    )
                    is_server_error = (
                        status_code == 503
                        or "503" in error_msg
                        or "unavailable" in error_msg
                    )
                    is_not_found = (
                        status_code == 404
                        or "404" in error_msg
                        or "not_found" in error_msg
                    )

                    # 404 means the model string is wrong — no point retrying
                    if is_not_found:
                        logger.warning(
                            f"Gemini model '{model_name}' not found (404). "
                            f"Check your model name in .env. Skipping to next model."
                        )
                        break

                    if is_rate_limit and attempt < max_retries:
                        # Respect the server-recommended delay if present
                        server_delay = _parse_retry_delay(ae)
                        if server_delay is not None:
                            delay = server_delay
                            logger.warning(
                                f"Gemini model '{model_name}' rate limited (attempt {attempt}). "
                                f"Server requests waiting {delay:.1f}s. Retrying..."
                            )
                        else:
                            delay = base_delay * (2 ** (attempt - 1))
                            logger.warning(
                                f"Gemini model '{model_name}' rate limited (attempt {attempt}). "
                                f"No server delay found, waiting {delay:.1f}s. Retrying..."
                            )
                        time.sleep(delay)
                        continue

                    if is_server_error and attempt < max_retries:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            f"Gemini model '{model_name}' server error (attempt {attempt}). "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue

                    # Rate limit exhausted all retries, or unretryable error
                    if is_rate_limit:
                        logger.warning(
                            f"Gemini model '{model_name}' rate limit persisted after "
                            f"{max_retries} attempts. Moving to next model."
                        )
                    else:
                        logger.warning(
                            f"Gemini model '{model_name}' unretryable error: {ae}. "
                            f"Moving to next model."
                        )
                    break  # Try next model

                except Exception as e:
                    last_error = e
                    logger.error(
                        f"Unexpected error during Gemini call or JSON validation: {e}"
                    )
                    break  # Try next model

        logger.error(
            f"All configured models ({models_to_try}) failed. Last error: {last_error}"
        )
        raise RuntimeError(
            f"Gemini processing failure after trying all models: {str(last_error)}"
        ) from last_error

    async def extract_raster_figure(
        self, image: Image.Image
    ) -> Tuple[FigureExtraction, Dict[str, int]]:
        """
        Extracts structured text from a cropped, isolated raster figure image.
        Uses asyncio.to_thread to prevent blocking the async event loop during HTTP requests.
        """
        return await asyncio.to_thread(
            self._call_gemini_sync,
            image=image,
            prompt=RASTER_FIGURE_PROMPT,
            response_schema=FigureExtraction,
        )

    async def extract_vector_page_figures(
        self, page_image: Image.Image
    ) -> Tuple[VectorPageExtraction, Dict[str, int]]:
        """
        Identifies and extracts figures on a full-page rendered image (for pages
        with vector shapes but no rasters).
        Uses asyncio.to_thread to prevent blocking the async event loop during HTTP requests.
        """
        return await asyncio.to_thread(
            self._call_gemini_sync,
            image=page_image,
            prompt=VECTOR_PAGE_PROMPT,
            response_schema=VectorPageExtraction,
        )


# Instantiate a single global Gemini client wrapper
gemini_client = GeminiClient()