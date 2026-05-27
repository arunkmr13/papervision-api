import os
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from app.config import settings
from app.api.router import router as api_router
from app.workers.task_queue import background_worker

# 1. Setup global logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("papervision")

# 2. Define Lifespan Context Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application lifecycle events:
    - Pre-creates necessary storage directories.
    - Launches the asynchronous background worker loop task.
    - Shuts down background tasks cleanly during server termination.
    """
    logger.info("Initializing PaperVision Service...")
    
    # Ensure temporary upload directory is present
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # Spawn background processing daemon
    logger.info("Starting background figure extraction daemon task...")
    worker_task = asyncio.create_task(background_worker.worker_loop())
    
    yield  # Handover execution control back to FastAPI
    
    # Tear down tasks on shutdown
    logger.info("Stopping background figure extraction daemon task...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        logger.info("Background worker daemon stopped successfully.")
    except Exception as e:
        logger.error(f"Error occurred while stopping background daemon: {e}")
        
    logger.info("PaperVision Service shutdown complete.")

# 3. Instantiate FastAPI Application
app = FastAPI(
    title="PaperVision API",
    description=(
        "Production-ready FastAPI service that extracts structured information "
        "and LaTeX text from meaningful figures in research paper PDFs using "
        "PyMuPDF and Google's Gemini Vision API."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# 4. Register REST Endpoints Router
app.include_router(api_router)

# 5. Define root utility redirect endpoint
@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirects the base URL to standard Swagger API docs."""
    return RedirectResponse(url="/docs")
