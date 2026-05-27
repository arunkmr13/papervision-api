import os
import asyncio
import logging
from typing import Tuple
from app.storage.job_store import job_store, JobStatus

logger = logging.getLogger("papervision")

class BackgroundWorker:
    """
    Asynchronous background job queue and execution worker.
    Maintains a FIFO task queue using asyncio.Queue and runs a worker daemon
    to process PDF figure extraction jobs without blocking API requests.
    """
    def __init__(self):
        self.queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()
        self._worker_task: asyncio.Task = None

    async def submit_job(self, job_id: str, file_path: str) -> None:
        """Pushes a new figure extraction job onto the background processing queue."""
        logger.info(f"Enqueuing job {job_id} for background processing...")
        await self.queue.put((job_id, file_path))

    async def worker_loop(self) -> None:
        """
        Long-running background task that polls the queue for jobs,
        updates progress status in storage, runs the PDF extraction service,
        and manages file cleanups.
        """
        logger.info("Background worker loop started.")
        # Local imports to avoid circular dependency loops during initialization
        from app.services.pdf_service import pdf_service
        
        while True:
            try:
                job_id, file_path = await self.queue.get()
            except asyncio.CancelledError:
                logger.info("Background worker task cancelled.")
                break
                
            logger.info(f"Worker dequeued job {job_id}. Starting execution...")
            
            try:
                # 1. Transition state from queued to processing
                await job_store.update_status(job_id, JobStatus.PROCESSING)
                
                # 2. Run core orchestrator service
                figures = await pdf_service.process_pdf(job_id, file_path)
                
                # 3. Transition state to completed and store structured results
                await job_store.update_result(job_id, JobStatus.COMPLETED, figures)
                logger.info(f"Successfully processed and completed job {job_id}.")
                
            except Exception as e:
                logger.error(f"Failed to process job {job_id}: {e}", exc_info=True)
                # Transition state to failed and store the error message
                await job_store.update_error(job_id, str(e))
                
            finally:
                # Signal task completion to the queue
                self.queue.task_done()
                
                # 4. Clean up temporary uploaded PDF to prevent disk space leaks
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Cleaned up temporary file: {file_path}")
                except Exception as cleanup_err:
                    logger.warning(f"Could not delete temporary file {file_path}: {cleanup_err}")

# Instantiate global background worker
background_worker = BackgroundWorker()
