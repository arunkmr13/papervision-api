import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime

class JobStatus:
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class JobStore:
    """
    In-memory, async-safe storage repository for tracking PaperVision extraction jobs.
    This module manages job state transitions and stores the final aggregated figure metadata.
    """
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()  # Protect against concurrent dictionary mutation

    async def create_job(self, job_id: str) -> None:
        """Initializes a new job entry with 'queued' status."""
        async with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": JobStatus.QUEUED,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "figures": [],
                "error": None
            }

    async def update_status(self, job_id: str, status: str) -> None:
        """Updates the status of a job (e.g. to 'processing')."""
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    async def update_result(self, job_id: str, status: str, figures: List[Dict[str, Any]]) -> None:
        """Saves the final extracted figure results and updates the status to 'completed'."""
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = status
                self._jobs[job_id]["figures"] = figures
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    async def update_error(self, job_id: str, error: str) -> None:
        """Marks a job as 'failed' and stores the exception/error details."""
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = JobStatus.FAILED
                self._jobs[job_id]["error"] = error
                self._jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a job by its ID, or returns None if not found."""
        async with self._lock:
            return self._jobs.get(job_id)

# Instantiate a single global job store
job_store = JobStore()
