import os
import uuid
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, status, Response
from app.config import settings
from app.storage.job_store import job_store, JobStatus
from app.workers.task_queue import background_worker
from app.models.schemas import JobSubmitResponse, JobResultResponse, FigureResult

logger = logging.getLogger("papervision")

router = APIRouter()

@router.post(
    "/extract", 
    response_model=JobSubmitResponse, 
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a research paper PDF for figure extraction"
)
async def extract_figures(file: UploadFile = File(...)) -> JobSubmitResponse:
    """
    Accepts a research paper PDF via multipart upload, saves it to storage, 
    and enqueues a background job for figure classification and extraction.
    Immediately returns the generated Job ID and queued status.
    """
    # 1. Validation: Ensure the uploaded file is a PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a valid PDF document."
        )
        
    # 2. Ensure storage upload directory exists
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    # 3. Create a unique Job ID and local file path
    job_id = str(uuid.uuid4())
    temp_filename = f"{job_id}.pdf"
    file_path = os.path.join(settings.UPLOAD_DIR, temp_filename)
    
    # 4. Save uploaded file content to disk asynchronously
    try:
        with open(file_path, "wb") as buffer:
            # Chunked writing prevents memory spikes for large uploads
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
        logger.info(f"Saved uploaded PDF file to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to write uploaded file to disk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save upload: {str(e)}"
        )
        
    # 5. Initialize Job State record in our store
    await job_store.create_job(job_id)
    
    # 6. Push job onto background worker queue
    await background_worker.submit_job(job_id, file_path)
    
    return JobSubmitResponse(job_id=job_id, status=JobStatus.QUEUED)


@router.get(
    "/status/{job_id}", 
    summary="Retrieve current job processing status"
)
async def get_job_status(job_id: str):
    """
    Queries the current processing status of a job.
    Returns: queued | processing | completed | failed
    """
    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found."
        )
        
    return {
        "job_id": job_id,
        "status": job["status"]
    }


@router.get(
    "/result/{job_id}", 
    response_model=JobResultResponse, 
    summary="Get structured figure extraction results for a completed job"
)
async def get_job_result(job_id: str) -> JobResultResponse:
    """
    Returns the complete list of structured figure extractions and LLM token usage 
    if the job completed successfully. Includes error details if it failed.
    """
    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found."
        )
        
    # If the job is still queued or processing, return status and empty lists
    if job["status"] in [JobStatus.QUEUED, JobStatus.PROCESSING]:
        return JobResultResponse(
            job_id=job_id,
            status=job["status"],
            figures=[]
        )
        
    # If the job failed, return status and the exception details
    if job["status"] == JobStatus.FAILED:
        return JobResultResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
            figures=[],
            error=job.get("error", "An unknown error occurred during pipeline execution.")
        )
        
    # If completed, map and return the list of figures
    figures_list: List[FigureResult] = []
    for fig in job["figures"]:
        figures_list.append(FigureResult(**fig))
        
    return JobResultResponse(
        job_id=job_id,
        status=JobStatus.COMPLETED,
        figures=figures_list
    )


@router.get(
    "/download/{job_id}",
    summary="Download figure extraction results as a formatted Markdown report"
)
async def download_job_markdown(job_id: str):
    """
    Generates a beautifully formatted Markdown report containing all extracted figures,
    formatted Markdown tables, LaTeX equations, and structured descriptions,
    delivering it as a downloadable file attachment.
    """
    job = await job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found."
        )
        
    if job["status"] in [JobStatus.QUEUED, JobStatus.PROCESSING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {job_id} is currently '{job['status']}'. Please wait until it completes."
        )
        
    if job["status"] == JobStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Job failed: {job.get('error', 'Unknown pipeline failure')}"
        )
        
    # Generate Markdown Report Content
    md = []
    md.append(f"# PaperVision Figure Extraction Report")
    md.append(f"**Job ID**: `{job_id}`  ")
    md.append(f"**Status**: `Completed`  ")
    md.append(f"**Total Figures Extracted**: {len(job['figures'])}  ")
    md.append("\n---\n")
    
    for fig in job["figures"]:
        page = fig["page"]
        fig_idx = fig["figure_index"]
        f_type = fig["figure_type"]
        method = fig["extraction_method"]
        text_data = fig["structured_text"]
        
        md.append(f"## Figure #{fig_idx} - Page {page} ({f_type.upper()})")
        md.append(f"* **Extraction Method**: `{method}`")
        md.append(f"* **Model Confidence**: `{text_data.get('confidence', 'N/A')}`")
        md.append(f"* **Reasoning**: {text_data.get('reasoning', 'N/A')}")
        md.append("")
        
        # 1. Format specific content types
        if f_type in ["chart", "graph"]:
            md.append("### Chart Metadata:")
            md.append(f"- **Title**: {text_data.get('title', 'None')}")
            md.append(f"- **X-Axis**: {text_data.get('x_axis', 'None')}")
            md.append(f"- **Y-Axis**: {text_data.get('y_axis', 'None')}")
            legends = text_data.get("legends")
            if legends:
                md.append(f"- **Legends**: {', '.join(legends)}")
            annotations = text_data.get("visible_annotations")
            if annotations:
                md.append(f"- **Visible Annotations**:")
                for ann in annotations:
                    md.append(f"  - {ann}")
                    
        elif f_type == "table_image":
            md.append("### Table Data:")
            headers = text_data.get("headers")
            rows = text_data.get("rows")
            
            # Format markdown table
            if rows:
                col_count = len(rows[0]) if rows else 0
                if headers:
                    col_count = max(col_count, len(headers))
                    header_line = "| " + " | ".join(headers) + " |"
                else:
                    header_line = "| " + " | ".join([f"Col {i+1}" for i in range(col_count)]) + " |"
                    
                separator_line = "| " + " | ".join(["---" for _ in range(col_count)]) + " |"
                md.append(header_line)
                md.append(separator_line)
                
                for r in rows:
                    padded_row = r + [""] * (col_count - len(r))
                    md.append("| " + " | ".join([str(cell) for cell in padded_row]) + " |")
            else:
                md.append("*No tabular rows extracted.*")
                
        elif f_type in ["flowchart", "scientific_diagram"]:
            md.append("### Diagram Structure:")
            nodes = text_data.get("node_labels")
            if nodes:
                md.append(f"- **Nodes**: {', '.join(nodes)}")
            edges = text_data.get("edge_labels")
            if edges:
                md.append(f"- **Edge Labels**: {', '.join(edges)}")
            rels = text_data.get("relationships")
            if rels:
                md.append("- **Relationships**:")
                for r in rels:
                    md.append(f"  - {r}")
                    
        elif f_type == "equation_image":
            md.append("### Extracted Equation (LaTeX):")
            latex = text_data.get("latex")
            if latex:
                md.append(f"$$\n{latex}\n$$")
            else:
                md.append("*No LaTeX formula string parsed.*")
                
        # Append general description if available
        desc = text_data.get("general_description")
        if desc:
            md.append("### Description:")
            md.append(desc)
            
        md.append("\n---\n")
        
    markdown_content = "\n".join(md)
    
    # Return as downloadable text/markdown file
    filename = f"papervision_report_{job_id}.md"
    return Response(
        content=markdown_content,
        media_type="text/markdown",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )
