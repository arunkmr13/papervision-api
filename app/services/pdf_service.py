import io
import logging
from typing import List, Dict, Any
from PIL import Image
import fitz  # PyMuPDF
from app.extractors.raster import RasterExtractor
from app.extractors.vector import VectorExtractor
from app.llm.gemini_client import gemini_client
from app.utils.helpers import optimize_image
from app.models.schemas import FigureResult, TokenUsage, FigureExtraction

logger = logging.getLogger("papervision")

class PDFService:
    """
    Orchestration service that executes the PaperVision core pipeline on a uploaded PDF.
    Implements a hybrid approach, extracting and processing raster images, or falling back
    to full-page vector renderings only when drawings are present.
    """
    
    async def process_pdf(self, job_id: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Processes a PDF document page-by-page.
        Extracts structured figure text using Gemini Vision API and aggregates the results.
        """
        logger.info(f"Processing PDF document: {file_path} for job: {job_id}")
        
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF file {file_path}: {e}")
            raise RuntimeError(f"Could not open uploaded PDF: {str(e)}") from e
            
        total_pages = len(doc)
        logger.info(f"PDF opened successfully. Total pages to scan: {total_pages}")
        
        # STEP 1 & 2: Pre-compute document-wide image hashes for deduplication
        hash_frequencies = RasterExtractor.compute_document_hash_frequencies(doc)
        
        aggregated_results: List[Dict[str, Any]] = []
        global_figure_idx = 1
        
        for page_idx in range(total_pages):
            page_num = page_idx + 1
            page = doc[page_idx]
            logger.info(f"--- Processing Page {page_num}/{total_pages} ---")
            
            # STEP 3: Attempt to extract isolated, meaningful raster images
            raster_figures = RasterExtractor.extract_meaningful_figures(page, doc, hash_frequencies)
            
            if raster_figures:
                logger.info(f"Page {page_num}: Found {len(raster_figures)} meaningful raster figures. Processing each independently.")
                
                for idx, (pil_img, xref) in enumerate(raster_figures):
                    try:
                        # Image Optimization: Resize longest side to max 1024px while keeping aspect ratio
                        optimized_img = optimize_image(pil_img, max_side=1024)
                        
                        # Process image using Gemini Vision API
                        extraction, tokens = await gemini_client.extract_raster_figure(optimized_img)
                        
                        # Apply logo / branding validation filter
                        if self._is_logo_or_branding(extraction):
                            logger.info(
                                f"Page {page_num}: Ignored raster figure (xref={xref}) "
                                f"as it was identified as a logo, branding, or metadata noise."
                            )
                            continue
                        
                        # Map to output structure
                        structured_data = self._map_extraction_data(extraction)
                        
                        figure_result = {
                            "page": page_num,
                            "figure_index": global_figure_idx,
                            "figure_type": extraction.figure_type,
                            "extraction_method": "raster",
                            "structured_text": structured_data,
                            "token_usage": tokens
                        }
                        
                        aggregated_results.append(figure_result)
                        global_figure_idx += 1
                        
                    except Exception as fig_err:
                        logger.error(
                            f"Error extracting raster figure index {idx} (xref={xref}) on page {page_num}: {fig_err}",
                            exc_info=True
                        )
                        # We continue processing other figures rather than crashing the whole job
                        continue
                        
            # STEP 4: Fallback to vector drawings if no raster images were found
            else:
                logger.info(f"Page {page_num}: No meaningful raster figures found. Checking for vector commands fallback.")
                
                # Verify if vector drawing commands exist
                if VectorExtractor.has_vector_drawings(page):
                    logger.info(f"Page {page_num}: Vector drawings detected. Rendering full page for Gemini Vision.")
                    
                    try:
                        # Render full page as high resolution image (150 DPI is crisp and readable)
                        pix = page.get_pixmap(dpi=150)
                        page_png_bytes = pix.tobytes("png")
                        page_pil_image = Image.open(io.BytesIO(page_png_bytes))
                        
                        # Optimize full-page image (resize if it exceeds 1024px, preserving ratio)
                        optimized_page_img = optimize_image(page_pil_image, max_side=1024)
                        
                        # Send optimized full page to Gemini
                        vector_extraction, tokens = await gemini_client.extract_vector_page_figures(optimized_page_img)
                        
                        if vector_extraction and vector_extraction.figures:
                            logger.info(
                                f"Page {page_num}: Gemini extracted {len(vector_extraction.figures)} figures from vector fallback."
                            )
                            
                            for extracted_fig in vector_extraction.figures:
                                # Apply logo / branding validation filter
                                if self._is_logo_or_branding(extracted_fig):
                                    logger.info(
                                        f"Page {page_num}: Ignored vector fallback figure "
                                        f"as it was identified as a logo, branding, or metadata noise."
                                    )
                                    continue
                                    
                                structured_data = self._map_extraction_data(extracted_fig)
                                
                                figure_result = {
                                    "page": page_num,
                                    "figure_index": global_figure_idx,
                                    "figure_type": extracted_fig.figure_type,
                                    "extraction_method": "vector",
                                    "structured_text": structured_data,
                                    # Since multiple figures are extracted in a single page API call, 
                                    # we assign the full call tokens to each figure or divide them. 
                                    # In our design, we log the usage metadata as reported.
                                    "token_usage": tokens
                                }
                                aggregated_results.append(figure_result)
                                global_figure_idx += 1
                        else:
                            logger.info(f"Page {page_num}: No figures isolated by Gemini on vector fallback page.")
                            
                    except Exception as vec_err:
                        logger.error(
                            f"Error running vector fallback rendering on page {page_num}: {vec_err}", 
                            exc_info=True
                        )
                        continue
                else:
                    logger.info(f"Page {page_num}: Skipped. No raster images and no vector drawing commands.")
                    
        doc.close()
        logger.info(f"PDF extraction completed for job {job_id}. Found {len(aggregated_results)} total figures.")
        return aggregated_results

    def _is_logo_or_branding(self, extraction: FigureExtraction) -> bool:
        """
        Heuristic post-processing filter that validates Gemini extractions.
        If a figure is classified as 'unknown' and the confidence is low, or if keywords
        like 'logo', 'branding', 'watermark', 'editorial' appear in reasoning or description,
        or if the confidence score is strictly 0.0, we mark it as logo noise to be ignored.
        """
        # Strictly ignore if model returned confidence 0.0 (marked explicitly as logo noise)
        if extraction.confidence <= 0.0:
            return True
            
        f_type = extraction.figure_type.lower()
        reasoning = (extraction.reasoning or "").lower()
        description = (extraction.general_description or "").lower()
        
        # If the type is unknown, check for logo indicators in description or reasoning
        if f_type == "unknown":
            branding_keywords = [
                "logo", "branding", "watermark", "publisher", "manuscript", 
                "journal title", "editorial text", "submission site", "copyright"
            ]
            for keyword in branding_keywords:
                if keyword in reasoning or keyword in description:
                    return True
            # Also ignore generic low-confidence unknown figures
            if extraction.confidence < 0.5:
                return True
                
        return False

    def _map_extraction_data(self, extraction: FigureExtraction) -> Dict[str, Any]:
        """
        Pulls out the specific non-null sub-field from the model extraction matching its figure type.
        """
        f_type = extraction.figure_type.lower()
        
        structured_data = {
            "confidence": extraction.confidence,
            "reasoning": extraction.reasoning
        }
        
        if f_type in ["chart", "graph"] and extraction.chart_graph_data:
            structured_data.update(extraction.chart_graph_data.model_dump(exclude_none=True))
        elif f_type == "table_image" and extraction.table_data:
            structured_data.update(extraction.table_data.model_dump(exclude_none=True))
        elif f_type in ["flowchart", "scientific_diagram"] and extraction.diagram_data:
            structured_data.update(extraction.diagram_data.model_dump(exclude_none=True))
        elif f_type == "equation_image" and extraction.equation_data:
            structured_data.update(extraction.equation_data.model_dump(exclude_none=True))
        
        if extraction.general_description:
            structured_data["general_description"] = extraction.general_description
            
        return structured_data

# Instantiate single global PDF Service
pdf_service = PDFService()
