import io
import logging
from typing import Dict, Any, List, Tuple
from PIL import Image
import fitz  # PyMuPDF
from app.config import settings
from app.utils.helpers import compute_image_hash

logger = logging.getLogger("papervision")

class RasterExtractor:
    """
    Handles extraction of embedded raster images from PDF pages and filters them 
    using configurable structural and position heuristics to remove noise.
    """
    
    @staticmethod
    def compute_document_hash_frequencies(doc: fitz.Document) -> Dict[str, int]:
        """
        Scans all pages in the PDF document first to build a frequency map of image hashes.
        This enables robust document-wide deduplication of publisher logos and header icons.
        """
        hash_frequencies: Dict[str, int] = {}
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                images = page.get_images(full=True)
                for img in images:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    if base_image:
                        img_hash = compute_image_hash(base_image["image"])
                        hash_frequencies[img_hash] = hash_frequencies.get(img_hash, 0) + 1
            except Exception as e:
                logger.warning(f"Error reading images on page {page_num + 1} during hashing: {e}")
        
        logger.info(f"Scanned PDF. Discovered {len(hash_frequencies)} unique image hashes.")
        return hash_frequencies

    @staticmethod
    def extract_meaningful_figures(
        page: fitz.Page, 
        doc: fitz.Document, 
        hash_frequencies: Dict[str, int]
    ) -> List[Tuple[Image.Image, int]]:
        """
        Extracts all raster images from a page that pass our heuristic filters.
        Returns a list of tuples containing: (PIL.Image, xref_id)
        """
        page_num = page.number + 1
        page_height = page.rect.height
        page_width = page.rect.width
        
        # Calculate exclusion bounds based on page height percentages
        top_exclusion = page_height * (settings.TOP_MARGIN_EXCLUSION_PCT / 100.0)
        bottom_exclusion = page_height * (1.0 - settings.BOTTOM_MARGIN_EXCLUSION_PCT / 100.0)
        
        meaningful_figures: List[Tuple[Image.Image, int]] = []
        
        try:
            image_list = page.get_images(full=True)
        except Exception as e:
            logger.error(f"Error fetching images for page {page_num}: {e}")
            return []
            
        logger.info(f"Page {page_num}: Found {len(image_list)} raw embedded images.")
        
        for idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                # 1. Extract base image metadata and raw bytes
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue
                    
                image_bytes = base_image["image"]
                img_ext = base_image["ext"]
                native_w = base_image["width"]
                native_h = base_image["height"]
                native_area = native_w * native_h
                
                # 2. Check: Size Threshold Heuristics
                if (native_w < settings.MIN_WIDTH or 
                    native_h < settings.MIN_HEIGHT or 
                    native_area < settings.MIN_AREA):
                    logger.debug(
                        f"Page {page_num} img [xref={xref}]: Filtered out (too small: {native_w}x{native_h}, area={native_area})"
                    )
                    continue
                    
                # 3. Check: Aspect Ratio Heuristic (Filters banners, dividers, page-lines)
                aspect_ratio = max(native_w / native_h, native_h / native_w)
                if aspect_ratio > settings.MAX_ASPECT_RATIO:
                    logger.debug(
                        f"Page {page_num} img [xref={xref}]: Filtered out (extreme aspect ratio: {aspect_ratio:.2f})"
                    )
                    continue
                    
                # 4. Check: Document-wide Deduplication Heuristic (Filters repeated publisher logos/icons)
                img_hash = compute_image_hash(image_bytes)
                occurrences = hash_frequencies.get(img_hash, 0)
                if occurrences > settings.MAX_HASH_OCCURRENCES:
                    logger.debug(
                        f"Page {page_num} img [xref={xref}]: Filtered out (repeated logo/watermark hash, occurrences={occurrences})"
                    )
                    continue
                    
                # 5. Check: Page Location & Bounding Box Heuristics
                rects = page.get_image_rects(xref)
                if rects:
                    # If an image appears multiple times, check if all placements are in margin exclusion zones
                    all_in_margins = True
                    for rect in rects:
                        # PyMuPDF coords: y0 is top, y1 is bottom
                        is_in_top_margin = rect.y1 <= top_exclusion
                        is_in_bottom_margin = rect.y0 >= bottom_exclusion
                        
                        if not (is_in_top_margin or is_in_bottom_margin):
                            all_in_margins = False
                            break
                            
                    if all_in_margins:
                        logger.debug(
                            f"Page {page_num} img [xref={xref}]: Filtered out (located entirely in page margins/exclusion zone)"
                        )
                        continue
                
                # If it passed all filters, load it as a PIL image
                pil_image = Image.open(io.BytesIO(image_bytes))
                meaningful_figures.append((pil_image, xref))
                logger.info(
                    f"Page {page_num} img [xref={xref}]: Kept (size={native_w}x{native_h}, aspect_ratio={aspect_ratio:.2f})"
                )
                
            except Exception as e:
                logger.error(f"Error processing image xref {xref} on page {page_num}: {e}", exc_info=True)
                
        return meaningful_figures
