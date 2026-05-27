import logging
import fitz  # PyMuPDF

logger = logging.getLogger("papervision")

class VectorExtractor:
    """
    Detects the presence of vector graphics (rendered from drawing commands) on PDF pages.
    Vector drawing commands are typically used for rendering charts, flowcharts, 
    schematics, and mathematical equation layouts.
    """
    
    @staticmethod
    def has_vector_drawings(page: fitz.Page) -> bool:
        """
        Determines if the specified page contains vector drawing commands.
        Uses page.get_drawings() which returns a list of path objects.
        """
        try:
            drawings = page.get_drawings()
            drawings_count = len(drawings)
            
            if drawings_count > 0:
                logger.info(
                    f"Page {page.number + 1}: Detected {drawings_count} vector drawing paths/commands."
                )
                return True
                
            logger.debug(f"Page {page.number + 1}: No vector drawing commands found.")
            return False
            
        except Exception as e:
            logger.error(
                f"Failed to check vector drawing commands on page {page.number + 1}: {e}", 
                exc_info=True
            )
            return False
