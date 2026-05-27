from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# API Request/Response Schemas
class JobSubmitResponse(BaseModel):
    job_id: str
    status: str

class TokenUsage(BaseModel):
    prompt: int = Field(default=0, description="Tokens used in prompt")
    completion: int = Field(default=0, description="Tokens used in generation")
    total: int = Field(default=0, description="Total tokens used")

class FigureResult(BaseModel):
    page: int = Field(..., description="1-based page number inside the PDF")
    figure_index: int = Field(..., description="Index of the figure on that page")
    figure_type: str = Field(..., description="Classified figure type")
    extraction_method: str = Field(..., description="Method used: raster or vector")
    structured_text: Dict[str, Any] = Field(..., description="Structured information parsed from the figure")
    token_usage: TokenUsage = Field(..., description="Token usage statistics for this extraction")

class JobResultResponse(BaseModel):
    job_id: str
    status: str
    figures: List[FigureResult] = []
    error: Optional[str] = None

# Gemini Output Target Schemas
class ChartGraphData(BaseModel):
    title: Optional[str] = Field(default=None, description="Title or header of the chart/graph")
    x_axis: Optional[str] = Field(default=None, description="Label and values/scale of the horizontal X-axis")
    y_axis: Optional[str] = Field(default=None, description="Label and values/scale of the vertical Y-axis")
    legends: Optional[List[str]] = Field(default=None, description="Labels for lines, bars, or data series")
    visible_annotations: Optional[List[str]] = Field(default=None, description="Additional annotations, labels, or callouts visible in the chart")

class TableData(BaseModel):
    headers: Optional[List[str]] = Field(default=None, description="List of column headers")
    rows: Optional[List[List[str]]] = Field(default=None, description="Rows of the table, each row as a list of cell strings")

class DiagramData(BaseModel):
    node_labels: Optional[List[str]] = Field(default=None, description="Labels or text located inside nodes/boxes")
    edge_labels: Optional[List[str]] = Field(default=None, description="Text labeling connector lines or arrows")
    relationships: Optional[List[str]] = Field(default=None, description="Explanatory list of paths (e.g. 'A leads to B')")

class EquationData(BaseModel):
    latex: Optional[str] = Field(default=None, description="Standard LaTeX format representing the formula")

class FigureExtraction(BaseModel):
    figure_type: str = Field(
        ...,
        description="One of: chart, graph, table_image, flowchart, scientific_diagram, equation_image, unknown"
    )
    confidence: float = Field(..., description="Model confidence score between 0.0 and 1.0")
    reasoning: Optional[str] = Field(default=None, description="Brief rationale for the classification and extraction details")
    
    # Extractors will populate ONLY the field matching the figure_type
    chart_graph_data: Optional[ChartGraphData] = Field(default=None)
    table_data: Optional[TableData] = Field(default=None)
    diagram_data: Optional[DiagramData] = Field(default=None)
    equation_data: Optional[EquationData] = Field(default=None)
    
    general_description: Optional[str] = Field(
        default=None, 
        description="Detailed text summary or general content fallback if the type is unknown or complex"
    )

class VectorPageExtraction(BaseModel):
    """
    Schema for vector pages fallback where we send a full page to Gemini
    and it discovers multiple figure zones on the single canvas.
    """
    figures: List[FigureExtraction] = Field(default=[], description="List of figures discovered on the full page image")
