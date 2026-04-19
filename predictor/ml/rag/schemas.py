from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RAGChunk:
    chunk_id: str
    source_file: str
    project_title: str
    chunk_type: str
    text: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    chunk: RAGChunk
    score: float
    source: str  # "vector" ou "bm25"