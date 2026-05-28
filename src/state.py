from enum import Enum
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List
import operator

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    NONE = "none"

class SecurityFinding(BaseModel):
    file_path: str = Field(description = "The file where the issue was found")
    line_number: int = Field(description = "The specific line number")
    description: str = Field(description = "Detailed explaination of the vulnerability")
    cwe_id: str = Field(description = "The Common Weakness Enumeration ID (eg. CWE-89)")
    severity: Severity

class AuditState(TypedDict):
    # LangGraph requires a messages list, operator.add appends instead of overwriting
    messages: Annotated[list, operator.add]

    # The list of structured findings the agent has disscovered
    security_findings: Annotated[List[SecurityFinding],operator.add]

    # Routing and Safety Flags
    human_decision: str | None
    iteration_count: int