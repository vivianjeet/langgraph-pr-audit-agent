from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import TypedDict, Annotated
import operator

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    NONE = "none"

class AuditDepth(str, Enum):
    SHALLOW = "shallow"
    STANDARD = "standard"
    DEEP = "deep"

class _FindingBase(BaseModel):
    severity: Severity

    @field_validator("severity", mode="before")
    @classmethod
    def _coerce_severity(cls, v):
        # Gemini sends the enum as a string ("critical");
        # normalise -> so strict validation downstream sees a real enum instance.
        if isinstance(v,str):
            return Severity(v.lower())
        return v

class SecurityFinding(_FindingBase):
    file_path:str = Field(description = "The file where the issue was found")
    line_number:int = Field(description = "The specific line number")
    description:str = Field(description = "Detailed explaination of the vulnerability")
    cwe_id:str = Field(description = "The Common Weakness Enumeration ID (eg. CWE-89)")

class QualityFinding(_FindingBase):
    file_path:str = Field(description = "The file where the issue was found")
    line_number:int = Field(description = "The specific line number")
    description:str = Field(description = "Detailed explaination of the code quality issue")

class CoverageFinding(_FindingBase):
    file_path:str = Field(description = "The file where the issue was found")
    line_number:int = Field(description = "The specific line number")
    description:str = Field(description = "Detailed explaination of the missing test case or issue with existing test")

class AuditPlan(BaseModel):
    focus_areas: list[str] = Field(description="Top security/quality themes to investigate")
    files_to_prioritize: list[str] = Field(description="Files most likely to contain risk")
    audit_depth: AuditDepth = Field(description="One of: shallow, standard, deep")
    risk_level: Severity = Field(description="Overall a-priori risk of this PR")

    @field_validator("risk_level", mode="before")
    @classmethod
    def _coerce_risk(cls, v):
        if isinstance(v, str):
            return Severity(v.lower())
        return v
    
    @field_validator("audit_depth", mode="before")
    @classmethod
    def _coerce_depth(cls, v):
        if isinstance(v, str):
            return AuditDepth(v.lower())
        return v

class AuditState(TypedDict):
    # LangGraph requires a messages list, operator.add appends instead of overwriting
    messages: Annotated[list, operator.add]
    parsed_diff: str # ingest writes, all audit nodes read

    # The list of structured findings the agent has disscovered
    security_findings: list[SecurityFinding]
    quality_findings: list[QualityFinding]
    test_findings: list[CoverageFinding]

    # Populated by ingest; used by routing (eg; "auth file changed but no findings")
    files_changed: list[str]

    # Computed determenistically by synthesize; drives routing. 1.0 = clean, lower
    # means riskier
    security_score: float
    quality_score: float
    test_score: float

    # Routing and Safety Flags
    human_decision: str
    iteration_count: int

    # Audit Plan
    audit_plan: dict
    confidence_score: float # reflection's own confidence the audit is complete
    gaps_identified: list[str] # what the first pass likely missed