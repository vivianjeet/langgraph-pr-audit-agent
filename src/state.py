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

class RuleStatus(str, Enum):
    SEEDED = "seeded"                      # baseline rule, authored not learned
    LEARNED_PENDING = "learned_pending"    # proposed by an audit, awaiting review
    LEARNED_APPROVED = "learned_approved"  # promoted after human approval
    REJECTED = "rejected"                  # human rejected a pending rule; kept (not re-learned)
    RETIRED = "retired"                    # deactivated an active rule; kept (un-learn-safe)

class RuleCategory(str, Enum):
    SECURITY = "security"
    QUALITY = "quality"
    COVERAGE = "coverage"

class _FindingBase(BaseModel):
    severity: Severity
    title: str = Field(description="A short one-line summary of the issue (max ~8 words)")
    
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

    # Compliance (MCP): regulatory passages this diff touched + their cited source spans.
    # Per-run audit data (NOT cross-run memory), so they live here, plain-replace.
    compliance_context: list      # [{text, source, framework, similarity}] from 
                                  # search_compliance_docs
    compliance_citations: list    # [{claim, citations:[{source, cited_text}]}]

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

    # Per-run LLM accounting the security node stashes so it rides the audit channel out to
    # the integration table (the router computes it on LLMResult, which is otherwise discarded).
    # Only the security node writes them; last-writer-wins under merge_audit, no clobber.
    llm_cost_usd: float    # the security call's cost in USD
    llm_tier: str          # "powerful" (thinking/cache regulated path) or "flash" (plain)
    llm_thinking: bool     # True only on the extended-thinking branch

    # Routing and Safety Flags
    human_decision: str
    iteration_count: int

    # Audit Plan
    audit_plan: dict
    confidence_score: float # reflection's own confidence the audit is complete
    gaps_identified: list[str] # what the first pass likely missed

    final_report: str # markdown report produced by finalize

    node_errors: Annotated[list[str], operator.add] # any errors nodes want to report but not raise (eg; audit degraded due to LLM issues, but we still want a report)

# NOTE: AuditState above is the IN-CONTEXT working-memory schema (one substate). The
# graph's full nested state - AMSState - and its `merge_audit` reducer live in
# src/memory.py, because they are memory-system concerns: AMS owns the four-channel
# state. This module stays the domain/schema leaf (findings, plan, AuditState) and
# imports nothing of ours, so memory.py can import AuditState from here without a cycle.

# --- Signal-message prefixes: the canonical "this line carries decision/finding
# signal" vocabulary, defined ONCE so reflexion (critique input) and compression
# (no-LLM fallback keep-list) can't drift. Grouped by phase; consumers slice.
# Every emit site uses startswith on these exact prefixes (see nodes/ and graph.py).
PLAN_PREFIX = "System: Audit plan"
AUDIT_RESULT_PREFIXES = (
    "System: Security checks complete",
    "System: Quality checks complete",
    "System: Test audit completed",
)
SYNTHESIS_PREFIX = "System: Synthesized report"
DECISION_PREFIXES = (                       # only exist post-synthesis
    "System: Human",
    "System: Final report",
)

# Reflexion critiques the audit INPUTS (plan + results + synthesis); decisions
# don't exist yet when it runs, so they're intentionally excluded.
REFLEXION_SIGNAL_PREFIXES = (PLAN_PREFIX, *AUDIT_RESULT_PREFIXES, SYNTHESIS_PREFIX)

# Compression's no-LLM fallback keeps EVERYTHING decision-bearing across a whole
# session = reflexion's set PLUS the post-synthesis decisions.
COMPRESSION_SIGNAL_PREFIXES = (*REFLEXION_SIGNAL_PREFIXES, *DECISION_PREFIXES)