import os
import instructor
from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.state import SecurityFinding, AuditState

load_dotenv()

# initialise instructor client with Gemini (Gemini 2.5 Turbo)
# client = instructor.from_genai(genai(api_key=os.environ.get("GEMINI_API_KEY")))
genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
client = instructor.from_genai(genai_client)

class SecurityAuditOutput(BaseModel):
    reasoning: str = Field(
        description= "ReAct reasoning: Reason (what security concerns exist?) -> Act (Analyse lines) -> Observe (findings) -> Reason (false positive check?)"
    )
    findings: list[SecurityFinding] = Field(
        default_factory=list,
        description= "List of identified security vulnerabilities. Emppty if none found"
    )

def security_audit_node(state: AuditState):
    """
    Analyses the parsed PR for security vulnerabilities using the ReAct pattern.
    Validates output via instructor to enforce compliance with the SecurityFinding schema.
    """

    # Get the parsed diff from the ingest node (should be the last message)
    parsed_diff = state.get("messages",[""][-1])

    system_prompt = """
    You are a senior security engineer conducting a PR audit.
    Analyse the following code changes for security vulnerabilities, specifically focussing on:
    - OWASP Top 10
    - SQL Injection
    - PII data leaks (e.g. accidentally committing secrets or keys, or exposing personal data like Customer Records, PAN, Aadhaar etc)
    - Authentication bypass

    Code diff to analyze:
    {diff}

    Use the ReAct pattern: Reason about the code, Act by checking compliance with above security concerns, Observe the vulnerability and Reason to rule out false positives.
    """
    response = client.chat.completions.create(
        model="gemini-2.5-turbo",
        max_tokens=10000,
        messages=[
            {"role":"user","content":system_prompt.format(diff=parsed_diff)}
        ],
        response_model=SecurityAuditOutput,
    )

    #Format a system message summary
    new_message = f"System: Security checks complete.\nReasoning: {response.reasoning}\nFound {len(response.findings)} issues"

    return {
        "messages" : [new_message],
        "security_findings": response.findings
    }