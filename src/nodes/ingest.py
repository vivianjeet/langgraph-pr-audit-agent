import re
from src.state import AuditState

def parse_github_diff(diff_text: str) -> str:
    """
    Parses a raw git diff and extracts added/modified lines per file.
    Reduces token usage by stripping out unchanged context lines
    """
    parsed_output: list[str] = []
    files_changed: list[str] = []
    current_file = None

    # Regex to match the start of a new file diff
    file_pattern = re.compile(r"^diff --git a/(.*?) b/(.*)$")

    for line in diff_text.split("\n"):
        file_match = file_pattern.match(line)

        if file_match:
            current_file = file_match.group(2)
            files_changed.append(current_file)
            parsed_output.append(f"\n[FILE MODIFIED]: {current_file}")
            continue

        # Capture added lines (starting with +, but not the +++ header)
        if line.startswith('+') and not line.startswith('+++') and current_file:
            parsed_output.append(f"[ADDED]: {line[1:]}")
        # Capture removed lines
        if line.startswith('-') and not line.startswith('---') and current_file:
            parsed_output.append(f"[REMOVED]: {line[1:]}")

    return "\n".join(parsed_output), files_changed

def ingest_pr_node(state: AuditState):
    """Ingests the raw PR diff from the user and parses it"""
    # We assume the very first human message contains our raw PR diff
    raw_diff = state.get("messages",[""])[-1]
    parsed_diff, files_changed = parse_github_diff(raw_diff)
    return {
        "messages":[f"System: Ingested PR data. Extracted changes:\n{parsed_diff}"],
        "parsed_diff": parsed_diff,
        "files_changed": files_changed,
    }
