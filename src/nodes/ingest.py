import re
from src.memory import AgentMemorySystem as AMS, AMSState

def parse_github_diff(diff_text: str) -> tuple[str, list[str]]:
    """
    Parses a raw git diff and extracts changed lines per file, keeping the surrounding
    code as labelled context. The diff is produced with --function-context, so each hunk
    carries its enclosing function: an auditor can tell a duplicate/relocated block from a
    real removal instead of judging a bare [REMOVED] in isolation. Context lines cost tokens
    but the diff is the cached part of the audit, so the extra is paid once and reused.
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

        if not current_file:
            continue

        # Capture added lines (starting with +, but not the +++ header)
        if line.startswith('+') and not line.startswith('+++'):
            parsed_output.append(f"[ADDED]: {line[1:]}")
        # Capture removed lines (starting with -, but not the --- header)
        elif line.startswith('-') and not line.startswith('---'):
            parsed_output.append(f"[REMOVED]: {line[1:]}")
        # Keep unchanged context lines (a leading space) so a change is read in situ, not blind.
        # Skip hunk headers (@@ ...) and the no-newline marker; they aren't source.
        elif line.startswith(' '):
            parsed_output.append(f"[CONTEXT]: {line[1:]}")

    return "\n".join(parsed_output), files_changed

def ingest_pr_node(state: AMSState):
    """Ingests the raw PR diff from the user and parses it"""
    ams = AMS(state)
    # We assume the very first human message contains our raw PR diff
    raw_diff = ams.read("messages",[""])[-1]
    parsed_diff, files_changed = parse_github_diff(raw_diff)
    return {"audit": {
        "messages":[f"System: Ingested PR data. Extracted changes:\n{parsed_diff}"],
        "parsed_diff": parsed_diff,
        "files_changed": files_changed,
    }}
