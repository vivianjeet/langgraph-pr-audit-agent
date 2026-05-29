from main import run_audit
from pathlib import Path

def run_smoke_test():
    """ Executes a smoke test with mock SQL - injection PR diff"""
    print("=============================================\n")
    print("   Initiating Smoke test   \n")
    print("=============================================\n")
    current_dir = Path(__file__)
    file_path = current_dir.parent / "files/smoke_test_sample_diff.txt"

    sample_diff = open(file_path,"r").read()

    run_audit(sample_diff)

    print("=============================================\n")
    print("   Smoke test complete   \n")
    print("=============================================\n")