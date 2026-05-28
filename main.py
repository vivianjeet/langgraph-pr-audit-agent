
import argparse
import uuid
from src.graph import app

def run_audit(diff_text: str):
    """Runs the end to end graph execution on the provided diff"""
    print("Starting LangGraph PR Audit Agent... \n")

    initial_state = {
        "messages" : [diff_text]
    }
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("Running graph...\n")
    for event in app.stream(initial_state,config = config):
        for node_name, node_state in event.items():
            print(f"--- Node Executed: {node_name} ---")
            if "messages" in node_state and node_state["messages"]:
                print(node_state["messages"][-1])
            print()
        

def main():
    parser = argparse.ArgumentParser(description="LangGraph PR Audit Agent")
    parser.add_argument("--test", action="store_true", help="Run the end-to-end smoke test")
    args = parser.parse_args()

    if args.test:
        # import smoke tests function and run it
        from tests.smoke_test import run_smoke_test
        run_smoke_test()
    else:
        print("LangGraph PR Audit Agent is ready.")
        print("Run 'python main.py --test' to execute the smoke test.")

if __name__ == "__main__":
    main()