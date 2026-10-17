from typing import Literal


def route_after_evaluate_documents(state) -> Literal["RELEVANT", "NOT_RELEVANT"]:

    print("\n========== CRAG ROUTING ==========")
    verdict = state["retrieval_verdict"]
    print(f"Retrieval verdict       : {verdict}")

    if verdict == "CORRECT":
        return "RELEVANT"
    else:
        return "NOT_RELEVANT"
