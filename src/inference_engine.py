def infer_decision(question):
    question = question.lower()
    if "discount" in question:
        return "This qualifies for a standard 10% discount if criteria met."
    elif "decline" in question or "risk" in question:
        return "This might need to be declined or referred."
    elif "refer" in question or "unclear" in question:
        return "Referring this case internally for further underwriting."
    elif "accept" in question or "coverage" in question:
        return "Coverage is within standard appetite. Proceeding to accept."
    else:
        return "Need more details. Referring this to underwriter."
