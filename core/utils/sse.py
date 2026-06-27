import json


def format_sse_event(event_type: str, payload: dict) -> str:
    event_body = {"type": event_type, **payload}
    return f"data: {json.dumps(event_body)}\n\n"
