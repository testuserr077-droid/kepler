from flask import Flask, request, jsonify
from openai import OpenAI
import os, json, re
from rapidfuzz import process

app = Flask(__name__)
api_key = os.environ.get("OPENAI_API_KEY").strip()

if api_key:
    print("✅ OpenAI Key Loaded, length:", len(api_key))
else:
    print("❌ OpenAI Key MISSING")
# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Load events JSON
try:
    with open("events.json", "r") as f:
        EVENTS = json.load(f)
except Exception as e:
    print(f"Error loading events.json: {e}")
    EVENTS = []

# Build lookup maps for event_id -> event, and group_id -> events safely
EVENT_ID_MAP = {e["event_id"].lower(): e for e in EVENTS if e.get("event_id")}
GROUP_MAP = {}
for e in EVENTS:
    g = e.get("group_id")
    if g:
        GROUP_MAP.setdefault(g.lower(), []).append(e)
VALID_GROUPS = [
    "Non Weather", "snow", "announcement_group", "Visibility", "Snow/Ice", 
    "Rain", "lightning", "storm", "wind_gusts", "Dust", "tornado", "snow_pack", 
    "hail", "flooding", "air_quality", "wind", "iceing", "shake_alert", 
    "red_flag -957", "flood", "psa_group", "temperature", "tropical_storm", "snow_acc"
]

# Helper to normalize text for matching safely
def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[\s\-]+", "_", text.lower()).strip("_")

def map_event_or_group_from_ai(ai_data: dict):
    """
    Map AI-provided hint to exact event_id or group_id from EVENTS JSON.
    Differentiates between events and groups.
    """
    hint = ai_data.get("event_id") or ai_data.get("summary") or ""
    norm_hint = normalize_text(hint)

    # 1️⃣ Check events first
    for e in EVENTS:
        event_id_norm = normalize_text(e.get("event_id"))
        event_name_norm = normalize_text(e.get("event_id_label"))  # optional
        if norm_hint == event_id_norm or norm_hint == event_name_norm or (event_name_norm and norm_hint in event_name_norm):
            return {
                "event_id": e.get("event_id"),
                "group_id": e.get("group_id"),
                "alertStatus": e.get("alert_status")
            }

    # 2️⃣ If no event match, check groups
    for g, events in GROUP_MAP.items():
        group_norm = normalize_text(g)
        if norm_hint == group_norm or (group_norm and norm_hint in group_norm):
            return {
                "event_id": None,
                "group_id": events[0].get("group_id"),  # first event's group_id
                "alertStatus": events[0].get("alert_status")
            }

    # fallback
    return {"event_id": None, "group_id": None, "alertStatus": None}


def map_event_from_ai(ai_data: dict):
    """
    Map AI-provided event description to exact event_id and group_id
    from EVENTS JSON.
    """
    event_hint = ai_data.get("event_id") or ai_data.get("summary") or ""
    norm_hint = normalize_text(event_hint)
    print('nnnn',norm_hint)

    for e in EVENTS:
        # Normalize event_id or event_name in your JSON to compare
        event_name_norm = normalize_text(e.get("event_id_label", ""))  # if your JSON has a name field
        event_id_norm = normalize_text(e.get("event_id", ""))
        # Exact or partial match on AI hint
        if norm_hint == event_id_norm or norm_hint == event_name_norm or norm_hint in event_name_norm:
            return {
                "event_id": e["event_id"],
                "group_id": e["group_id"],
                "alertStatus": e.get("alert_status")
            }

    # fallback if no match
    return {"event_id": None, "group_id": None, "alertStatus": None}

@app.route("/extract_id", methods=["POST"])
def extract_id():
    try:
        data = request.get_json()
        text = data.get("text", "")

        # AI extracts facility_id, event_id, group_id, intent, summary, alertStatus
        prompt = f"""
        You are a smart assistant that extracts structured information from user queries about facilities and events. 

        Rules:
        1. Extract the facility_id: this is always a user-provided identifier like GREAT_FALLS_100, NV1, abc2, etc. USe underscore (_) if space are there in facility_id. If user says specifically current location then only provide CURRENT_LOCATION in facility_id otherwise null.
        2. If user says *current location* then only provide CURRENT_LOCATION in facility_id.
        3. Extract the event_id: this is a specific event in the system, like flash_flood_advisory_level3, low_temperature_1, rain_advisory_level2.
        4. group_id → must be one of: {VALID_GROUPS}. if event_id is present then group id is just a parent category of that so extract from that if matching with {VALID_GROUPS}. 
        Match even if user writes without underscore or with spaces 
        (e.g. "snow ice" → "Snow/Ice"). 
        5. Extract alertStatus: classify as alert, warning, safe, or null based on user language.
        6. Determine intent:
            - "ALL" if user asks for all IDs
            - "SUBSCRIBED" if user asks for my ids, subscribed IDs or related to user.
            - "SPECIFIC" if they mention specific facility IDs
        7. Write a short plain-English summary of the user's request.

        Important:
        - Facility ID, event_id, and group_id are all independent; they may appear together or separately.
        - User input may not match exactly — handle synonyms, spacing, capitalization, and small typos.
        - Return **valid JSON only** with keys: facility_id, event_id, group_id, alertStatus, intent, summary.

        Examples:

        Input: "show me flash flood advisory for GREAT_FALLS_100"
        Output:
        {{
            "facility_id": "GREAT_FALLS_100",
            "event_id": "flash_flood_advisory_level3",
            "group_id": "flooding",
            "alertStatus": "alert",
            "intent": "SPECIFIC",
            "summary": "User requests flash flood advisory for facility GREAT_FALLS_100."
        }}

        Input: "show all warnings for abc2"
        Output:
        {{
            "facility_id": "abc2",
            "event_id": null,
            "group_id": null,
            "alertStatus": "warning",
            "intent": "ALL",
            "summary": "User requests all warnings related to abc2."
        }}

        Input: "{text}"
        """


        ai_data = {}
        if client:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw_answer = response.choices[0].message.content.strip()
            if raw_answer.startswith("```"):
                raw_answer = raw_answer.strip("`").replace("json\n", "").replace("json", "")
            ai_data = json.loads(raw_answer)

        print(ai_data)
        # Resolve event_id, group_id from JSON
        mapped_event = map_event_or_group_from_ai(ai_data)
        # event_data = resolve_event_group(text)

        # Combine AI extracted facility_id with deterministic event/group
        result = {
            "facility_id": ai_data.get("facility_id"),  # directly from user input
            "intent": ai_data.get("intent"),
            "summary": ai_data.get("summary"),
            "alertStatus": ai_data.get("alertStatus") or mapped_event.get("alertStatus"),
            "event_id": mapped_event.get("event_id"),
            "group_id": mapped_event.get("group_id")
        }

        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
