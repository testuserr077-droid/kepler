from flask import Flask, request, jsonify
from openai import OpenAI
import os, json

app = Flask(__name__)

# Load ChatGPT client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Allowed groups (normalize user input like "snow ice" → "Snow/Ice")
VALID_GROUPS = [
    "Non Weather", "snow", "announcement_group", "Visibility", "Snow/Ice", 
    "Rain", "lightning", "storm", "wind_gusts", "Dust", "tornado", "snow_pack", 
    "hail", "flooding", "air_quality", "wind", "iceing", "shake_alert", 
    "red_flag -957", "flood", "psa_group", "temperature", "tropical_storm", "snow_acc"
]

@app.route("/extract_id", methods=["POST"])
def extract_id():
    data = request.get_json()
    text = data.get("text", "")

    # System prompt for AI
    prompt = f"""
    You are an assistant that extracts structured info from user queries.

    Tasks:
    1. facility_id → IDs with letters, numbers, underscores (no spaces). 
       Examples: TARA, NV1, GREAT_FALLS_MT, BNG9, IdJson2.
    2. event_id → anything that represents a specific event, e.g. low_temperature_1, rain_advisory_level2. 
    3. group_id → must be one of: {VALID_GROUPS}. if event_id is present then group id is just a parent category of that so extract from that if matching with {VALID_GROUPS}. 
       Match even if user writes without underscore or with spaces 
       (e.g. "snow ice" → "Snow/Ice").
    4. alertStatus:
        - "alert" → if user asks for alerts or high-alert IDs
        - "warning" → if user asks for warnings
        - "safe" → if user asks for safe/green IDs
        - null if not specified
    5. intent:
       - "ALL" if user asks for all IDs
       - "SUBSCRIBED" if user asks for subscribed IDs
       - "SPECIFIC" if they mention specific facility IDs
    6. summary → plain-English description.

    Return valid JSON only with keys:
    facility_id, event_id, group_id, alertStatus, intent, summary.

    Text: "{text}"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw_answer = response.choices[0].message.content.strip()

        # Clean up JSON if wrapped in ```json
        if raw_answer.startswith("```"):
            raw_answer = raw_answer.strip("`")
            raw_answer = raw_answer.replace("json\n", "").replace("json", "")

        parsed = json.loads(raw_answer)

        return jsonify({"success": True, "data": parsed})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    

if __name__ == "__main__":
    app.run(debug=True)
