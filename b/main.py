from flask import Flask, request, jsonify
from openai import OpenAI
import os,json

app = Flask(__name__)

# Load ChatGPT client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# # Load your 1000+ IDs into memory
# with open("ids.txt", "r") as f:
#     VALID_IDS = [line.strip() for line in f if line.strip()]

@app.route("/extract_id", methods=["POST"])
def extract_id():
    data = request.get_json()
    text = data.get("text", "")

    # Ask ChatGPT to find the ID(s) from text
    prompt = f"""
    You are an assistant that extracts structured info from user queries.

    Rules:
    1. IDs can contain letters, numbers, and underscores, but no spaces.
    Examples: TARA, NV1, GREAT_FALLS_MT, BNG9, IdJson2.
    2. If the user says "all ids", "everything", or similar → set "ids" to "ALL".
    3. Classify intent into one of:
    - "alert" → if user asks for alerts or high-alert IDs
    - "warning" → if user asks for warnings
    - "safe" → if user asks for safe/green IDs
    - "regular" → if user does not specify anything about alert/safe/warning
    4. Summary is a short plain-English description of what the user is asking.

    Return JSON only with keys: ids, intent, summary.

    Text: "{text}"
    """


    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-4o / gpt-3.5-turbo
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw_answer = response.choices[0].message.content.strip()

        if raw_answer.startswith("```"):
            raw_answer = raw_answer.strip("`")
            raw_answer = raw_answer.replace("json\n", "").replace("json", "")

        parsed = json.loads(raw_answer)

        return jsonify({"success": True, "data": parsed})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)
