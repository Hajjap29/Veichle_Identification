
import streamlit as st
import os, base64, json, requests

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def image_file_to_data_uri(path):
    with open(path, "rb") as f:
        data = f.read()
    mime = "image/jpeg"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

PROMPT_JSON = """
You are an image-understanding assistant. Identify the car make, model.
Respond with JSON only: { "make": "...", "model": "..." }
"""

def call_vision_api_with_image(data_uri, prompt_text=PROMPT_JSON):
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4.1",
        "input": [
            {"role":"user","content":[
                {"type":"input_text","text":prompt_text},
                {"type":"input_image","image_url": data_uri}
            ]}
        ]
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload))
    r.raise_for_status()
    return r.json()

def extract_json_from_response(resp_json):
    # naive extraction
    try:
        text = resp_json["output"][0]["content"][0]["text"]
    except:
        text = resp_json["choices"][0]["message"]["content"]
    try:
        return json.loads(text)
    except:
        return {}

st.title("Car Image Analyzer")

uploaded = st.file_uploader("Upload an image of a car", type=["jpg","jpeg","png"])
if uploaded:
    temp_path = "/tmp/upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded.read())
    data_uri = image_file_to_data_uri(temp_path)
    st.image(uploaded, caption="Uploaded Image")

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            resp = call_vision_api_with_image(data_uri)
            result = extract_json_from_response(resp)
        st.json(result)
