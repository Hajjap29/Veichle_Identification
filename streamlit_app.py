import streamlit as st
import base64
import json
import requests
import os

# --- Page Config ---
st.set_page_config(
    page_title="Car Carbon Estimator",
    page_icon="🚗",
    layout="centered"
)

# --- Constants & Configuration ---
MODEL = "gpt-4o-mini"
API_URL = "https://api.openai.com/v1/chat/completions"

PROMPT_JSON = """
You are an image-understanding assistant. I will provide an image. 
Respond ONLY with JSON (no extra text). The JSON must have the following keys:
- make: string or null
- model: string or null
- year_range: string or null (e.g., "2016-2020")
- vehicle_class: one of ["compact", "midsize", "fullsize", "suv", "pickup", "van", "motorcycle", "bus", "truck", "unknown"]
- powertrain: one of ["gasoline", "diesel", "hybrid", "plug-in hybrid", "electric", "unknown"]
- confidence: number between 0 and 1 (estimate of how confident you are)

Make conservative guesses. If uncertain, put null or "unknown". Don't output any explanatory text — ONLY the JSON object.
"""

CARBON_TABLE = {
    "compact":    {"lifetime_tons_min": 30, "lifetime_tons_max": 50},
    "midsize":    {"lifetime_tons_min": 40, "lifetime_tons_max": 65},
    "fullsize":   {"lifetime_tons_min": 55, "lifetime_tons_max": 85},
    "suv":        {"lifetime_tons_min": 50, "lifetime_tons_max": 80},
    "pickup":     {"lifetime_tons_min": 60, "lifetime_tons_max": 100},
    "van":        {"lifetime_tons_min": 45, "lifetime_tons_max": 80},
    "motorcycle": {"lifetime_tons_min": 10, "lifetime_tons_max": 25},
    "bus":        {"lifetime_tons_min": 150, "lifetime_tons_max": 400},
    "truck":      {"lifetime_tons_min": 120, "lifetime_tons_max": 350},
    "unknown":    {"lifetime_tons_min": None, "lifetime_tons_max": None}
}

# --- Helper Functions ---

def get_api_key():
    """
    Tries to get API key from Streamlit secrets first, then environment variables.
    """
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (FileNotFoundError, KeyError):
        return os.getenv("OPENAI_API_KEY")

def encode_image_from_buffer(uploaded_file):
    """
    Encodes a Streamlit UploadedFile (BytesIO) to base64 string.
    """
    bytes_data = uploaded_file.getvalue()
    b64 = base64.b64encode(bytes_data).decode("utf-8")
    # Determine mime type
    mime = "image/jpeg"
    if uploaded_file.type == "image/png":
        mime = "image/png"
    elif uploaded_file.type == "image/webp":
        mime = "image/webp"
    return f"data:{mime};base64,{b64}"

def call_vision_api(data_uri, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT_JSON},
                    {"type": "input_image", "image_url": data_uri}
                ],
            }
        ]
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def extract_json_from_response(resp_json):
    """
    Robustly extracts JSON from the API response.
    """
    candidates = []
    
    # Check for errors first
    if "error" in resp_json:
        return {"error": resp_json["error"]}

    # 1. Check 'output' (standard pattern in some libs)
    try:
        if isinstance(resp_json.get("output"), list):
            for item in resp_json.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        if isinstance(c, dict):
                            txt = c.get("text") or c.get("content")
                            if isinstance(txt, str):
                                candidates.append(txt)
    except Exception:
        pass

    # 2. Standard OpenAI 'choices' pattern
    if not candidates and "choices" in resp_json:
        try:
            choice_msg = resp_json["choices"][0].get("message", {}).get("content")
            if isinstance(choice_msg, str):
                candidates.append(choice_msg)
        except Exception:
            pass
            
    full_text = "\n".join(candidates)
    
    # Attempt to find JSON object
    decoder = json.JSONDecoder()
    for i in range(len(full_text)):
        try:
            obj, idx = decoder.raw_decode(full_text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return {"raw_text": full_text, "error": "No valid JSON object found in response"}

def estimate_carbon(detection):
    cls = detection.get("vehicle_class", "unknown")
    powertrain = detection.get("powertrain", "unknown")
    cfg = CARBON_TABLE.get(cls, CARBON_TABLE["unknown"])
    
    if cfg["lifetime_tons_min"] is None:
        return {"note": "Unknown category"}

    min_est = cfg["lifetime_tons_min"]
    max_est = cfg["lifetime_tons_max"]
    note = "Category-level estimate"

    # Simple EV adjustment
    if powertrain == "electric" and cls != "unknown":
        min_est = max(0, int(min_est * 0.7))
        max_est = int(max_est * 0.8)
        note = "EV estimate; grid dependency not included"
        
    return {
        "lifetime_min_tons": min_est,
        "lifetime_max_tons": max_est,
        "note": note
    }

def calculate_detailed_emissions(detection, miles_per_year, years):
    cls = detection.get("vehicle_class", "unknown")
    cfg = CARBON_TABLE.get(cls, {})
    
    if not cfg or cfg.get("lifetime_tons_min") is None:
        return None

    mid_lifetime = (cfg["lifetime_tons_min"] + cfg["lifetime_tons_max"]) / 2.0
    manufacturing = mid_lifetime * 0.3
    use_phase_total = mid_lifetime * 0.7
    
    lifetime_miles = miles_per_year * years
    if lifetime_miles <= 0:
        return None
        
    per_mile_tons = use_phase_total / lifetime_miles
    annual_emissions = per_mile_tons * miles_per_year
    total_lifetime = manufacturing + (annual_emissions * years)
    
    return {
        "manufacturing": manufacturing,
        "annual_emissions": annual_emissions,
        "total_lifetime": total_lifetime,
        "per_mile_g": per_mile_tons * 1e6
    }

# --- Main App Layout ---

st.title("🚗 AI Car Carbon Estimator")
st.markdown("Upload a photo of a vehicle to detect its model and estimate its lifecycle carbon footprint.")

# 1. API Key Check
api_key = get_api_key()
if not api_key:
    st.warning("⚠️ OpenAI API Key not found.")
    api_key = st.text_input("Enter OpenAI API Key:", type="password")

# 2. File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file and api_key:
    # Display Image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Analyze Vehicle"):
        with st.spinner("Analyzing image with AI..."):
            # Encode
            data_uri = encode_image_from_buffer(uploaded_file)
            
            # Call API
            api_response = call_vision_api(data_uri, api_key)
            
            # Parse
            detection = extract_json_from_response(api_response)
            
            if "error" in detection:
                st.error(f"Analysis Failed: {detection['error']}")
            else:
                st.success("Vehicle Detected!")
                
                # Save to session state so it persists if we change inputs below
                st.session_state['detection'] = detection
                st.session_state['carbon_est'] = estimate_carbon(detection)

# 3. Display Results (if available in session state)
if 'detection' in st.session_state:
    det = st.session_state['detection']
    est = st.session_state['carbon_est']
    
    # Columns for details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Detection")
        st.write(f"**Make:** {det.get('make', 'Unknown')}")
        st.write(f"**Model:** {det.get('model', 'Unknown')}")
        st.write(f"**Class:** {det.get('vehicle_class', 'Unknown')}")
        st.write(f"**Powertrain:** {det.get('powertrain', 'Unknown')}")
        st.metric("Confidence", f"{det.get('confidence', 0)*100:.0f}%")

    with col2:
        st.subheader("🌍 Carbon Estimate")
        if "lifetime_min_tons" in est:
            st.metric("Lifetime CO2e", f"{est['lifetime_min_tons']} - {est['lifetime_max_tons']} tons")
            st.info(est['note'])
        else:
            st.warning("Could not estimate carbon for this vehicle class.")

    st.markdown("---")
    st.subheader("📉 Refine Estimate")
    
    # Inputs for refinement
    miles = st.number_input("Miles driven per year", value=12000, step=1000)
    years = st.number_input("Years of ownership", value=12, step=1)
    
    detailed = calculate_detailed_emissions(det, miles, years)
    
    if detailed:
        st.markdown(f"### Estimated Breakdown over {years} years")
        d_col1, d_col2, d_col3 = st.columns(3)
        d_col1.metric("Manufacturing", f"{detailed['manufacturing']:.1f} tons")
        d_col2.metric("Annual Use", f"{detailed['annual_emissions']:.1f} tons/yr")
        d_col3.metric("Per Mile", f"{detailed['per_mile_g']:.0f} g/mile")
        
        st.progress(min(detailed['total_lifetime'] / 100, 1.0), text=f"Total Lifetime: {detailed['total_lifetime']:.1f} tons")
