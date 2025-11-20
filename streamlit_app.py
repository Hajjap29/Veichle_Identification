import os
# The dotenv library is only needed if you are loading the .env file locally.
# If you are deploying on Streamlit Cloud and using st.secrets, you won't need this.
from dotenv import load_dotenv 
import base64
import json
import requests

# Load environment variables from a local .env file (explicit path)
load_dotenv(dotenv_path=r"D:\Project for fun\Car Model Recognition\notebooks\.env") 

# 🛡️ SECURE: Get the key only from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

MODEL = "gpt-4o-mini" 

# ⚠️ Important: The standard OpenAI API URL for Chat Completions is generally /v1/chat/completions 
# or /v1/images/generations. The endpoint /v1/responses is non-standard or might be incorrect.
# For a standard call (assuming you're using chat/vision):
API_URL = "https://api.openai.com/v1/chat/completions" 

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# Now you can use HEADERS in your API call...



def image_file_to_data_uri(path):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    # Try to infer mime type from extension (very simple)
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg"
    if ext in [".png"]:
        mime = "image/png"
    elif ext in [".webp"]:
        mime = "image/webp"
    return f"data:{mime};base64,{b64}"

# Quick test
# data_uri = image_file_to_data_uri("example.jpg")
# print(data_uri[:200])  # preview start (do not print full binary)


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


def call_vision_api_with_image(data_uri, prompt_text=PROMPT_JSON, model=MODEL):
    """
    Sends the image (as a data URI) and prompt to the Responses API.
    Returns the raw text output (we'll parse JSON out of it).
    """
    # Build the request body in a compact "input" form.
    # Many example doc patterns send a list with an image content block together
    # with a text block. The exact schema may vary slightly—check docs if you get errors.
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": data_uri}
                ],
            }
        ]
    }

    resp = requests.post(API_URL, headers=HEADERS, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()

# Example call:
# data_uri = image_file_to_data_uri("car_photo.jpg")
# r = call_vision_api_with_image(data_uri)
# print(json.dumps(r, indent=2)[:2000])


def extract_json_from_response(resp_json):
    """Robustly extracts the first JSON object from likely text fields in the API response.
    Prefers structured 'output' content, falls back to 'choices' and then the full dump.
    Uses json.JSONDecoder().raw_decode to find the first valid object substring.
    Returns a dict (parsed JSON) or {'raw_text':..., 'error':...}."""
    candidates = []

    # 1) Check Responses API 'output' shape for message/content blocks
    try:
        if isinstance(resp_json.get("output"), list):
            for item in resp_json.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        # handle 'output_text' or simple 'text' keys
                        if isinstance(c, dict):
                            txt = c.get("text") or c.get("content") or None
                            if isinstance(txt, str):
                                candidates.append(txt)
    except Exception:
        pass

    # 2) Fallback: older 'choices' shape
    if not candidates and "choices" in resp_json:
        try:
            choice_msg = resp_json["choices"][0].get("message", {}).get("content")
            if isinstance(choice_msg, str):
                candidates.append(choice_msg)
        except Exception:
            pass

    # 3) Final fallback: stringify the whole response
    if not candidates:
        candidates.append(json.dumps(resp_json))

    full_text = "\n".join(candidates)

    # Attempt to find the first JSON object substring using a raw_decode scan
    decoder = json.JSONDecoder()
    text = full_text
    for i in range(len(text)):
        try:
            obj, idx = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return {"raw_text": full_text, "error": "no JSON object found"}

# Example usage:
# parsed = extract_json_from_response(r)
# print(parsed)


# A simple mapping from coarse class → lifetime CO2e range (tons CO2e)
# These numbers are illustrative; refine with GREET/ICCT or regional data.
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

def estimate_carbon_from_detection(detection):
    # detection is the parsed JSON from the model
    cls = detection.get("vehicle_class", "unknown")
    powertrain = detection.get("powertrain", "unknown")
    cfg = CARBON_TABLE.get(cls, CARBON_TABLE["unknown"])
    # Optionally adjust EVs / hybrids: EVs have higher manufacturing but lower use-phase
    if powertrain == "electric" and cls != "unknown":
        # approximate adjustment for EV: slightly lower lifetime for many contexts (refine as needed)
        # shift the min/max down by ~20% after manufacturing accounted separately (illustrative)
        if cfg["lifetime_tons_min"] is not None:
            min_est = max(0, int(cfg["lifetime_tons_min"] * 0.7))
            max_est = int(cfg["lifetime_tons_max"] * 0.8)
            return {"lifetime_min_tons": min_est, "lifetime_max_tons": max_est, "note": "EV estimate; grid dependency not included"}
    return {"lifetime_min_tons": cfg["lifetime_tons_min"], "lifetime_max_tons": cfg["lifetime_tons_max"], "note": "category-level estimate"}


def analyze_image_file(path):
    print("Encoding image...")
    data_uri = image_file_to_data_uri(path)
    print("Calling vision API...")
    resp = call_vision_api_with_image(data_uri)
    print("API returned; extracting JSON...")
    parsed = extract_json_from_response(resp)
    print("Parsed detection:", parsed)
    est = estimate_carbon_from_detection(parsed if isinstance(parsed, dict) else {})
    result = {"detection": parsed, "carbon_estimate": est}
    return result

# Example usage:
# result = analyze_image_file("my_car_photo.jpg")
# print(json.dumps(result, indent=2))


def refined_estimate_from_user_inputs(detection, miles_per_year=None, years=None, regional_grid_emissions=None):
    """Refine lifetime estimate using optional user inputs.

    Approach:
    - mid_lifetime (tons) is midpoint from CARBON_TABLE
    - manufacturing = 30% of mid_lifetime, use_phase_total = 70%
    - if `miles_per_year` and `years` provided, compute per-mile and annual use-phase estimates
    """
    cls = (detection or {}).get("vehicle_class", "unknown")
    cfg = CARBON_TABLE.get(cls, {})
    if not cfg or cfg.get("lifetime_tons_min") is None:
        return {"error": "Cannot estimate; unknown class"}

    mid_lifetime = (cfg["lifetime_tons_min"] + cfg["lifetime_tons_max"]) / 2.0
    manufacturing = mid_lifetime * 0.3
    use_phase_total = mid_lifetime * 0.7  # tons over lifetime

    result = {
        "manufacturing_tons": manufacturing,
        "use_phase_total_tons": use_phase_total,
    }

    if miles_per_year and years:
        lifetime_miles = miles_per_year * years
        per_mile_tons = (use_phase_total / lifetime_miles) if lifetime_miles > 0 else None
        annual_emissions_tons = (per_mile_tons * miles_per_year) if per_mile_tons is not None else (use_phase_total / years)
        lifetime_estimate = manufacturing + annual_emissions_tons * years
        # also provide per-mile in grams for easier interpretation (1 ton = 1e6 grams)
        per_mile_g = per_mile_tons * 1e6 if per_mile_tons is not None else None

        result.update({
            "per_mile_tons": per_mile_tons,
            "per_mile_g": per_mile_g,
            "use_phase_est_annual_tons": annual_emissions_tons,
            "estimated_lifetime_tons": lifetime_estimate,
            "note": "Estimates are illustrative. For EVs, grid intensity affects use-phase.",
        })
    else:
        result["note"] = "Provide miles_per_year and years to refine per-mile and annual estimates."

    # If regional_grid_emissions provided and detection indicates electric, add a note (no calculation here).
    if regional_grid_emissions is not None and (detection or {}).get("powertrain") == "electric":
        result["grid_emissions_note"] = "Regional grid CO2 intensity provided; consider integrating it into EV use-phase calculation."

    return result


# Test extract_json_from_response with canned Responses API shapes
sample_resp_1 = {
    'output': [
        {'type': 'message', 'content': [{'type': 'output_text', 'text': '{"make":"Toyota","model":null,"year_range":"2016-2020","vehicle_class":"midsize","powertrain":"gasoline","confidence":0.8}'}]}
    ]
}
# Use a single-quoted string with explicit \n for newlines and valid JSON embedded
sample_resp_2 = {
    'choices': [
        {
            'message': {
                'content': 'Intro text\n{"make":"Honda","model":null,"year_range":null,"vehicle_class":"compact","powertrain":"diesel","confidence":0.6}\nThanks'
            }
        }
    ]
}

print('--- Test 1: structured output message with JSON in text ---')
print(extract_json_from_response(sample_resp_1))
print('--- Test 2: choices-style response with JSON embedded ---')
print(extract_json_from_response(sample_resp_2))

# Demonstrate refined estimate using the parsed detection (no API call required)
parsed = extract_json_from_response(sample_resp_1)
if isinstance(parsed, dict) and parsed.get('vehicle_class'):
    print('Refined estimate (miles_per_year=12000, years=12):')
    print(refined_estimate_from_user_inputs(parsed, miles_per_year=12000, years=12))
else:
    print('Parsing failed; parsed object:', parsed)




import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.title("Vehicle Carbon Estimator (Vision API)")

uploaded_image = st.file_uploader("Upload an image of a vehicle", type=["jpg","jpeg","png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", width=400)

    if st.button("Analyze Vehicle"):
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.getvalue())

        result = analyze_image_file("temp_image.jpg")
        st.subheader("Vision API Output")
        st.json(result)
import os
# The dotenv library is only needed if you are loading the .env file locally.
# If you are deploying on Streamlit Cloud and using st.secrets, you won't need this.
from dotenv import load_dotenv 
import base64
import json
import requests

# Load environment variables from a local .env file (explicit path)
load_dotenv(dotenv_path=r"D:\Project for fun\Car Model Recognition\notebooks\.env") 

# 🛡️ SECURE: Get the key only from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

MODEL = "gpt-4o-mini" 

# ⚠️ Important: The standard OpenAI API URL for Chat Completions is generally /v1/chat/completions 
# or /v1/images/generations. The endpoint /v1/responses is non-standard or might be incorrect.
# For a standard call (assuming you're using chat/vision):
API_URL = "https://api.openai.com/v1/chat/completions" 

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# Now you can use HEADERS in your API call...



def image_file_to_data_uri(path):
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    # Try to infer mime type from extension (very simple)
    ext = os.path.splitext(path)[1].lower()
    mime = "image/jpeg"
    if ext in [".png"]:
        mime = "image/png"
    elif ext in [".webp"]:
        mime = "image/webp"
    return f"data:{mime};base64,{b64}"

# Quick test
# data_uri = image_file_to_data_uri("example.jpg")
# print(data_uri[:200])  # preview start (do not print full binary)


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


def call_vision_api_with_image(data_uri, prompt_text=PROMPT_JSON, model=MODEL):
    """
    Sends the image (as a data URI) and prompt to the Responses API.
    Returns the raw text output (we'll parse JSON out of it).
    """
    # Build the request body in a compact "input" form.
    # Many example doc patterns send a list with an image content block together
    # with a text block. The exact schema may vary slightly—check docs if you get errors.
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt_text},
                    {"type": "input_image", "image_url": data_uri}
                ],
            }
        ]
    }

    resp = requests.post(API_URL, headers=HEADERS, json=body, timeout=60)
    resp.raise_for_status()
    return resp.json()

# Example call:
# data_uri = image_file_to_data_uri("car_photo.jpg")
# r = call_vision_api_with_image(data_uri)
# print(json.dumps(r, indent=2)[:2000])


def extract_json_from_response(resp_json):
    """Robustly extracts the first JSON object from likely text fields in the API response.
    Prefers structured 'output' content, falls back to 'choices' and then the full dump.
    Uses json.JSONDecoder().raw_decode to find the first valid object substring.
    Returns a dict (parsed JSON) or {'raw_text':..., 'error':...}."""
    candidates = []

    # 1) Check Responses API 'output' shape for message/content blocks
    try:
        if isinstance(resp_json.get("output"), list):
            for item in resp_json.get("output", []):
                if item.get("type") == "message":
                    for c in item.get("content", []):
                        # handle 'output_text' or simple 'text' keys
                        if isinstance(c, dict):
                            txt = c.get("text") or c.get("content") or None
                            if isinstance(txt, str):
                                candidates.append(txt)
    except Exception:
        pass

    # 2) Fallback: older 'choices' shape
    if not candidates and "choices" in resp_json:
        try:
            choice_msg = resp_json["choices"][0].get("message", {}).get("content")
            if isinstance(choice_msg, str):
                candidates.append(choice_msg)
        except Exception:
            pass

    # 3) Final fallback: stringify the whole response
    if not candidates:
        candidates.append(json.dumps(resp_json))

    full_text = "\n".join(candidates)

    # Attempt to find the first JSON object substring using a raw_decode scan
    decoder = json.JSONDecoder()
    text = full_text
    for i in range(len(text)):
        try:
            obj, idx = decoder.raw_decode(text[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return {"raw_text": full_text, "error": "no JSON object found"}

# Example usage:
# parsed = extract_json_from_response(r)
# print(parsed)


# A simple mapping from coarse class → lifetime CO2e range (tons CO2e)
# These numbers are illustrative; refine with GREET/ICCT or regional data.
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

def estimate_carbon_from_detection(detection):
    # detection is the parsed JSON from the model
    cls = detection.get("vehicle_class", "unknown")
    powertrain = detection.get("powertrain", "unknown")
    cfg = CARBON_TABLE.get(cls, CARBON_TABLE["unknown"])
    # Optionally adjust EVs / hybrids: EVs have higher manufacturing but lower use-phase
    if powertrain == "electric" and cls != "unknown":
        # approximate adjustment for EV: slightly lower lifetime for many contexts (refine as needed)
        # shift the min/max down by ~20% after manufacturing accounted separately (illustrative)
        if cfg["lifetime_tons_min"] is not None:
            min_est = max(0, int(cfg["lifetime_tons_min"] * 0.7))
            max_est = int(cfg["lifetime_tons_max"] * 0.8)
            return {"lifetime_min_tons": min_est, "lifetime_max_tons": max_est, "note": "EV estimate; grid dependency not included"}
    return {"lifetime_min_tons": cfg["lifetime_tons_min"], "lifetime_max_tons": cfg["lifetime_tons_max"], "note": "category-level estimate"}


def analyze_image_file(path):
    print("Encoding image...")
    data_uri = image_file_to_data_uri(path)
    print("Calling vision API...")
    resp = call_vision_api_with_image(data_uri)
    print("API returned; extracting JSON...")
    parsed = extract_json_from_response(resp)
    print("Parsed detection:", parsed)
    est = estimate_carbon_from_detection(parsed if isinstance(parsed, dict) else {})
    result = {"detection": parsed, "carbon_estimate": est}
    return result

# Example usage:
# result = analyze_image_file("my_car_photo.jpg")
# print(json.dumps(result, indent=2))


def refined_estimate_from_user_inputs(detection, miles_per_year=None, years=None, regional_grid_emissions=None):
    """Refine lifetime estimate using optional user inputs.

    Approach:
    - mid_lifetime (tons) is midpoint from CARBON_TABLE
    - manufacturing = 30% of mid_lifetime, use_phase_total = 70%
    - if `miles_per_year` and `years` provided, compute per-mile and annual use-phase estimates
    """
    cls = (detection or {}).get("vehicle_class", "unknown")
    cfg = CARBON_TABLE.get(cls, {})
    if not cfg or cfg.get("lifetime_tons_min") is None:
        return {"error": "Cannot estimate; unknown class"}

    mid_lifetime = (cfg["lifetime_tons_min"] + cfg["lifetime_tons_max"]) / 2.0
    manufacturing = mid_lifetime * 0.3
    use_phase_total = mid_lifetime * 0.7  # tons over lifetime

    result = {
        "manufacturing_tons": manufacturing,
        "use_phase_total_tons": use_phase_total,
    }

    if miles_per_year and years:
        lifetime_miles = miles_per_year * years
        per_mile_tons = (use_phase_total / lifetime_miles) if lifetime_miles > 0 else None
        annual_emissions_tons = (per_mile_tons * miles_per_year) if per_mile_tons is not None else (use_phase_total / years)
        lifetime_estimate = manufacturing + annual_emissions_tons * years
        # also provide per-mile in grams for easier interpretation (1 ton = 1e6 grams)
        per_mile_g = per_mile_tons * 1e6 if per_mile_tons is not None else None

        result.update({
            "per_mile_tons": per_mile_tons,
            "per_mile_g": per_mile_g,
            "use_phase_est_annual_tons": annual_emissions_tons,
            "estimated_lifetime_tons": lifetime_estimate,
            "note": "Estimates are illustrative. For EVs, grid intensity affects use-phase.",
        })
    else:
        result["note"] = "Provide miles_per_year and years to refine per-mile and annual estimates."

    # If regional_grid_emissions provided and detection indicates electric, add a note (no calculation here).
    if regional_grid_emissions is not None and (detection or {}).get("powertrain") == "electric":
        result["grid_emissions_note"] = "Regional grid CO2 intensity provided; consider integrating it into EV use-phase calculation."

    return result


# Test extract_json_from_response with canned Responses API shapes
sample_resp_1 = {
    'output': [
        {'type': 'message', 'content': [{'type': 'output_text', 'text': '{"make":"Toyota","model":null,"year_range":"2016-2020","vehicle_class":"midsize","powertrain":"gasoline","confidence":0.8}'}]}
    ]
}
# Use a single-quoted string with explicit \n for newlines and valid JSON embedded
sample_resp_2 = {
    'choices': [
        {
            'message': {
                'content': 'Intro text\n{"make":"Honda","model":null,"year_range":null,"vehicle_class":"compact","powertrain":"diesel","confidence":0.6}\nThanks'
            }
        }
    ]
}

print('--- Test 1: structured output message with JSON in text ---')
print(extract_json_from_response(sample_resp_1))
print('--- Test 2: choices-style response with JSON embedded ---')
print(extract_json_from_response(sample_resp_2))

# Demonstrate refined estimate using the parsed detection (no API call required)
parsed = extract_json_from_response(sample_resp_1)
if isinstance(parsed, dict) and parsed.get('vehicle_class'):
    print('Refined estimate (miles_per_year=12000, years=12):')
    print(refined_estimate_from_user_inputs(parsed, miles_per_year=12000, years=12))
else:
    print('Parsing failed; parsed object:', parsed)




import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.title("Vehicle Carbon Estimator (Vision API)")

uploaded_image = st.file_uploader("Upload an image of a vehicle", type=["jpg","jpeg","png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", width=400)

    if st.button("Analyze Vehicle"):
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_image.getvalue())

        result = analyze_image_file("temp_image.jpg")
        st.subheader("Vision API Output")
        st.json(result)
