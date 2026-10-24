"""Utility script to verify API connectivity and check availability of gemma-4-31b-it.

SECURITY: Never prints, logs, or exposes GEMINI_API_KEY.
"""

from ai_analyzer import AIAnalyzer, UnsupportedModelError, AIAnalysisError, get_gemini_api_key
from google import genai

def main():
    try:
        key = get_gemini_api_key()
        client = genai.Client(api_key=key)
        print("Connected to Google GenAI Client successfully.")
    except Exception as e:
        print(f"Failed to initialize client: {type(e).__name__}")
        return

    # Check model list
    try:
        models = [m.name for m in client.models.list()]
        gemma_models = [m for m in models if "gemma" in m.lower()]
        print(f"Total models available: {len(models)}")
        print(f"Gemma models found: {gemma_models}")
        is_gemma_4_avail = any("gemma-4-31b-it" in m for m in models)
        print(f"Is 'gemma-4-31b-it' available in model list: {is_gemma_4_avail}")
    except Exception as e:
        print(f"Model listing failed: {type(e).__name__}: {e}")

    # Test analyze_expression with gemma-4-31b-it
    print("Testing analyze_expression with gemma-4-31b-it...")
    analyzer = AIAnalyzer(model="gemma-4-31b-it")
    try:
        res = analyzer.analyze_expression("v = d / t", context="kinematics")
        print("gemma-4-31b-it call succeeded!")
        print("Summary:", res.summary)
        print("Candidate units:", res.candidate_units)
    except UnsupportedModelError as e:
        print(f"gemma-4-31b-it is currently unsupported/unavailable as expected: {e}")
    except Exception as e:
        print(f"Call produced: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
