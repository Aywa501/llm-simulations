#!/usr/bin/env python3

import os
from openai import OpenAI

def main():
    print("OPENAI_API_KEY set:", bool(os.environ.get("OPENAI_API_KEY")))

    client = OpenAI()

    try:
        resp = client.responses.create(
            model="gpt-5.2",  # change if needed
            input="Return exactly this JSON: {\"status\":\"ok\"}",
        )

        print("\n=== RAW OUTPUT TEXT ===")
        print(resp.output_text)

        print("\n=== FULL RESPONSE OBJECT ===")
        print(resp)

    except Exception as e:
        print("\nAPI CALL FAILED")
        print(type(e).__name__, ":", e)

if __name__ == "__main__":
    main()

