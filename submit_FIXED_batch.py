"""
Submit FIXED Batch - Based on Alexander's Feedback
"""

import openai
import json

API_KEY = "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"
openai.api_key = API_KEY

print("="*70)
print("SUBMITTING FIXED BATCH - 500 TRIALS")
print("Based on Alexander's feedback:")
print("  ✓ Full text responses (no JSON structure)")
print("  ✓ No token limits")
print("  ✓ Elaborative reasoning for each round")
print("="*70)

# Upload
print("\n1. Uploading conformity_batch_FIXED_500trials.jsonl...")
with open("conformity_batch_FIXED_500trials.jsonl", "rb") as f:
    batch_file = openai.files.create(file=f, purpose="batch")
print(f"   ✓ File uploaded: {batch_file.id}")

# Create batch
print("\n2. Creating batch job...")
batch = openai.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

print(f"   ✓ Batch created: {batch.id}")
print(f"   Status: {batch.status}")

# Save info
with open("batch_info_FIXED.json", "w") as f:
    json.dump({
        "batch_id": batch.id,
        "file_id": batch_file.id,
        "submitted_at": "2025-11-03 (fixed version)"
    }, f, indent=2)

print("\n" + "="*70)
print("✅ BATCH SUBMITTED!")
print("="*70)
print(f"\nBatch ID: {batch.id}")
print("\nThis will take 2-4 hours to complete.")
print("The script exits immediately (Batch API runs on OpenAI servers).")
print("\nTo check status later:")
print("  Visit: https://platform.openai.com/batches")
print("  Or use the API to check batch status")
print("="*70)
