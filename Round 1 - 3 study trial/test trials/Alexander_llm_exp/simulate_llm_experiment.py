# simulate_llm_experiment.py
import asyncio, os, uuid, json
from tqdm import tqdm
from openai import AsyncOpenAI
from config import make_prompt, get_groups

client = AsyncOpenAI()

MODEL = "gpt-5-mini"
N_PER_GROUP = 2          # raise to 500 for a full run
WRITE_INTERVAL = 100
DATA_DIR = "data"
PREFIX = "llm_carbon_experiment"


def next_filename():
    os.makedirs(DATA_DIR, exist_ok=True)
    existing = [
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(DATA_DIR)
        if f.startswith(PREFIX) and f.endswith(".json")
        and f.split("_")[-1].split(".")[0].isdigit()
    ]
    idx = max(existing, default=-1) + 1
    return os.path.join(DATA_DIR, f"{PREFIX}_{idx}.json")


async def simulate_one(group: str):
    pid = str(uuid.uuid4())
    prompt = make_prompt(group)
    try:
        # plain text input, low reasoning effort, no output cap
        r = await client.responses.create(
            model=MODEL,
            input=prompt,
            reasoning={"effort": "low"},
        )

        text = getattr(r, "output_text", None)
        if text and text.strip():
            return {"id": pid, "group": group, "response": text.strip()}
        return {
            "id": pid,
            "group": group,
            "error": "empty output_text",
            "raw_response": json.loads(r.model_dump_json()),
        }

    except Exception as e:
        return {"id": pid, "group": group, "error": str(e)}


async def main():
    groups = get_groups()
    outfile = next_filename()
    buffer, total = [], 0
    print(f"\nWriting results to {outfile}")

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("[\n")
        for g in groups:
            print(f"\n--- {g} ---")
            tasks = [simulate_one(g) for _ in range(N_PER_GROUP)]
            for t in tqdm(asyncio.as_completed(tasks), total=N_PER_GROUP):
                res = await t
                buffer.append(res)
                if len(buffer) >= WRITE_INTERVAL:
                    for entry in buffer:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write(",\n")
                        total += 1
                    f.flush()
                    buffer = []
        for i, entry in enumerate(buffer):
            json.dump(entry, f, ensure_ascii=False)
            if i < len(buffer) - 1:
                f.write(",\n")
        f.write("\n]\n")

    print(f"\n✅ Finished. {total + len(buffer)} records written.")


if __name__ == "__main__":
    asyncio.run(main())

