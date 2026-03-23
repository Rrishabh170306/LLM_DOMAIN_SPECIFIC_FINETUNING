import json
import random
import requests
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
TOTAL_SAMPLES = 200
BATCH_SIZE = 32
OUTPUT_FILE = "indian_law_alpaca_200.json"
MAX_WORKERS = 1
REQUEST_TIMEOUT = 120

random.seed(42)

topics = [
    "Indian Penal Code (IPC)",
    "Constitution of India",
    "Indian Contract Act, 1872",
    "Law of Torts",
    "Criminal Procedure Code (CrPC)",
    "Civil Procedure Code (CPC)",
    "Information Technology Act, 2000",
    "Indian Evidence Act, 1872"
]

case_scenarios = [
    "A kills B during a sudden fight without premeditation.",
    "A refuses to honor a signed contract causing loss to B.",
    "A person is arrested without being informed of grounds.",
    "Government restricts free speech citing public order.",
    "A company leaks sensitive personal data of users.",
    "A driver negligently hits a pedestrian causing injury.",
    "Police obtain confession under coercion.",
    "A minor enters into a contract and later refuses performance."
]

instruction_types = [
    "Analyze the following case under {}.",
    "Provide a detailed legal answer under {}.",
    "Apply {} to the given scenario.",
    "Frame legal issues and decide the case under {}.",
    "Present moot court arguments for both sides under {}."
]

def mutate_scenario(base):
    variations = [
        base,
        base + " The incident occurred at night.",
        base + " There were multiple witnesses.",
        base + " The accused claims self-defense.",
        base + " The victim survived with injuries."
    ]
    return random.choice(variations)


def query_llm(prompt, retries=2):
    for _ in range(retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive":-1,
                    "options": {
                        "temperature": 0.7,
                        "num_ctx": 4096,      
                        "num_predict": 1200   
                    }
                },
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except:
            time.sleep(1)
    return "ERROR"


def build_prompt(instruction, scenario, topic):
    return f"""
You are an Indian lawyer. Use real laws and case precedents.

Instruction:
{instruction}

Scenario:
{scenario}

Format:
Facts:
Issues:
Relevant Law:
Analysis:
Conclusion:
"""


def is_valid(entry):
    out = entry["output"]
    required = ["Facts:", "Issues:", "Relevant Law:", "Analysis:", "Conclusion:"]
    return out and not out.startswith("ERROR") and all(r in out for r in required)


def hash_entry(e):
    return hashlib.md5((e["instruction"] + e["input"] + e["output"]).encode()).hexdigest()

def generate_entry():
    topic = random.choice(topics)
    scenario = mutate_scenario(random.choice(case_scenarios))
    instruction = random.choice(instruction_types).format(topic)

    prompt = build_prompt(instruction, scenario, topic)
    output = query_llm(prompt)

    return {
        "instruction": instruction.strip(),
        "input": scenario.strip(),
        "output": output.strip()
    }


def generate_dataset():
    dataset, seen = [], set()

    while len(dataset) < TOTAL_SAMPLES:
        batch_size = min(BATCH_SIZE, TOTAL_SAMPLES - len(dataset))

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(generate_entry) for _ in range(batch_size)]

            for f in as_completed(futures):
                e = f.result()

                if not is_valid(e):
                    continue

                h = hash_entry(e)
                if h in seen:
                    continue

                seen.add(h)
                dataset.append(e)

                if len(dataset) >= TOTAL_SAMPLES:
                    break

        print(f"Progress: {len(dataset)}/{TOTAL_SAMPLES}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print("✅ Dataset generation complete")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print("✅ Dataset generation complete")
    
    print("🧹 Unloading model from memory...")
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": MODEL_NAME, "keep_alive": 0}
        )
    except Exception as e:
        pass


if __name__ == "__main__":
    generate_dataset()