# config.py
SYSTEM_AND_TASK = """You are participating in a short behavioral economics survey.

In each round you must choose between:
• A car trip that uses 30 L of fuel
• A train trip that costs €60

Act as an ordinary person in Germany who owns a car and sometimes uses trains.
Write which option you would choose in each round, and briefly explain why.
"""

TREATMENTS = {
    "price_only": "Prices simply rise each round.",
    "emissions": "You are told the car emits 75 kg CO₂, the train almost none.",
    "carbon_price": "Each fuel price already includes a rising CO₂ price.",
    "carbon_price_info": (
        "Each fuel price includes a rising CO₂ price. "
        "Revenue is used to lower electricity costs and fund climate projects."
    ),
}

PRICE_LIST = [54, 60, 66, 72, 78, 84, 90, 94, 100]

def make_prompt(group: str) -> str:
    intro = SYSTEM_AND_TASK + "\n\nTreatment information:\n" + TREATMENTS[group]
    rounds = "\n\nPrice list:\n" + "\n".join(
        [f"Round {i+1}: Fuel = €{p} | Train = €60" for i,p in enumerate(PRICE_LIST)]
    )
    return intro + rounds + "\n\nFor each round, say which option you choose and why."

def get_groups():
    return list(TREATMENTS.keys())

