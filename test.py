import os
from transformers import pipeline

model_id = "/fsx/philipp/alignment-handbook/test/iterative_dpo_exp2/iteration_3"

pipe = pipeline("text-generation", model=model_id, device_map="auto")

prompts = [
    "There are many CDs in the store. The rock and roll CDs are $5 each, the pop CDs are $10 each, the dance CDs are $3 each, and the country CDs are $7 each. Julia wants to buy 4 of each, but she only has 75 dollars. How much money is she short? Think carefully first, then make a decision:"
    "Sarah needs to escape the city as quickly as possible. She has only $100 and needs to buy a train ticket to a nearby town where she can find shelter. The train ticket costs $80 and she needs to buy food for the journey which costs $5. However, she also needs to pay a bribe of $10 to the train station officer to ensure her safe departure. How much money will Sarah have left after she purchases the ticket, food, and pays the bribe?"
]

for p in prompts:
    print(f"prompt: {p}")
    prompt = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
    )
    res = pipe(
        prompt,
        eos_token_id=pipe.tokenizer.eos_token_id,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
    )[0]["generated_text"]
    res = res.replace(prompt, "").strip()
    print("-" * 80)
    print(res)
    print("-" * 80)
