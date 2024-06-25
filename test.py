  
import os 
from transformers import pipeline
model_id = "/fsx/philipp/alignment-handbook/test/offline_dpo"

pipe =  pipeline("text-generation", model=model_id,device_map="auto")

prompts = [
    'What should I do if I do not receive a confirmation upon casting my ballot for the CICR election?\nGenerate according to: All members of CICR are asked to vote.\nVoting will be conducted electronically through myAACR. You will need a myAACR login to vote. Due to a recent upgrade in the myAACR function, please view how to easily recover your AACR login password, here.\nClick on \'View All Activities\'\nPlease remember to complete all steps to ensure that you have officially cast your ballot. You will receive a confirmation immediately upon casting your ballot. If you do not receive this confirmation, you will need to access the ballot again to ensure it is completed.\nIf you have difficulties logging in or voting, please email cicr@aacr.org for assistance.\nThe following two individuals are standing for election to the office of chairperson-elect of the Steering Committee of the Chemistry in Cancer Research (CICR) Working Group of the American Association for Cancer Research (AACR). The individual elected shall serve as chairperson-elect commencing at the AACR Annual Meeting 2019, until the following Annual Meeting after which that individual will assume the office of chairperson for one year, and past-chairperson for an additional year.\nCandidates background and goals are listed on the election ballot.',
    'Write a eulogy for a public figure who inspired you.',
    'How long does an average course of chiropractic treatment last, and how often might a patient need to schedule appointments?'
]

for p in prompts:
    print(f'prompt: {p}')
    prompt = pipe.tokenizer.apply_chat_template([{'role': 'user', 'content': p}],tokenize=False,add_generation_prompt=True)
    res = pipe(prompt, eos_token_id=pipe.tokenizer.eos_token_id,max_new_tokens=1024, do_sample=True, temperature=0.9, top_k=50, top_p=0.95)[0]['generated_text']
    res = res.replace(prompt, '').strip()
    print('-' * 80)
    print(res)
    print('-' * 80)