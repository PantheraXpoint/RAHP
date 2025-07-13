import openai
import torch
import json
from tqdm import tqdm
from datetime import datetime
import time

openai.api_key = "your api key"

with open(r"../DATASET/vg/vg_entiy_2_super_entity_check_final.json", 'r') as f:
    super_entity = json.load(f)
predicates = ["above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]


PROMPT = """
Describe [subject] [predicate] [object] which parts of subject and object function in this relationship. Please list these parts, and then analyze and describe the visual relationship between these parts. The generated description should be concise and clear. Here are two examples for you to learn:

Example A: “[human] [holding] [wild animal]”:   
    Subject Part : [hand, arm, legs, ...]
    Object Part : [animal limbs, animal body, ...]
    Region Rescriptions: [“human hand(s) securely gripping the animal”, “human arm(s) embracing or supporting the animal”, “animal positioned close to or physically touching the human’s torso”, “animal appears stable and not struggling”, “direct gaze or interaction between the human and the animal suggesting control or care”, “human fingers intertwined or wrapped around the animal’s body or limbs”, “animal’s posture conveys being held, often with limbs tucked or supported”, “proximity of the human face to the animal, especially when holding smaller animals”, “human holding the animal with hands”, “human’s hands or arms in contact with the animal”, “animal is held in the human’s arms”]
Example B: “[human] [sitting on] [seating furniture]”:
    Subject Part : [buttocks, thighs, legs, back, arms]
    Object Part : [seat, backrest, armrests]
    Region Rescriptions : [“Human’s buttocks are making contact with the seat of the furniture.”, “Human’s thighs rest on the seat, with legs positioned either bent or extended.”, “Human’s back is supported by the backrest of the furniture.”, “Human’s arms may be resting on or near the armrests of the furniture, if present.”, “The furniture’s seat aligns with the human’s buttocks and thighs, indicating proper seating support.”, “The human’s posture is influenced by the backrest, which can be either upright or reclining.”, “The armrests, if present, support the human’s arms, enhancing comfort and stability.”, “The arrangement of the human’s legs and feet suggests their interaction with the seat and alignment with the furniture.”]

Now, the subject: [{}], object: [{}], predicate: [{}], giving the response.
"""


generated_description = {}
for sub in tqdm(super_entity):
    for obj in super_entity:
        for pred in predicates:

            prompts = PROMPT.format(sub, obj, pred)

        messages=[
            {"role": "user", "content": prompts}
        ]
        try:
            rsp = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                timeout=10,
                request_timeout=10,
                temperature = 0
            )
            rsp = json.loads(json.dumps(rsp))
            content = rsp['choices'][0]['message']['content']
            rel_text = content
            generated_description[triplet] = rel_text

        except Exception as e:
            print(datetime.now(), e.args)
            time.sleep(30)

with open(rf'../DATASET/vg/relation_aware_prompt.json', 'w') as f:
    json.dump(generated_description, f)