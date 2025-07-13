import json
import os
from tqdm import tqdm
import random
from datetime import datetime
import time
from openai import OpenAI
import re
import os
import base64

# vg_super_entities = [
#     'male', 'female', 'children', 'pets', 'wild animal', 'ground transport',
#     'water transport', 'air transport', 'sports equipment', 'seating furniture',
#     'decorative item', 'table', 'upper body clothing', 'lower body clothing',
#     'footwear', 'accessory', 'fruit', 'vegetable', 'prepared food', 'beverage',
#     'utensils', 'container', 'textile', 'landscape', 'urban feature', 'plant',
#     'structure', 'household item', 'head part', 'limb and appendage'
# ]
# oiv6_super_entities =  ['male', 'female', 'children', 'head feature', 'limb feature','torso feature', 'accessorie', 'mammal', 'bird', 'reptile', 'insect', 'marine animal','bike', 'ground vehicle', 'watercraft', 'aircraft', 'vehicle part item', 'ball-relatedsport item', 'water sport item', 'winter sportitem', 'seating furniture', 'table furniture','storage furniture', 'bedding', 'upper bodyclothing', 'lower body clothing', 'footwear', 'fruit', 'vegetable', 'prepared food', 'beverage', 'appliance', 'utensil', 'decorative item','textile', 'hand tool', 'power tool', 'kitchentool', 'personal electronic', 'home electronic', 'office electronic', 'land vehicle', 'watervehicle', 'air vehicle', 'string instrument','wind instrument', 'percussion instrument', 'firearm', 'container', 'toy', 'stationery', 'landscape', 'urban feature']

with open(r"your super entity path", 'r') as f:
    vg = json.load(f)


super_entities_text = "[" + ', '.join(vg_super_entities) + "]"

_PROMPT = """Task Description: You will be provided with a set of predicates related to specific actions, states, or relationships. Your task is to generate an appropriate superclass category name that effectively encapsulates the common characteristics of these predicates.

Input: You will receive the following set of predicates.

Output: Please provide a concise and specific superclass category name that encompasses all the given predicates. The superclass name should be between one to three words and should use general and easily understandable vocabulary.

Now, input predicates: {}
"""

api_key = "your api key"
client = OpenAI(api_key=api_key)

with open(r"./DATA/vg/vg_cate_info.json", 'r') as f:
    res = json.load(f)

ent_cate = res["ent_cate"]

total_step = {}
# idx =0
for entity in tqdm(ent_cate):
    system_prompt=_PROMPT.format(entity)

    messages = [
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": system_prompt},
                        ]
                    }
                ]

    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                top_p=1.0,
            )
            response = response.choices[0].message.content
            total_step[entity] = response
            # print(entity, response)
            # import ipdb; ipdb.set_trace()
            if response in vg_super_entities:
                break
        except Exception as e:
            print(datetime.now(), e.args)
            time.sleep(10)
    

save_url = "../DATASET/vg/vg_entiy_2_super_entity_check_final.json"
print("save sucessfully")
with open(save_url, "w") as file:
    json.dump(total_step, file, indent=4)

    
