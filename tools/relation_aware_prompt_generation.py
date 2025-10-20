# import openai
import torch
import json
from tqdm import tqdm
from datetime import datetime
import time
import os
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

# openai.api_key = "your api key"

# with open(r"../DATASET/vg/vg_entiy_2_super_entity_check_final.json", 'r') as f:
#     super_entity = json.load(f)

# Define file paths
CHECKPOINT_FILE = 'DATASET/VG150/relation_aware_prompt_temp.json'
FINAL_OUTPUT_FILE = 'DATASET/VG150/relation_aware_prompt.json'

# --- Load existing data from checkpoint if it exists ---
generated_description = {}
if os.path.exists(CHECKPOINT_FILE):
    print(f"🔄 Loading existing results from checkpoint: {CHECKPOINT_FILE}")
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            generated_description = json.load(f)
        print(f"✓ Loaded {len(generated_description)} existing results.")
    except json.JSONDecodeError:
        print(f"⚠️ Warning: Checkpoint file {CHECKPOINT_FILE} is corrupted. Starting from scratch.")
        generated_description = {}
    except Exception as e:
        print(f"⚠️ Warning: Could not load checkpoint file {CHECKPOINT_FILE}. Error: {e}. Starting from scratch.")
        generated_description = {}
else:
    print("🏁 No checkpoint file found. Starting fresh.")


# --- Define predicates and super_entities ---
predicates = ["above", "across", "against", "along", "and", "at", "attached to", "behind", "belonging to", "between", "carrying", "covered in", "covering", "eating", "flying in", "for", "from", "growing on", "hanging from", "has", "holding", "in", "in front of", "laying on", "looking at", "lying on", "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on", "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on", "to", "under", "using", "walking in", "walking on", "watching", "wearing", "wears", "with"]
super_entity = [
    'male', 'female', 'children', 'pets', 'wild animal', 'ground transport',
    'water transport', 'air transport', 'sports equipment', 'seating furniture',
    'decorative item', 'table', 'upper body clothing', 'lower body clothing',
    'footwear', 'accessory', 'fruit', 'vegetable', 'prepared food', 'beverage',
    'utensils', 'container', 'textile', 'landscape', 'urban feature', 'plant',
    'structure', 'household item', 'head part', 'limb and appendage'
]

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

print("="*60)
print("STEP 1: Generate all prompts")
print("="*60)

# Generate prompts ONLY for triplets not already in the checkpoint
prompts_to_process = []
triplets_to_process = []
total_combinations_overall = 0

print("Filtering prompts based on checkpoint...")
for sub in tqdm(super_entity, desc="Subjects"):
    for obj in super_entity:
        for pred in predicates:
            total_combinations_overall += 1
            triplet = f"{sub}_{pred}_{obj}"
            # Check if this triplet is already done or marked as failed
            if triplet not in generated_description or generated_description[triplet] == "GENERATION_FAILED":
                 prompt = PROMPT.format(sub, obj, pred)
                 prompts_to_process.append(prompt)
                 triplets_to_process.append(triplet)

total_to_process = len(prompts_to_process)
print(f"✓ Total combinations: {total_combinations_overall}")
print(f"✓ Already processed: {len(generated_description)}")
print(f"✓ Remaining prompts to process: {total_to_process}\n")

if total_to_process == 0:
    print("✅ All prompts already processed! Saving final results.")
    # Save the final results (which are just the loaded checkpoint data)
    os.makedirs(os.path.dirname(FINAL_OUTPUT_FILE), exist_ok=True)
    with open(FINAL_OUTPUT_FILE, 'w') as f:
        json.dump(generated_description, f, indent=2)
    print(f"Output saved to: {FINAL_OUTPUT_FILE}")
    exit() # Exit the script early

print("="*60)
print("STEP 2: Initialize LMDeploy pipeline")
print("="*60)

model_type = "Qwen/Qwen2.5-14B-Instruct-AWQ"  # Change this to your model path

# Configuration
tp = 1  # Number of GPUs for tensor parallelism
batch_size = 512  # Larger batch size since we're doing pure batch processing

print(f"Model path: {model_type}")
print(f"Batch size: {batch_size}")
print(f"Tensor parallelism: {tp}")

pipe = pipeline(
    model_type,
    backend_config=TurbomindEngineConfig(
        session_len=8192*4, 
        tp=tp, 
        cache_max_entry_count=0.3
    )
)

# Generation config
gen_config = GenerationConfig(
    do_sample=True, 
    max_new_tokens=1024, 
    temperature=0.5
)

print("✓ Pipeline initialized successfully!\n")

print("="*60)
print("STEP 3: Batch inference")
print("="*60)

newly_processed = 0
errors = 0
num_batches = (total_to_process + batch_size - 1) // batch_size
start_point_total_processed = len(generated_description) # Count from where we left off

print(f"Processing {total_to_process} remaining prompts in {num_batches} batches...")

# Process the filtered prompts in batches
for batch_idx in tqdm(range(0, total_to_process, batch_size), desc="Batches"):
    batch_end = min(batch_idx + batch_size, total_to_process)
    current_batch_prompts = prompts_to_process[batch_idx:batch_end]
    current_batch_triplets = triplets_to_process[batch_idx:batch_end]

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Batch inference
            responses = pipe(current_batch_prompts, gen_config=gen_config)

            # Store responses
            for triplet, response in zip(current_batch_triplets, responses):
                generated_description[triplet] = response.text # Overwrites "GENERATION_FAILED" if retry successful
                newly_processed += 1

            break  # Success, exit retry loop

        except Exception as e:
            retry_count += 1
            print(f"\n✗ ERROR on batch {batch_idx//batch_size + 1} (attempt {retry_count}/{max_retries}):")
            print(f"  {str(e)[:200]}")

            if retry_count < max_retries:
                wait_time = 5 * retry_count
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  FAILED after {max_retries} attempts, marking batch as failed...")
                for triplet in current_batch_triplets:
                    # Only mark as failed if it wasn't already processed successfully before
                    if triplet not in generated_description or generated_description[triplet] == "GENERATION_FAILED":
                        generated_description[triplet] = "GENERATION_FAILED"
                batch_errors = len(current_batch_triplets)
                errors += batch_errors
                newly_processed += batch_errors # Count errors as processed for progress tracking
    
    # Save checkpoint every 1000 *newly* processed items
    # Checkpoint saving condition is based on newly processed count
    # Check if newly_processed crosses a 1000 boundary since the last save
    if (newly_processed // 1000) > ((newly_processed - len(current_batch_prompts)) // 1000):
        os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)
        try:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(generated_description, f, indent=2)
            # Display total progress (old + new)
            print(f"  ✓ Checkpoint saved: {start_point_total_processed + newly_processed}/{total_combinations_overall}")
        except Exception as write_e:
            print(f"  ✗ FAILED to save checkpoint: {write_e}")


print("\n" + "="*60)
print("STEP 4: Save final results")
print("="*60)

# Save final complete results
os.makedirs(os.path.dirname(FINAL_OUTPUT_FILE), exist_ok=True)
try:
    with open(FINAL_OUTPUT_FILE, 'w') as f:
        json.dump(generated_description, f, indent=2)
    print(f"Final output saved to: {FINAL_OUTPUT_FILE}")
    # Optionally remove the temp file after successful final save
    # if os.path.exists(CHECKPOINT_FILE):
    #     os.remove(CHECKPOINT_FILE)
except Exception as final_write_e:
    print(f"  ✗ FAILED to save final output: {final_write_e}")
    print(f"  Intermediate results might still be available in: {CHECKPOINT_FILE}")

print(f"\n{'='*60}")
print("✓ Generation Complete!")
print(f"{'='*60}")
final_processed_count = len([k for k,v in generated_description.items() if v != "GENERATION_FAILED"])
final_error_count = len([k for k,v in generated_description.items() if v == "GENERATION_FAILED"])
print(f"Total successful generations: {final_processed_count}/{total_combinations_overall}")
print(f"Total errors/failed generations: {final_error_count}")
if total_combinations_overall > 0:
    print(f"Success rate: {(final_processed_count / total_combinations_overall) * 100:.2f}%")
print(f"{'='*60}")