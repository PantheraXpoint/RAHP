import torch
import json
from tqdm import tqdm
import os
import sys
import clip

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def generate_cache(json_path, output_path):
    """
    Generates and saves CLIP text embeddings from relation-aware prompts.
    Creates a list of tensors organized by predicate, matching the expected format.
    """
    print("=" * 70)
    print("CLIP Relation-Aware Weight Cache Generator")
    print("=" * 70)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Config] Device: {device}")
    print(f"[Config] Prompt JSON: {json_path}")
    print(f"[Config] Output Path: {output_path}")
    
    # --- Define predicates and super_entities ---
    predicates = [
        "above", "across", "against", "along", "and", "at", "attached to", "behind",
        "belonging to", "between", "carrying", "covered in", "covering", "eating",
        "flying in", "for", "from", "growing on", "hanging from", "has", "holding",
        "in", "in front of", "laying on", "looking at", "lying on", "made of",
        "mounted on", "near", "of", "on", "on back of", "over", "painted on",
        "parked on", "part of", "playing", "riding", "says", "sitting on",
        "standing on", "to", "under", "using", "walking in", "walking on",
        "watching", "wearing", "wears", "with"
    ]
    
    super_entities = [
        'male', 'female', 'children', 'pets', 'wild animal', 'ground transport',
        'water transport', 'air transport', 'sports equipment', 'seating furniture',
        'decorative item', 'table', 'upper body clothing', 'lower body clothing',
        'footwear', 'accessory', 'fruit', 'vegetable', 'prepared food', 'beverage',
        'utensils', 'container', 'textile', 'landscape', 'urban feature', 'plant',
        'structure', 'household item', 'head part', 'limb and appendage'
    ]
    
    print(f"\n[1/4] Configuration:")
    print(f"✓ Loaded {len(predicates)} predicates")
    print(f"✓ Loaded {len(super_entities)} super entities")
    
    # Load prompts
    print(f"\n[2/4] Loading relation-aware prompts...")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Prompt JSON not found: {json_path}")
    
    with open(json_path, 'r') as f:
        prompts_data = json.load(f)
    print(f"✓ Loaded {len(prompts_data)} prompt entries")
    
    # Initialize CLIP
    print(f"\n[3/4] Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    print("✓ CLIP model loaded")
    
    # Generate embeddings
    print(f"\n[4/4] Generating embeddings...")
    total_combinations = len(predicates) * len(super_entities) * len(super_entities)
    print(f"Total combinations: {len(predicates)} predicates × {len(super_entities)}² entities = {total_combinations}")
    
    relation_aware_weight_list = []
    
    with torch.no_grad():
        for pred_idx, pred in enumerate(tqdm(predicates, desc="Processing predicates")):
            pred_embeddings = []
            
            for sub in super_entities:
                for obj in super_entities:
                    # Construct the key as it appears in the JSON
                    triplet_key = f"{sub}|{pred}|{obj}"
                    
                    if triplet_key in prompts_data:
                        prompt_text = prompts_data[triplet_key]
                        
                        # Parse the prompt text
                        if isinstance(prompt_text, str):
                            # Try to extract descriptions from string
                            import re
                            if '[' in prompt_text and '"' in prompt_text:
                                # Extract quoted strings from list format
                                matches = re.findall(r'"([^"]*)"', prompt_text)
                                if matches:
                                    descriptions = matches[:3]  # Take first 3
                                else:
                                    descriptions = [prompt_text]
                            else:
                                descriptions = [prompt_text]
                        elif isinstance(prompt_text, list):
                            descriptions = prompt_text[:3]  # Take first 3
                        else:
                            descriptions = [f"a photo of {sub} {pred} {obj}"]
                    else:
                        # Fallback for missing keys
                        descriptions = [f"a photo of {sub} {pred} {obj}"]
                    
                    # Tokenize and encode with CLIP
                    text_tokens = clip.tokenize(descriptions, truncate=True).to(device)
                    text_features = clip_model.encode_text(text_tokens)
                    
                    # Normalize
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Average if multiple descriptions
                    text_embedding = text_features.mean(dim=0)
                    text_embedding = text_embedding / text_embedding.norm()
                    
                    # Store on CPU to save GPU memory
                    pred_embeddings.append(text_embedding.cpu())
            
            # Stack all embeddings for this predicate
            pred_embedding_tensor = torch.stack(pred_embeddings)
            relation_aware_weight_list.append(pred_embedding_tensor)
    
    # Save
    print(f"\n[5/5] Saving cache...")
    print(f"✓ Generated {len(relation_aware_weight_list)} predicate tensors")
    print(f"✓ Shape per predicate: {relation_aware_weight_list[0].shape}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(relation_aware_weight_list, output_path)
    
    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"✓ Saved to: {output_path}")
    print(f"✓ File size: {file_size_mb:.2f} MB")
    
    print("\n" + "=" * 70)
    print("Cache generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Configuration
    PROMPT_JSON_PATH = "DATASET/VG150/relation_aware_prompt.json"
    OUTPUT_CACHE_PATH = "MODEL/relation_aware_weight_cache.pth"
    
    # Verify input file exists
    if not os.path.exists(PROMPT_JSON_PATH):
        print(f"ERROR: Prompt JSON not found at {PROMPT_JSON_PATH}")
        print("Please generate it first using relation_aware_prompt_generation.py")
        sys.exit(1)
    
    # Generate cache
    generate_cache(PROMPT_JSON_PATH, OUTPUT_CACHE_PATH)