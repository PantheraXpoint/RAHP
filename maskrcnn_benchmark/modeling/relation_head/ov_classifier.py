import pickle
import os
import random
import copy
import json
import sys
import time

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
from tqdm import tqdm

from ..clip_utils import build_one_text_embedding, build_text_embedding, reduced_templates

# Following "Towards Open-vocabulary Scene Graph Generation with Prompt-based Finetuning"
VG150_BASE_OBJ_CATEGORIES = {
    'human': ['boy', 'girl', 'child', 'guy', 'lady', 'man', 'men', 'person', 'people', 'woman', 'kid', 'player', 'skier', 'play', 'skate', 'surf'],
    'animal': ['bear', 'bird', 'cat', 'cow', 'dog', 'elephant', 'giraffe', 'horse', 'sheep', 'zebra', 'animal'],
    'transportation': ['airplane', 'bike', 'boat', 'bus', 'car', 'engine', 'motorbike', 'motorcycle', 'plane', 'skateboard', 'ski', 'surfboard', 'train', 'truck', 'vehicle', 'wheel', 'wing', 'windscreen', 'seat', 'tire', 'windshield'],
    'furniture': ['bed', 'bench', 'chair', 'counter', 'desk', 'drawer', 'handle', 'lamp', 'pillow', 'table', 'cabinet', 'clock', 'curtain', 'shelf', 'light', 'sink', 'toilet'],
    'clothing': ['cap', 'coat', 'glove', 'hat', 'jacket', 'jean', 'pant', 'shirt', 'shoe', 'short', 'sneaker', 'sock', 'tie', 'boot'],
    'food': ['banana', 'fruit', 'pizza', 'glass', 'food', 'orange', 'vegetable'],
    'product': ['bottle', 'bowl', 'fork', 'knife', 'plate', 'pot', 'bag', 'towel', 'umbrella', 'vase', 'tile', 'screen', 'box', 'logo', 'number', 'basket', 'book', 'helmet', 'kite', 'laptop', 'letter', 'paper', 'phone', 'racket', 'seat', 'cup', 'flag'],
    'outdoor': ['beach', 'branch', 'fence', 'flower', 'hill', 'leaf', 'mountain', 'plant', 'rock', 'snow', 'street', 'tree', 'wave', 'railing', 'sidewalk'],
    'buildings': ['board', 'building', 'door', 'roof', 'room', 'sign', 'stand', 'tower', 'trunk', 'wire', 'pole', 'post', 'track', 'house', 'window'],
    'body Part': ['ear', 'eye', 'face', 'hair', 'hand', 'head', 'leg', 'mouth', 'neck', 'nose', 'paw', 'tail', 'arm', 'finger']
}
BASIC_OBJ_CLASS = list(VG150_BASE_OBJ_CATEGORIES.keys())
# TOTAL_TRIPLET = torch.load("/storage/data/v-liutao/pesudo_label/union_description_after_gpt_filter.pt")

RAHP_SUPER_ENTITIES = [
    'male', 'female', 'children', 'pets', 'wild animal', 'ground transport',
    'water transport', 'air transport', 'sports equipment', 'seating furniture',
    'decorative item', 'table', 'upper body clothing', 'lower body clothing',
    'footwear', 'accessory', 'fruit', 'vegetable', 'prepared food', 'beverage',
    'utensils', 'container', 'textile', 'landscape', 'urban feature', 'plant',
    'structure', 'household item', 'head part', 'limb and appendage'
]


def load_categorical_clip_text_embedding(dataset_name):

    if dataset_name == 'VG':
        with open('/RAHP/DATA/vg/vg_cate_info.json', 'r') as f:
            cate_info = json.load(f)
            obj_classes = cate_info["ent_cate"][:-1]
            rel_classes = cate_info["pred_cate"][1:]
            # use predicate background 
            # rel_classes = cate_info["pred_cate"]
            # zeroshot_w = build_text_embedding(obj_classes, templates=reduced_templates)
            # zeroshot_w = zeroshot_w.cpu()


    elif dataset_name == 'GQA':
        with open('/mnt/petrelfs/lirongjie/project/cvpods/datasets/gqa/annotations/gqa_cate_metainfo.json', 'r') as f:
            cate_info = json.load(f)
        obj_classes = cate_info['ind_to_classes'][:-1]
        rel_classes = cate_info["ind_to_predicates"][1:]
        zeroshot_w = build_text_embedding(obj_classes, templates=reduced_templates)
        zeroshot_w = zeroshot_w.cpu()

    elif dataset_name == 'PSG':
        with open('/mnt/petrelfs/lirongjie/project/cvpods/datasets/psg/categories_dict.json', 'r') as f:
            cate_info = json.load(f)
        obj_classes = cate_info['obj']
        rel_classes = cate_info["rel"]
        zeroshot_w = build_text_embedding(obj_classes, templates=reduced_templates)
        zeroshot_w = zeroshot_w.cpu()

    elif dataset_name == 'OIV6':
        with open('/mnt/petrelfs/lirongjie/project/cvpods/datasets/openimages/open-imagev6/annotations/categories_dict.json', 'r') as f:
            cate_info = json.load(f)
        obj_classes = cate_info['obj']
        rel_classes = cate_info["rel"]
        zeroshot_w = build_text_embedding(obj_classes, templates=reduced_templates)
        zeroshot_w = zeroshot_w.cpu()

    return obj_classes, rel_classes

class CLIPDynamicClassifierBaseline(nn.Module):
    def __init__(self, cfg, input_dim, clip_model, clip_preprocess):
        super(CLIPDynamicClassifierBaseline, self).__init__()
        
        self.cfg = cfg
        obj_classes, rel_classes = load_categorical_clip_text_embedding(cfg.MODEL.DYHEAD.OV.DATASET_NAME)

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.obj_classes = obj_classes
        self.rel_classes =  rel_classes

        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=False)

        self.text_encoder = TextEncoder(clip_model)
        weight_list = []
        with torch.no_grad():
            print('build clip text emb tmp')
            for pred_txt in tqdm(self.rel_classes): # N_p
                weight_list_rel = []
                obj_classes = self.obj_classes

                for ent_text in obj_classes: # N_p
                    trp_templete_w, trp_templete_txt  = build_baseline_text_embedding(
                    f"{ent_text} {pred_txt} something", 
                        templates=simple_reduced_templates, text_model=self.text_encoder
                    )
                    trp_templete_w = trp_templete_w / trp_templete_w.norm(dim=-1, keepdim=True)
                    weight_list_rel.append(trp_templete_w)
                # trp_templete_w_collect = torch.cat(trp_templete_w_collect, dim=0)
                weight_list.append(torch.stack(weight_list_rel))


        self.classifier_cache = nn.ParameterDict()
        self.classifier_cache['cls_weight'] = nn.Parameter(torch.stack(weight_list), requires_grad=False) # Num_pred * num_ent^2 * hdim

        if self.cfg.MODEL.DYHEAD.OV.ENABLED:
            seen_cls_weight = []
            for cls_i in range(len(self.rel_classes)):
                if cls_i not in self.cfg.MODEL.DYHEAD.OV.ZS_PREDICATES:
                    seen_cls_weight.append(weight_list[cls_i])
            seen_cls_weight = torch.stack(seen_cls_weight)
            # print(f'seen class weight {seen_cls_weight.shape}')
            self.classifier_cache['cls_weight_train'] = nn.Parameter(seen_cls_weight, requires_grad=False)

    
    def forward(self, rel_hs):
        # assemble the triplets embedding

        all_entity_pairs = list()

        # classification with embedding by matching
        inter_hs = rel_hs

        cate_weights = self.classifier_cache['cls_weight']
        if self.cfg.MODEL.DYHEAD.OV.ENABLED:
            if self.training:
                cate_weights = self.classifier_cache['cls_weight_train']
            else:
                cate_weights = self.classifier_cache['cls_weight']

        # cosine distance
        inter_hs_norm = inter_hs / inter_hs.norm(dim=-1, keepdim=True)
        cate_weights = cate_weights / cate_weights.norm(dim=-1, keepdim=True)

        dis_mat = (inter_hs_norm @ cate_weights.permute(2, 0, 1).reshape(inter_hs_norm.shape[-1], -1)) # num_inst,dim x Num_pred, num_ent, dim => num_inst, Num_pred, num_ent
        dis_mat = dis_mat * self.logit_scale.exp()

        class_num = cate_weights.shape[0]
        dis_mat = dis_mat.reshape((dis_mat.shape[0], class_num, -1))
        cls_res = dis_mat.max(-1)[0] # num_inst, Num_pred

        return cls_res

class CLIPDynamicClassifierSimple(nn.Module):
    def __init__(self, cfg, input_dim, clip_model, clip_preprocess):
        super(CLIPDynamicClassifierSimple, self).__init__()
        
        self.cfg = cfg
        obj_classes, rel_classes = load_categorical_clip_text_embedding(cfg.MODEL.DYHEAD.OV.DATASET_NAME)

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes

        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=False)

        self.tree_type = ''
        self.mask = torch.zeros((50,10,10))
        
        self.text_encoder = TextEncoder(clip_model)
        
        # Paper's 30 Super Entity Categories (Section B.2, AAAI 2025)
        RAHP_SUPER_ENTITIES = [
            'male', 'female', 'children', 'pets', 'wild animal',
            'ground transport', 'water transport', 'air transport',
            'sports equipment', 'seating furniture', 'decorative item',
            'table', 'upper body clothing', 'lower body clothing',
            'footwear', 'accessory', 'fruit', 'vegetable', 'prepared food',
            'beverage', 'utensils', 'container', 'textile', 'landscape',
            'urban feature', 'plant', 'structure', 'household item',
            'head part', 'limb and appendage'
        ]
        
        # Initialize split index lists BEFORE using them
        self.relation_aware_split_index = []
        self.relation_aware_split_index_train = []
        self.leaf_split_index = []
        self.leaf_split_index_train = []
        self.classifier_cache = {}
        
        # Generate entity-aware weights
        entity_aware_weight_list = []
        with torch.no_grad():
            print('Building entity-aware CLIP text embeddings...')
            for pred_txt in tqdm(self.rel_classes, desc="Entity-aware prompts"):
                weight_list_rel = []
                obj_classes = RAHP_SUPER_ENTITIES  # Use paper's 30 categories

                for sub_text in obj_classes:
                    for obj_text in obj_classes:
                        trp_templete_w, trp_templete_txt = build_baseline_text_embedding(
                            f"{sub_text} {pred_txt} {obj_text}", 
                            templates=simple_reduced_templates, text_model=self.text_encoder
                        )
                        trp_templete_w = trp_templete_w / trp_templete_w.norm(dim=-1, keepdim=True)
                        weight_list_rel.append(trp_templete_w)
                entity_aware_weight_list.append(torch.stack(weight_list_rel))

        # Load relation-aware weights from cache
        if cfg.MODEL.DYHEAD.OV.DYNAMIC_CLIP_CLASSIFIER_WEIGHT_CACHE_PTH != '':
            print(f'Loading relation-aware weights from cache: {cfg.MODEL.DYHEAD.OV.DYNAMIC_CLIP_CLASSIFIER_WEIGHT_CACHE_PTH}')
            relation_aware_weight_list = torch.load(cfg.MODEL.DYHEAD.OV.DYNAMIC_CLIP_CLASSIFIER_WEIGHT_CACHE_PTH)
            
            # Verify cache structure
            if not isinstance(relation_aware_weight_list, list) or len(relation_aware_weight_list) != len(self.rel_classes):
                raise ValueError(f"Cache should be a list of {len(self.rel_classes)} tensors, got {type(relation_aware_weight_list)} with length {len(relation_aware_weight_list) if isinstance(relation_aware_weight_list, list) else 'N/A'}")
            
            print(f'✓ Cache loaded: {len(relation_aware_weight_list)} predicates')
            print(f'✓ First predicate shape: {relation_aware_weight_list[0].shape}')
            print(f'✓ Expected shape: [900, 512] for 30 super-entities')
            
            # Build split indices from cache
            for num_index, pred_weights in enumerate(relation_aware_weight_list):
                num_entity_pairs = pred_weights.shape[0]  # Should be 900 (30×30)
                self.relation_aware_split_index.append(num_entity_pairs)
        else:
            raise ValueError("DYNAMIC_CLIP_CLASSIFIER_WEIGHT_CACHE_PTH cannot be empty")

        # Store weights in classifier cache
        self.classifier_cache = nn.ParameterDict()
        self.classifier_cache['entity_aware_weight'] = nn.Parameter(
            torch.stack(entity_aware_weight_list), requires_grad=False
        )  # Shape: [50, 900, 512]
        self.classifier_cache['relation_aware_weight'] = nn.Parameter(
            torch.cat(relation_aware_weight_list, dim=0), requires_grad=False
        )  # Shape: [45000, 512] = 50 * 900

        print(f'✓ Entity-aware weight shape: {self.classifier_cache["entity_aware_weight"].shape}')
        print(f'✓ Relation-aware weight shape: {self.classifier_cache["relation_aware_weight"].shape}')

        # Build training splits (exclude zero-shot predicates)
        if self.cfg.MODEL.DYHEAD.OV.ENABLED:
            entity_aware_weight_list_train = []
            relation_aware_weight_list_train = []

            num_index = 0
            for cls_i in range(len(self.rel_classes)):
                if cls_i not in self.cfg.MODEL.DYHEAD.OV.ZS_PREDICATES:
                    # Include this predicate in training
                    entity_aware_weight_list_train.append(entity_aware_weight_list[cls_i])
                    relation_aware_weight_list_train.append(relation_aware_weight_list[cls_i])
                    self.relation_aware_split_index_train.append(self.relation_aware_split_index[cls_i])

            self.classifier_cache['entity_aware_weight_train'] = nn.Parameter(
                torch.stack(entity_aware_weight_list_train), requires_grad=False
            )
            self.classifier_cache['relation_aware_weight_train'] = nn.Parameter(
                torch.cat(relation_aware_weight_list_train, dim=0), requires_grad=False
            )
            
            print(f'✓ Training splits created:')
            print(f'  - Entity-aware train shape: {self.classifier_cache["entity_aware_weight_train"].shape}')
            print(f'  - Relation-aware train shape: {self.classifier_cache["relation_aware_weight_train"].shape}')
            print(f'  - Zero-shot predicates excluded: {len(self.cfg.MODEL.DYHEAD.OV.ZS_PREDICATES)}')

    
    def forward(self, rel_hs, union_features=None, tree_features=None):
        # classification with embedding by matching
        inter_hs = rel_hs
        # cosine distance
        inter_hs_norm = inter_hs / inter_hs.norm(dim=-1, keepdim=True)

        entity_aware_weight = self.classifier_cache['entity_aware_weight'].to(inter_hs.device)
        relation_aware_weight = self.classifier_cache['relation_aware_weight'].to(inter_hs.device)
        cate_split_index = self.leaf_split_index
        if self.cfg.MODEL.DYHEAD.OV.ENABLED:
            if self.training:
                entity_aware_weight = self.classifier_cache['entity_aware_weight_train'].to(inter_hs.device)
                relation_aware_weight = self.classifier_cache['relation_aware_weight_train'].to(inter_hs.device)
                cate_split_index = self.leaf_split_index_train
            else:
                entity_aware_weight = self.classifier_cache['entity_aware_weight'].to(inter_hs.device)
                relation_aware_weight = self.classifier_cache['relation_aware_weight'].to(inter_hs.device)
                cate_split_index = self.leaf_split_index

        # entity-aware prompt scpre
        entity_aware_weight = entity_aware_weight / entity_aware_weight.norm(dim=-1, keepdim=True)
        dis_mat_entity = (inter_hs_norm @ entity_aware_weight.permute(2, 0, 1).reshape(inter_hs_norm.shape[-1], -1)) # num_inst,dim x Num_pred, num_ent, dim => num_inst, Num_pred, num_ent
        class_num = entity_aware_weight.shape[0]
        dis_mat_entity = dis_mat_entity.reshape((dis_mat_entity.shape[0], class_num, -1))

        # relation-aware prompt scpre
        union_features_norm = union_features / union_features.norm(dim=-1, keepdim=True)
        relation_aware_weight = relation_aware_weight / relation_aware_weight.norm(dim=-1, keepdim=True)
        dis_mat_relation, _ = self.compute_scores(union_features_norm, relation_aware_weight, cate_split_index, class_num, self.cfg.MODEL.DYHEAD.OV.PROMPT_SELECT_K)

        # final predicate score
        cls_res = dis_mat_masked.max(-1)[0] * (1 - self.cfg.MODEL.DYHEAD.OV.VLM_BIAS_WEIGHT) + dis_mat_union.max(-1)[0] * self.cfg.MODEL.DYHEAD.OV.VLM_BIAS_WEIGHT# num_inst, Num_pred

        return cls_res
    
    def compute_scores(self, inter_hs, cate_weights_tree, cate_leaf_split_index, class_num, selct_top_k=3):
        dis_mat_tree = (inter_hs_norm @ cate_weights_tree.permute(1, 0).reshape(inter_hs_norm.shape[-1], -1)) # num_inst,dim x Num_pred, num_ent, dim => num_inst, Num_pred, num_ent
        dis_mat_tree = dis_mat_tree * self.logit_scale.exp()

        dis_mat_leaf = torch.split(dis_mat_tree, cate_leaf_split_index, dim=-1)
        
        # top K
        topK = selct_top_k
        dis_mat_leaf_max = []
        for leaf in dis_mat_leaf:
            if leaf.shape[-1] >= topK:
                dis_mat_leaf_max.append(torch.topk(leaf,k=topK,dim=-1,largest=True)[0].mean(-1))
            else:
                dis_mat_leaf_max.append(leaf.mean(-1))

        dis_mat_leaf_max = torch.stack(dis_mat_leaf_max, dim=-1)
        dis_mat_tree_sum = dis_mat_leaf_max.reshape(dis_mat_leaf_max.shape[0],class_num,-1)

        return dis_mat_tree_sum, dis_mat_tree
    

# TBD
class BLIPDynamicClassifierSimple(nn.Module):
    def __init__(self, cfg, input_dim, clip_model, clip_preprocess):
        super(BLIPDynamicClassifierSimple, self).__init__()
        
        self.cfg = cfg
        obj_classes, rel_classes, obj_text_embed = load_categorical_clip_text_embedding(cfg.MODEL.DYHEAD.OV.DATASET_NAME)

        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.obj_text_embed = obj_text_embed # N x dim
        self.obj_classes = obj_classes
        self.rel_classes =  rel_classes

        self.cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=False)

        self.tree_type = ''


    

from torch.nn.init import kaiming_uniform_
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            # nn.LayerNorm(c_in // reduction),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
        
        for linear in self.fc.modules():
            if isinstance(linear, nn.Linear):
                kaiming_uniform_(linear.weight, a=0, nonlinearity='relu')

    def forward(self, x):
        x = self.fc(x)
        return x


def union_pixel(box1, box2, w, h):
    # 寻找左上角和右下角的最小值和最大值
    combined_left = torch.clamp(torch.minimum(box1[:, 0], box2[:, 0]), min=0, max=1)
    combined_top = torch.clamp(torch.minimum(box1[:, 1], box2[:, 1]), min=0, max=1)
    combined_right = torch.clamp(torch.maximum(box1[:, 2], box2[:, 2]), min=0, max=1)
    combined_bottom = torch.clamp(torch.maximum(box1[:, 3], box2[:, 3]), min=0, max=1)

    # 构建并集坐标
    combined_box = torch.cat([combined_left.unsqueeze(-1),
                                combined_top.unsqueeze(-1),
                                combined_right.unsqueeze(-1),
                                combined_bottom.unsqueeze(-1)], dim=-1)
    
    return combined_box

def sigmoid_focal_loss(
        inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
            
@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # import ipdb; ipdb.set_trace()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def standardization(features):
    mean = torch.mean(features)
    std = torch.std(features)
    features = (features - mean) / std

    return features

from clip import clip

def article(name):
    return "an" if name[0] in "aeiou" else "a"

baseline_templates = [
    "A photo of {article1} {sub_text} and {article2} {obj_text}."
]

simple_reduced_templates = [
    "There is {article} {} in the scene.",
    # "There is the {} in the scene.",
    "A photo of {article} {} in the scene.",
    # "a photo of the {} in the scene.",
    # "a photo of one {} in the scene.",
]

def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res

def build_baseline_text_embedding(category, templates=simple_reduced_templates, text_model=None):
    """
    Build text embeddings using CLIP text encoder
    Args:
        text: The text to embed (e.g., "male riding ground transport")
        templates: List of prompt templates
        text_model: TextEncoder instance
    """
    from clip import clip
    
    # Apply templates
    texts = [template.format(category) for template in templates]
    
    # Tokenize texts
    tokenized_texts = clip.tokenize(texts).cuda()
    
    # Get embeddings from text encoder
    # TextEncoder.forward expects (prompts, tokenized_prompts)
    with torch.no_grad():
        # Create prompt embeddings (normally this would be learned prompts, but we use token embeddings)
        prompt_embeddings = text_model.transformer.token_embedding(tokenized_texts).type(text_model.dtype)
        
        # Call forward with both arguments
        text_embeddings = text_model(prompt_embeddings, tokenized_texts)
    
    # Average across templates
    text_embeddings = text_embeddings.mean(dim=0, keepdim=True)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    
    return text_embeddings, texts

def build_simple_text_embedding(category, templates=baseline_templates, text_model=None):
    run_on_gpu = torch.cuda.is_available()

    if run_on_gpu:
        text_model = text_model.to(torch.device(type='cuda'))
    texts = [
        # template.format(processed_name(category, rm_dot=True), article=article(category))
        template.format(sub_text=category[0], obj_text=category[1], article1=article(category[0]), article2=article(category[1])) 
        for template in templates
    ]

    texts = clip.tokenize(texts).long()  # tokenize
    if run_on_gpu:
        texts = texts.cuda()
        
    text_embeddings = text_model(texts)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    text_embedding = text_embeddings.mean(dim=0)
    text_embedding /= text_embedding.norm()
    text_embedding = text_embedding.cpu()

    return text_embedding, texts

class TextEncoder_2(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        clip_model = clip_model.float()
        self.dtype = clip_model.dtype

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x



from torch.nn.modules.utils import _pair
from functools import reduce, partial

class PatchPrompter(nn.Module):
    def __init__(self, prompt_size, patch_size, image_size):
        super(PatchPrompter, self).__init__()
        self.patch_size = patch_size
        self.prompt_size = prompt_size
        self.fg_size = self.patch_size - prompt_size * 2

        self.patch = nn.Parameter(torch.randn([1, 3, image_size, image_size]))

    def forward(self, x):
        _, _, h, w = x.size()

        fg_in_patch = torch.zeros([1, 3, self.fg_size, self.fg_size]).cuda()
        fg_in_patch = F.pad(fg_in_patch, (self.prompt_size, self.prompt_size, self.prompt_size, self.prompt_size), "constant", 1)
        mask = fg_in_patch.repeat(1, 1, h//self.patch_size, w//self.patch_size)
        self.prompt = self.patch * mask

        return x + self.prompt


class SharedPrompter(nn.Module):
    def __init__(self, prompt_size, patch_size):
        super(SharedPrompter, self).__init__()
        self.patch_size = patch_size
        self.prompt_size = prompt_size
        self.fg_size = self.patch_size - prompt_size * 2

        self.patch = nn.Parameter(torch.randn([1, 3, self.patch_size, self.patch_size]))

    def forward(self, x):
        _, _, h, w = x.size()

        fg_in_patch = torch.zeros([1, 3, self.fg_size, self.fg_size]).cuda()
        fg_in_patch = F.pad(fg_in_patch, (self.prompt_size, self.prompt_size, self.prompt_size, self.prompt_size), "constant", 1)
        
        mask = fg_in_patch.repeat(1, 1, h//self.patch_size, w//self.patch_size)
        patch = self.patch.repeat(1, 1, h//self.patch_size, w//self.patch_size)

        self.prompt = patch * mask

        return x + self.prompt
        

class PadPrompter(nn.Module):
    def __init__(self, prompt_size, image_size):
        super(PadPrompter, self).__init__()
        pad_size = prompt_size
        image_size = image_size

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        # self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        # self.last_layer.weight_g.data.fill_(1)
        # if norm_last_layer:
        #     self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_proj = self.mlp(x)
        return x_proj
    

from collections import OrderedDict
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = x.half()
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class PromptLearner_v2(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 8
        ctx_init = ""
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.cfg = cfg

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

        classnames = [name.replace("_", "") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts_training = []
        for i, name in enumerate(classnames):
            if i-1 not in cfg.MODEL.DYHEAD.OV.ZS_PREDICATES:
                prompts_training.append(prompt_prefix + " " + name + ".")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        n_cls = self.n_cls
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        return prompts

class PromptLearner_v1(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 8
        ctx_init = ""
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                prompt = prompt.to(device)
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim,)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = ["a photo of " + " " + prompt_prefix + " " + "in the scene." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            tokenized_prompts = tokenized_prompts.to(device)
            embedding = clip_model.token_embedding(tokenized_prompts)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS


        self.cfg = cfg
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = 'end'


    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        n_cls = self.n_cls

        ctx = self.ctx
        # if ctx.dim() == 2:
        #     ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError
        return prompts
