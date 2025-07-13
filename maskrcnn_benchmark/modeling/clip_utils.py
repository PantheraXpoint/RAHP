# Modified from [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)

import os
from tempfile import template

import torch
import torch.nn as nn
from clip import clip



def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


single_template = ["a photo of a {}."]

reduced_templates = [
    "There is {article} {} in the scene.",
    # "There is the {} in the scene.",
    "A photo of {article} {} in the scene.",
    # "a photo of the {} in the scene.",
    # "a photo of one {} in the scene.",
]

multiple_templates = [
    "There is {article} {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {article} {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of {article} {}.",
    "itap of my {}.",  # itap: I took a picture of
    "itap of the {}.",
    "a photo of {article} {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {article} {}.",
    "a good photo of the {}.",
    "a bad photo of {article} {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {article} {}.",
    "a bright photo of the {}.",
    "a dark photo of {article} {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {article} {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {article} {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {article} {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {article} {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {article} {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {article} {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {article} {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]


def load_clip_to_cpu(visual_backbone):
    backbone_name = visual_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    model = model.cpu()
    return model


class TextEncoder(nn.Module):
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

run_on_gpu = torch.cuda.is_available()
clip_model = load_clip_to_cpu("ViT-B/32")
text_model = TextEncoder(clip_model)



def build_one_text_embedding(category, templates=multiple_templates):
    run_on_gpu = torch.cuda.is_available()
    global text_model

    if run_on_gpu:
        text_model = text_model.to(torch.device(type='cuda'))
    texts = [
        template.format(processed_name(category, rm_dot=True), article=article(category))
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

def build_batch_text_embedding(txts, templates=multiple_templates):
    global text_model
    if run_on_gpu:
        text_model = text_model.to(torch.device(type='cuda'))
    texts_toked = []
    for category in txts:
        texts = [
            template.format(processed_name(category, rm_dot=True), article=article(category))
            for template in templates
        ]
        texts_word = [
            "This is " + text if text.startswith("a") or text.startswith("the") else text
            for text in texts
        ]
        
    texts = clip.tokenize(texts_word).long()  # tokenize
    texts_toked.append(texts)
    
    if run_on_gpu:
        texts = texts.cuda()
    text_embeddings = text_model(texts)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    text_embedding = text_embeddings.mean(dim=0)
    text_embedding /= text_embedding.norm()

    return text_embedding, texts_word



def build_text_embedding(categories, templates=None):
    run_on_gpu = torch.cuda.is_available()
    clip_model = load_clip_to_cpu("ViT-B/32")
    text_model = TextEncoder(clip_model)
    if run_on_gpu:
        text_model = text_model.cuda()

    if templates is None:
        templates = multiple_templates
    for _, param in text_model.named_parameters():
        param.requires_grad = False
    with torch.no_grad():
        zeroshot_weights = []
        for category in categories:
            text_embedding, texts_word = build_one_text_embedding(category, templates=templates)
            zeroshot_weights.append(text_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
        if run_on_gpu:
            zeroshot_weights = zeroshot_weights.cuda()
    zeroshot_weights = zeroshot_weights.t()
    return zeroshot_weights


