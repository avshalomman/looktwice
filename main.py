# !pip install qwen-vl-utils
# !pip install -U transformers
# !pip install open-clip-torch
import argparse
import logging
from pathlib import Path
from typing import Optional
import pandas as pd
import os
import pickle
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, LogitsProcessorList
from cd import EnsembleLogitsProcessor
from qwen_vl_utils import process_vision_info
from collections import OrderedDict
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
CAPTIONING_PROMPT = "Describe this image shortly."
hf_token = None

def set_hf_token(token: str):
    global hf_token
    hf_token = token

def set_device(device_str: str):
    global device
    device = device_str

def setup_logging():
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

def prepare_naturalbench(indices: pd.Series, output_path: Path, template=None):
    import hashlib
    assert len(indices) > 0, "Indices must be non-empty"
    if template is None:
        template = {
            "yes_no": " Please answer Yes or No.",
            "multiple_choice": " Please output the letter corresponding to the correct option."
        } 
    path_dir = f"{output_path}/nb_processed"
    os.makedirs(path_dir, exist_ok=True)
    indices_hash = hashlib.md5(indices.to_numpy()).hexdigest()
    template_hash = hashlib.md5(str(template).encode()).hexdigest()
    filename = f"nb_{indices_hash}_{template_hash}.pkl"
    if os.path.exists(path_dir + '/' + filename):
        logging.info(f"Loading NaturalBench samples from {path_dir + '/' + filename}")
        with open(path_dir + '/' +  filename, "rb") as f:
            return pickle.load(f)
    logging.info("Preparing NaturalBench...")
    
    dataset = load_ds()

    naturalbench = []
    for item in dataset["train"]:
        if item['Index'] not in indices:
            continue
        naturalbench.append([item["Question_0"] + template[item["Question_Type"]],
                            item["Image_0"], item["Image_0_Question_0"], item['Question_Type']])
        naturalbench.append([item["Question_0"] + template[item["Question_Type"]],
                            item["Image_1"], item["Image_1_Question_0"], item['Question_Type']])
        naturalbench.append([item["Question_1"] + template[item["Question_Type"]],
                            item["Image_0"], item["Image_0_Question_1"], item['Question_Type']])
        naturalbench.append([item["Question_1"] + template[item["Question_Type"]],
                            item["Image_1"], item["Image_1_Question_1"], item['Question_Type']])
    nb = [{'question': x[0], 'image': x[1], 'answer': x[2], 'type': x[3]} for x in naturalbench]
    with open(path_dir + '/' + filename, "wb") as f:
        pickle.dump(nb, f)
    logging.info(f"Prepared {len(nb)} samples for NaturalBench")
    return nb


def load_ds(vqa_dataset: str = 'BaiqiL/NaturalBench'):
    from datasets import load_dataset
    ds = load_dataset(vqa_dataset, cache_dir='/dev/shm')
    return ds

def prepare_splits(vqa_dataset: str, output_path: Path,
                train_ratio: float = 0.6, val_ratio: float = 0.2):
    path_dir = f"{output_path}/splits/{vqa_dataset.replace('/', '_')}"
    os.makedirs(path_dir, exist_ok=True)
    filename = f"splits_{train_ratio}_{val_ratio}.pkl"
    if os.path.exists(path_dir + '/' +  filename):
        logging.info(f"Loading splits from {path_dir + '/' + filename}")
        with open(path_dir + '/' + filename, "rb") as f:
            return pickle.load(f)
    test_ratio = 1.0 - train_ratio - val_ratio
    logging.info(f"Preparing splits... train: {train_ratio}, "
                f"val: {val_ratio}, test: {test_ratio}")
    ds = load_ds(vqa_dataset)
    ds_length = len(ds['train']['Index'])
    train_len, dev_len, test_len = map(int, (ds_length * train_ratio,
                                            ds_length * val_ratio,
                                            ds_length * test_ratio))
    
    assert train_len + dev_len + test_len == ds_length, \
        "Total length of splits must match dataset length"
    indices = pd.Series(ds['train']['Index'])
    # shuffle dataset
    indices = indices.sample(frac=1.0, random_state=1)
    train, dev, test = (indices[:train_len],
                        indices[train_len:dev_len + train_len],
                            indices[dev_len + train_len:test_len + dev_len + train_len])
    with open(path_dir + '/' + filename, "wb") as f:
        pickle.dump((train, dev, test), f)
    return train, dev, test

def encode_image_batch(batch, model, preprocess_val):
    x = torch.stack([preprocess_val(img) for img in batch])
    x = x.to(device)
    with torch.inference_mode():
        y = model.encode_image(x)
    return y

def encode_text(text, model, tokenizer):
    x = tokenizer(text)
    x = x.to(device)
    with torch.inference_mode():
        y = model.encode_text(x)
    return y

def load_image_encoder(model_name: str = 'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'):
    import open_clip
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        model_name, cache_dir='/dev/shm')
    model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess_val, tokenizer

def encode_d3_images(d3_path: str, output_path: str, enforce_compute: bool = False):
    import json
    from PIL import Image
    with open(d3_path + '/dataset/dataset.json') as f:
        d3 = json.load(f)
    logging.info(f"Encoding {len(d3) * 2} images for D3...")
    path_dir = f"{output_path}/d3_vecs"
    os.makedirs(path_dir, exist_ok=True)
    filename = f"d3_vecs.pkl"
    if os.path.exists(path_dir + '/' + filename) and not enforce_compute:
        logging.info(f"Loading d3 with image vectors from {path_dir + '/' + filename}")
        with open(path_dir + '/' + filename, "rb") as f:
            return pickle.load(f)
    from tqdm import tqdm
    model, preprocess_val, _ = load_image_encoder()
    for key, sample in tqdm(d3.items(), total=len(d3)):
        left, right = sample['left'], sample['right']
        limage = Image.open(d3_path + left)
        rimage = Image.open(d3_path + right)
        vecs = encode_image_batch([limage, rimage], model, preprocess_val)
        d3[key]['left_vec'] = vecs[0].cpu()
        d3[key]['right_vec'] = vecs[1].cpu()

    with open(path_dir + '/' + filename, 'wb') as f:
        pickle.dump(d3, f)
    logging.info(f"Encoded images for D3")
    return d3


def encode_nb_images(nb: list, output_path: str, split: str,
                    bs: int = 64):
    logging.info(f"Encoding {len(nb)} images for NaturalBench {split} split...")
    path_dir = f"{output_path}/nb_vecs"
    os.makedirs(path_dir, exist_ok=True)
    filename = f"nb_{split}_vecs.pkl"
    if os.path.exists(path_dir + '/' + filename):
        logging.info(f"Loading image vectors from {path_dir + '/' + filename}")
        with open(path_dir + '/' + filename, "rb") as f:
            return pickle.load(f)
    from tqdm import tqdm
    model, preprocess_val, _ = load_image_encoder()
    image_vecs = []
    batch = []  
    for item in tqdm(nb):
        batch.append(item['image'])
        if len(batch) < bs:
            continue
        image_vecs.append(encode_image_batch(batch, model, preprocess_val))
        batch = []
    if len(batch) > 0:
        image_vecs.append(encode_image_batch(batch, model, preprocess_val))
    image_vecs = torch.vstack(image_vecs)
    with open(path_dir + '/' + filename, 'wb') as f:
        pickle.dump(image_vecs, f)
    logging.info(f"Encoded {len(image_vecs)} images for NaturalBench {split} split")
    return image_vecs

def encode_index_images(image_dataset: str, output_path: Path,
                bs: int = 64):
    from datasets import load_dataset
    from tqdm import tqdm
    path_dir = f"{output_path}/index_vecs/{image_dataset.replace('/', '_')}"
    os.makedirs(path_dir, exist_ok=True)
    filename = f"/all_vecs.pkl"
    if os.path.exists(path_dir + '/' + filename):
        logging.info(f"Loading image vectors from {path_dir + '/' + filename}")
        with open(path_dir + '/' + filename, "rb") as f:
            return pickle.load(f)
    model, preprocess_val, _ = load_image_encoder()
    image_ds = load_dataset(image_dataset, cache_dir='/dev/shm',
                            split='test', streaming=True)
    it = iter(image_ds)
    image_vecs = []
    mapping = {}
    batch = []
    for sample in tqdm(it):
        mapping[sample['img_id']] = sample['filename']
        img_path = f"{path_dir}/{sample['filename']}"
        sample['image'].save(img_path)
        batch.append(sample['image'])
        if len(batch) < bs:
            continue
        image_vecs.append(encode_image_batch(batch, model, preprocess_val))
        batch = []
    if len(batch) > 0:
        image_vecs.append(encode_image_batch(batch, model, preprocess_val))
    image_vecs = torch.vstack(image_vecs)

    with open(path_dir + '/' + filename, 'wb') as f:
        pickle.dump((image_vecs, mapping), f)
    
    return image_vecs, mapping
    

def prepare_raw_data(output_path: Path,
        vqa_dataset: str = 'BaiqiL/NaturalBench',
        image_dataset: str = 'nlphuji/flickr30k',
        train_ratio: float = 0.6, val_ratio: float = 0.2,
        return_classifier_inputs: bool = False,
        splits: list = ['train', 'dev', 'test']):
    """Prepare and preprocess the data."""
    logging.info("Starting data preparation...")
    supported_vqa_dataset = ['BaiqiL/NaturalBench']
    supported_image_dataset = ['nlphuji/flickr30k', 'coco_train']
    if vqa_dataset not in supported_vqa_dataset:
        raise ValueError(f"Unsupported VQA dataset: {vqa_dataset}")
    if image_dataset not in supported_image_dataset:
        raise ValueError(f"Unsupported image dataset: {image_dataset}")
    assert train_ratio + val_ratio < 1.0, "Train and validation ratio must sum to less than 1.0"
    train_indices, dev_indices, test_indices = prepare_splits(vqa_dataset, output_path, train_ratio, val_ratio)
    logging.info(f"prepared splits train: {len(train_indices)}, dev: {len(dev_indices)}, test: {len(test_indices)}")
    res = {}
    if 'test' in splits:
        nb_test = prepare_naturalbench(test_indices, output_path)
        res['nb_test'] = nb_test
        res['test_indices'] = test_indices
    if 'dev' in splits:
        nb_dev = prepare_naturalbench(dev_indices, output_path)
        res['nb_dev'] = nb_dev
        res['dev_indices'] = dev_indices
    if 'train' in splits:
        nb_train = prepare_naturalbench(train_indices, output_path)
        res['nb_train'] = nb_train
        res['train_indices'] = train_indices
    if return_classifier_inputs:
        index_vecs, mapping = encode_index_images(image_dataset, output_path)
        if 'dev' in splits:
            dev_vecs = encode_nb_images(nb_dev, output_path, 'dev')
            res['vecs_dev'] = dev_vecs
        if 'test' in splits:
            test_vecs = encode_nb_images(nb_test, output_path, 'test')
            res['vecs_test'] = test_vecs
        if 'train' in splits:
            train_vecs = encode_nb_images(nb_train, output_path, 'train')
            res['vecs_train'] = train_vecs
        res['index_vecs'] = index_vecs
        res['img_mapping'] = mapping
    return res
    
def load_lvlm(model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    model = AutoModelForVision2Seq.from_pretrained(
            model_name, cache_dir='/dev/shm', token=hf_token)
    model.to(device)
    model.to(dtype=torch.bfloat16)
    model.eval()
    additional_kwargs = {}
    if "Qwen2-VL" in model_name:
        additional_kwargs["max_pixels"] = 2048*28*28
    processor = AutoProcessor.from_pretrained(model_name, **additional_kwargs, token=hf_token)
    if "mistral" in model_name.lower():
        processor.tokenizer.pad_token_id = 11
    if "llava-onevision" in model_name.lower():
        processor.tokenizer.padding_side = "left"
    return model, processor

def inference_llama(question, image, processor, model, sample=True, **generation_kwargs):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
            {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(device)
    inputs = inputs.to(torch.bfloat16)
    generated_ids = model.generate(**inputs, do_sample=sample, max_new_tokens=100, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def inference(question, image, processor, model, **kwargs):
    if 'qwen2-vl' in model.config._name_or_path.lower():
        return inference_qwen(question, image, processor, model, **kwargs)
    elif 'mistral' in model.config._name_or_path.lower():
        return inference_mistral(question, image, processor, model, **kwargs)
    elif 'llava-onevision' in model.config._name_or_path.lower():
        return inference_llava_ov(question, image, processor, model, **kwargs)
    elif 'llama-3.2' in model.config._name_or_path.lower():
        return inference_llama(question, image, processor, model, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model.config._name_or_path}")
    

def batch_inference(questions, images, processor, model, **kwargs):
    if 'qwen2-vl' in model.config._name_or_path.lower():
        return batch_inference_qwen(questions, images, processor, model, **kwargs)
    # elif 'mistral' in model.config._name_or_path.lower():
    #     return inference_mistral(question, image, processor, model, **kwargs)
    elif 'llava-onevision' in model.config._name_or_path.lower():
        return batch_inference_llava_ov(questions, images, processor, model, **kwargs)
    elif 'llama-3.2' in model.config._name_or_path.lower():
        return batch_inference_llama(questions, images, processor, model, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model.config._name_or_path}")

def batch_inference_llama(questions, images, processor, model, sample=True, **generation_kwargs):
    texts = []
    for question in questions:
        conversation = [
            {
                "role": "user",
            "content": [
                {"type": "image"},
            {"type": "text", "text": question},
            ],
        }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        texts.append(prompt)
    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True, add_special_tokens=False)
    inputs = inputs.to(device)
    inputs = inputs.to(torch.bfloat16)
    generated_ids = model.generate(**inputs, do_sample=sample, max_new_tokens=100, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def batch_inference_qwen(questions, images, processor, model, sample=True, **generation_kwargs):
    conversations = []
    texts = []
    for image, question in zip(images, questions):
        messages = [
            {
                "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question}
            ],
        }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)
        conversations.append(messages)

    image_inputs, video_inputs = process_vision_info(conversations)
    inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
    )
    inputs = inputs.to(device)
    inputs = inputs.to(torch.bfloat16)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, do_sample=sample, max_new_tokens=100, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts
    

def inference_cd(question, images, processor, model, **kwargs):
    if 'qwen2-vl' in model.config._name_or_path.lower():
        return inference_cd_qwen(question, images, processor, model, **kwargs)
    elif 'mistral' in model.config._name_or_path.lower():
        return inference_cd_mistral(question, images, processor, model, **kwargs)
    elif 'llava-onevision' in model.config._name_or_path.lower():
        return inference_cd_llava_ov(question, images, processor, model, **kwargs)
    elif 'llama-3.2' in model.config._name_or_path.lower():
        return inference_cd_llama(question, images, processor, model, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model.config._name_or_path}")

def inference_cd_llama(question, images, processor, model, alpha=0.5, generation_kwargs={}):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
            {"type": "text", "text": question},
            ],
        }
    ] 
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images,
        [prompt] * 2,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    lp = EnsembleLogitsProcessor(num_beams=1, source_weights=[1.0, -alpha])
    generated_ids = model.generate(**inputs, logits_processor = LogitsProcessorList([lp]),
                                do_sample=True, max_new_tokens=200, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]
    

def inference_cd_llava_ov(question, images, processor, model, alpha=0.5, generation_kwargs={}):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
            {"type": "text", "text": question},
            ],
        }
    ] 
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=images, text=[prompt] * 2, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    inputs = inputs.to(torch.bfloat16)
    lp = EnsembleLogitsProcessor(num_beams=1, source_weights=[1.0, -alpha])
    generated_ids = model.generate(**inputs, 
                                logits_processor = LogitsProcessorList([lp]),
                                do_sample=True,
                                max_new_tokens=200, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def inference_cd_mistral(question, images, processor, model, alpha=0.5, generation_kwargs={}):
    PROMPT = f"<s>[INST]{question}\n[IMG][/INST]"
    assert len(images) == 2
    inputs = processor(text=[PROMPT] * 2, images=images, return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    lp = EnsembleLogitsProcessor(num_beams=1, source_weights=[1.0, -alpha])
    generated_ids = model.generate(**inputs, logits_processor = LogitsProcessorList([lp]),
                                do_sample=True, max_new_tokens=200,
                                **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def inference_cd_qwen(question, images, processor, model, alpha=0.5, generation_kwargs={}):
    proc_images = []
    texts = []
    for image in images:
        messages = [
            {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question}
            ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        proc_images.append(image_inputs)
        texts.append(text)
    inputs = processor(
        text=texts,
        images=proc_images,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    inputs = inputs.to(torch.bfloat16)
    lp = EnsembleLogitsProcessor(num_beams=1, source_weights=[1.0, -alpha])
    generated_ids = model.generate(**inputs, 
                                logits_processor = LogitsProcessorList([lp]),
                                do_sample=True,
                                max_new_tokens=200, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def inference_mistral(question, image, processor, model, sample=True, **generation_kwargs):
    image = image.convert('RGB')
    PROMPT = f"<s>[INST]{question}\n[IMG][/INST]"
    inputs = processor(text=PROMPT, images=[image], return_tensors="pt")
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, do_sample=sample, max_new_tokens=100, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def inference_llava_ov(question, image, processor, model, sample=True, **generation_kwargs):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
            {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = inputs.to(device)
    inputs = inputs.to(torch.bfloat16)
    generated_ids = model.generate(**inputs, do_sample=sample, max_new_tokens=100, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def batch_inference_llava_ov(questions, images, processor, model, sample=True, **generation_kwargs):
    texts = []
    for question in questions:
        messages = [
            {
                "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {"type": "text", "text": question}
            ],
        }
        ]

        text = processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        texts.append(text)

    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    inputs = inputs.to(torch.bfloat16)
    generated_ids = model.generate(**inputs, do_sample=sample, max_new_tokens=100, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text


def inference_qwen(question, image, processor, model, sample=True, **generation_kwargs):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question}
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    inputs = inputs.to(torch.bfloat16)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, do_sample=sample, max_new_tokens=200, **generation_kwargs)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def inference_loss(model, processor, images, question):
    if 'qwen2-vl' in model.config._name_or_path.lower():
        return inference_loss_qwen(model, processor, images, question)
    elif 'mistral' in model.config._name_or_path.lower():
        return inference_loss_mistral(model, processor, images, question)
    elif 'llava-onevision' in model.config._name_or_path.lower():
        return inference_loss_llava_ov(model, processor, images, question)
    elif 'llama-3.2' in model.config._name_or_path.lower():
        return inference_loss_llama(model, processor, images, question)
    else:
        raise ValueError(f"Unsupported model: {model.config._name_or_path}")

def inference_loss_llama(model, processor, images, question):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
            {"type": "text", "text": question},
            ],
        }
    ] 
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt",
                       add_special_tokens=False,
                       padding=True)
    inputs = inputs.to(device)
    inputs = inputs.to(torch.bfloat16)
    with torch.inference_mode():
        return model(**inputs).logits

def inference_loss_llava_ov(model, processor, images, question):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
            {"type": "text", "text": question},
            ],
        }
    ] 
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    inputs = inputs.to(torch.bfloat16)
    with torch.inference_mode():
        return model(**inputs).logits

def inference_loss_mistral(model, processor, images, question):
    PROMPT = f"<s>[INST]{question}\n[IMG][/INST]"
    inputs = processor(text=[PROMPT] * len(images), images=images, return_tensors="pt", padding=True)
    inputs = inputs.to(device)

    with torch.inference_mode():
        return model(**inputs).logits

def inference_loss_qwen(model, processor, images, question):
    texts = []
    proc_images = []
    for image in images:
        messages = [
            {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": question}],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        proc_images.append(image_inputs)
        texts.append(text)
        
    inputs = processor(
        text=texts,
        images=proc_images,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to(device)
    inputs = inputs.to(torch.bfloat16)
    with torch.inference_mode():
        return model(**inputs).logits
    
def compute_sims(nb_vecs, index_vecs):
    nb_sims = torch.matmul(nb_vecs, index_vecs.T)
    topk = torch.topk(nb_sims, 101)
    processed_sims = []
    processed_indices = []
    for values_row, indices_row in zip(topk.values, topk.indices):
        if values_row[0] == 1.0:
            values_row = values_row[1:]
            indices_row = indices_row[1:]
        else:
            values_row = values_row[:-1]
            indices_row = indices_row[:-1]
        processed_sims.append(values_row)
        processed_indices.append(indices_row)
    sim_indices = torch.vstack(processed_indices)
    return sim_indices 

def process_question_rephrases(nb: list, question_rephrases: list):
    assert len(nb) == len(question_rephrases), "Number of samples must match number of question rephrases"
    rephrased_nb = []
    buffer = []
    for orig_nb_sample, sample_with_rephrases in zip(nb, question_rephrases):
        buffer.append((orig_nb_sample, sample_with_rephrases))
        if len(buffer) < 2:
            continue
        assert len(buffer) == 2
        (sample1, rephrases1), (sample2, rephrases2) = buffer
        assert sample1['question'] == sample2['question']
        rephrased_nb.append(sample1)
        rephrased_nb.append(sample2)
        # interleave rephrases with original sample
        for reph in rephrases1['processed_rephrases']:
            sample1_copy = sample1.copy()
            sample1_copy['question'] = reph
            rephrased_nb.append(sample1_copy)
            sample2_copy = sample2.copy()
            sample2_copy['question'] = reph
            rephrased_nb.append(sample2_copy)
        buffer = []
    return rephrased_nb


def load_img(idx, img_mapping, output_path: str, image_dataset: str):
    path_dir = f"{output_path}/index_vecs/{image_dataset.replace('/', '_')}"
    from PIL import Image
    if idx not in img_mapping:
        idx = str(idx)
    filename = img_mapping[idx]
    im = Image.open(path_dir + '/' + filename)
    return im

def get_transforms():
    from torchvision import transforms
    augment =  transforms.Compose([
        transforms.RandomResizedCrop(size=680, scale=(0.95,0.95)),
        transforms.RandomRotation(7),
        transforms.CenterCrop(640),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        transforms.ElasticTransform(alpha=4.0),
    ])
    return augment
        

def prepare_classifier_triplets(output_path: Path, split: str,
                                use_augmentations: bool = True,
                                question_rephrases_path: Optional[str] = None,
                                model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    path_dir = f"{output_path}/{model_name.replace('/', '_')}/classifier_triplets"
    os.makedirs(path_dir, exist_ok=True)
    filename = f"{split}_triplets.pickle"
    if os.path.exists(path_dir + '/' + filename):
        logging.info(f"Loading triplets from {path_dir + '/' + filename}")
        with open(path_dir + '/' + filename, "rb") as f:
            return pickle.load(f)
    from torch.nn import functional as F
    from tqdm import tqdm
    logging.info("First preparing raw data...")
    raw_data = prepare_raw_data(output_path, return_classifier_inputs=True)
    logging.info("Preparing classifier triplets...")
    index_vecs = raw_data['index_vecs']
    index_vecs = F.normalize(index_vecs, dim=-1).to(device)
    img_mapping = raw_data['img_mapping']
    assert split in ['train', 'dev']
    nb_vecs = raw_data[f'vecs_{split}']
    nb = raw_data[f'nb_{split}']
    
    if question_rephrases_path is not None:
        logging.info("Processing question rephrases...")
        with open(question_rephrases_path, 'rb') as f:
            question_rephrases = pickle.load(f)
        nb = process_question_rephrases(nb, question_rephrases)
    
    nb_vecs = F.normalize(nb_vecs, dim=-1).to(device)

    if len(nb_vecs) != len(nb):
        # need to replicate pairs of vectors for rephrased samples
        k = len(nb) // len(nb_vecs)
        pairs = nb_vecs.reshape(-1, 2, nb_vecs.shape[1])
        repeated = pairs.repeat_interleave(k, dim=0)
        # Reshape back to 2D: (n_pairs * k * 2, features)
        nb_vecs = repeated.reshape(-1, nb_vecs.shape[1])
        assert len(nb_vecs) == len(nb)

    sim_indices = compute_sims(nb_vecs, index_vecs)

    def find_least_similar_to(idx, candidates):
        candidate_vecs = index_vecs[candidates]
        target_vec = nb_vecs[idx]
        sims = torch.matmul(target_vec, candidate_vecs.T)
        return [candidates[x] for x in torch.topk(sims, k=5, largest=False).indices.squeeze()]

    model, processor = load_lvlm(model_name)
    hidden_states = []

    def get_hidden(module, input, output):
        hidden_states.append(output[0][0, -1])

    
    def register_hook(model):
        if 'qwen2-vl' in model.config._name_or_path.lower():
            model.model.layers[-1]._forward_hooks = OrderedDict()
            model.model.layers[-1].register_forward_hook(get_hidden)
        elif 'mistral' in model.config._name_or_path.lower() or \
            'llava-onevision' in model.config._name_or_path.lower() or \
            'llama-3.2' in model.config._name_or_path.lower():
            model.language_model.model.layers[-1]._forward_hooks = OrderedDict()
            model.language_model.model.layers[-1].register_forward_hook(get_hidden)
        else:
            raise ValueError(f"Unsupported model: {model.config._name_or_path}")

    register_hook(model)
    buffer = []
    dataset = []

    def inference(images, question):
        return inference_loss(model, processor, images, question)
    
    if use_augmentations:
        logging.info("Using augmentations...")
        augment = get_transforms()
    else:
        augment = lambda x: x

    load_image_f = lambda x: load_img(x, img_mapping, output_path, 'nlphuji/flickr30k')
    logging.info(f"Preparing {len(nb)} triplets...")
    for i, (sample, sim_images) in tqdm(enumerate(zip(nb, sim_indices)), total=len(nb)):
        buffer.append((sample, sim_images))
        if len(buffer) < 2:
            continue
        assert len(buffer) == 2

        (sample1, sim_images1), (sample2, sim_images2) = buffer
        assert sample1['question'] == sample2['question']
        
        # sample 1
        least_similar_to_1 = find_least_similar_to(i - 1, sim_images2)
        sim_images = [load_image_f(x.item()) for x in least_similar_to_1]
        negative_for_image1_hiddens = []
        for sim_image in sim_images:
            inference([augment(sim_image)], sample1['question'])
            hiddens = hidden_states[0]
            negative_for_image1_hiddens.append(hiddens)
            hidden_states = []
        
        inference([augment(sample2['image'])], sample2['question'])
        image_2_hiddens = hidden_states[0]
        hidden_states = []

        # sample 2
        least_similar_to_2 = find_least_similar_to(i, sim_images1)
        sim_images = [load_image_f(x.item()) for x in least_similar_to_2]
        negative_for_image2_hiddens = []
        for sim_image in sim_images:
            inference([augment(sim_image)], sample2['question'])
            hiddens = hidden_states[0]
            negative_for_image2_hiddens.append(hiddens)
            hidden_states = []
        
        inference([augment(sample1['image'])], sample1['question'])
        image_1_hiddens = hidden_states[0]
        hidden_states = []

        dataset.append({'query': image_1_hiddens.cpu(),
                        'positive': image_2_hiddens.cpu(),
                        'negative': torch.vstack(negative_for_image1_hiddens).cpu()})
        dataset.append({'query': image_2_hiddens.cpu(),
                        'positive': image_1_hiddens.cpu(),
                        'negative': torch.vstack(negative_for_image2_hiddens).cpu()})
        buffer = []

    with open(path_dir + '/' + filename, 'wb') as f:
        pickle.dump(dataset, f)
    return dataset

def train_classifier(data_path: Path, model_path: Path):
    """Train the classifier model."""
    logging.info("Starting classifier training...")
    # TODO: Implement model training logic
    pass

def run_self_contrast_in_captioning(output_path: Path,
                                 split: str,
                                 model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
                                 generation_kwargs: dict = {},
                                 image_dataset: str = "nlphuji/flickr30k",
                                 alpha=0.5,
                                 k_values: list = [5, 10, 15, 30, 60, 100]):
    """Run CD with classifier-selected contrasts on specified split."""
    logging.info(f"Running CD classifier on {split} split with k values: {k_values}")
    
    # Setup paths and check for existing results
    generation_kwargs_hash = _filename_hash(generation_kwargs=generation_kwargs,
                                            alpha=alpha,
                                            image_dataset=image_dataset,
                                            k_values=k_values)
    os.makedirs(f"{output_path}/{model_name.replace('/', '_')}", exist_ok=True)
    
    # Load data and models
    data = prepare_raw_data(output_path, splits=[split],
                            image_dataset=image_dataset,
                            return_classifier_inputs=True)
    assert f'nb_{split}' in data, f"No {split} split found in data"
    nb = data[f'nb_{split}']
    nb_vecs = data[f'vecs_{split}']
    index_vecs = data['index_vecs']
    img_mapping = data['img_mapping']
    index_vecs = F.normalize(index_vecs, dim=-1).to(device)
    nb_vecs = F.normalize(nb_vecs, dim=-1).to(device)
    
    model, processor = load_lvlm(model_name)
    
    # Compute similarities and get indices
    sim_indices = compute_sims(nb_vecs, index_vecs)
    
    chunk_size = 10
    from tqdm import tqdm
    max_k = max(k_values)
    assert max_k % chunk_size == 0
    load_image_f = lambda x: load_img(x, img_mapping, output_path, image_dataset)
    img_encoder, _, img_encoder_tokenizer = load_image_encoder()
    results = []
    for sample, sim_imgs in tqdm(zip(nb, sim_indices), total=len(nb)):
        all_images = [load_image_f(x.item()) for x in sim_imgs[:max_k]]
        buffer = []
        captions = []
        for img in all_images:
            buffer.append(img)
            if len(buffer) < chunk_size:
                continue
            with torch.inference_mode():
                batch_caps = batch_inference([CAPTIONING_PROMPT] * chunk_size, buffer, processor, model)
            captions.extend(batch_caps)
            buffer = []
        caption_vecs = encode_text(captions, img_encoder, img_encoder_tokenizer)
        caption_vecs = F.normalize(caption_vecs, dim=-1).to(device)
        target_image = sample['image']
        with torch.inference_mode():
            target_cap = batch_inference([CAPTIONING_PROMPT] * 1, [target_image], processor, model)
        target_vec = encode_text(target_cap, img_encoder, img_encoder_tokenizer)
        target_vec = F.normalize(target_vec, dim=-1).to(device)
        sims = caption_vecs @ target_vec.squeeze(0)
        
        prev_idx = -1
        prev_pred = None
        for k in k_values:
            sim_idx = sims[:k].argmax().item()
            if sim_idx == prev_idx:
                pred_k = prev_pred
            else:
                most_sim_image_k = all_images[sim_idx]
                pred_k = inference_cd(sample['question'], [target_image, most_sim_image_k], processor, model)
                prev_idx = sim_idx
                prev_pred = pred_k
            sample[f"pred_cap_k_{k}"] = pred_k
        sample['sim_image_captions'] = captions
        sample['target_caption'] = target_cap
        del sample['image']
        results.append(sample)

    with open(f"{output_path}/{model_name.replace('/', '_')}/self_captioning_cd_{split}_{generation_kwargs_hash}.pickle", 'wb') as f:
        pickle.dump(results, f)
    return results

        

def run_captioning_cd_oracle(output_path: Path, split: str,
                model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
                generation_kwargs: dict = {}, alpha=0.5):
    logging.info(f"Running CD oracle on {split} split...")
    import hashlib
    
    # for generating the hash only
    generation_kwargs['alpha'] = alpha
    generation_kwargs_hash = hashlib.md5(str(generation_kwargs).encode()).hexdigest()
    del generation_kwargs['alpha']
    os.makedirs(f"{output_path}/{model_name.replace('/', '_')}", exist_ok=True)
    filename = f"{output_path}/{model_name.replace('/', '_')}/captioning_cd_oracle_{split}_{generation_kwargs_hash}.pickle"
    if os.path.exists(filename):
        logging.info(f"Loading captioning CD oracle from {filename}")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    data = prepare_raw_data(output_path, splits=[split], return_classifier_inputs=False)
    assert f'nb_{split}' in data, f"No {split} split found in data"
    data_split = data[f'nb_{split}']
    model, processor = load_lvlm(model_name)
    from tqdm import tqdm
    buffer = []
    results_cd = []
    for sample in tqdm(data_split):
        buffer.append(sample)
        if len(buffer) < 4:
            continue
        assert len(buffer) == 4
        sample1, sample2, sample3, sample4 = buffer
        assert sample1['question'] == sample2['question']
        assert sample1['image'] != sample2['image']
        assert sample1['image'] == sample3['image']
        assert sample2['image'] == sample4['image']
        outs1 = inference_cd(CAPTIONING_PROMPT, [sample1['image'], sample2['image']],
                            processor=processor,
                            model=model,
                            alpha=alpha,
                            generation_kwargs=generation_kwargs)
        sample1['pred'] = outs1
        outs2 = inference_cd(CAPTIONING_PROMPT, [sample2['image'], sample1['image']],
                            processor=processor,
                            model=model,
                            alpha=alpha,
                            generation_kwargs=generation_kwargs)
        sample2['pred'] = outs2
        sample3['pred'] = outs1
        sample4['pred'] = outs2
        # remove images from samples
        del sample1['image']
        del sample2['image']    
        del sample3['image']
        del sample4['image']
        results_cd.append(sample1)
        results_cd.append(sample2)
        results_cd.append(sample3)
        results_cd.append(sample4)
        buffer = []
    with open(filename, 'wb') as f:
        pickle.dump(results_cd, f)
    return results_cd

def run_cd_oracle(output_path: Path, split: str,
                model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
                generation_kwargs: dict = {}, alpha=0.5):
    logging.info(f"Running CD oracle on {split} split...")
    import hashlib
    
    # for generating the hash only
    generation_kwargs['alpha'] = alpha
    generation_kwargs_hash = hashlib.md5(str(generation_kwargs).encode()).hexdigest()
    del generation_kwargs['alpha']
    os.makedirs(f"{output_path}/{model_name.replace('/', '_')}", exist_ok=True)
    filename = f"{output_path}/{model_name.replace('/', '_')}/cd_oracle_{split}_{generation_kwargs_hash}.pickle"
    if os.path.exists(filename):
        logging.info(f"Loading CD oracle from {filename}")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    data = prepare_raw_data(output_path, return_classifier_inputs=False)
    assert f'nb_{split}' in data, f"No {split} split found in data"
    data_split = data[f'nb_{split}']
    model, processor = load_lvlm(model_name)
    from tqdm import tqdm
    buffer = []
    results_cd = []
    for sample in tqdm(data_split):
        buffer.append(sample)
        if len(buffer) < 2:
            continue
        assert len(buffer) == 2
        sample1, sample2 = buffer
        assert sample1['question'] == sample2['question']
        question = sample1['question']
        outs1 = inference_cd(question, [sample1['image'], sample2['image']],
                            processor=processor,
                            model=model,
                            alpha=alpha,
                            generation_kwargs=generation_kwargs)
        sample1['pred'] = outs1
        outs2 = inference_cd(question, [sample2['image'], sample1['image']],
                            processor=processor,
                            model=model,
                            alpha=alpha,
                            generation_kwargs=generation_kwargs)
        sample2['pred'] = outs2
        # remove images from samples
        del sample1['image']
        del sample2['image']
        results_cd.append(sample1)
        results_cd.append(sample2)
        buffer = []
    with open(filename, 'wb') as f:
        pickle.dump(results_cd, f)
    return results_cd

def run_captioning_baseline(output_path: Path, split: str,
                model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
                generation_kwargs: dict = {}):
    """Run captioning baseline inference on specified split (dev/test)."""
    logging.info(f"Running captioning baseline inference on {split} split...")
    generation_kwargs_hash = _filename_hash(generation_kwargs=generation_kwargs)
    os.makedirs(f"{output_path}/{model_name.replace('/', '_')}", exist_ok=True)
    filename = f"{output_path}/{model_name.replace('/', '_')}/captioning_baseline_{split}_{generation_kwargs_hash}.pickle"
    if os.path.exists(filename):
        logging.info(f"Loading captioning baseline from {filename}")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    data = prepare_raw_data(output_path, splits=[split], return_classifier_inputs=False)
    assert f'nb_{split}' in data, f"No {split} split found in data"
    data_split = data[f'nb_{split}']
    model, processor = load_lvlm(model_name)
    from tqdm import tqdm
    results = []
    img_hashes = {}
    for sample in tqdm(data_split):
        sample_copy = sample.copy()
        img_hash = hash(sample_copy['image'].tobytes())
        if img_hash in img_hashes:
            sample_copy['caption'] = img_hashes[img_hash]
        else:
            out = inference(CAPTIONING_PROMPT, sample_copy['image'], processor, model,
                        sample=True, **generation_kwargs)
            sample_copy['caption'] = out
            img_hashes[img_hash] = out
        del sample_copy['image']
        results.append(sample_copy)
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    return results
    


def run_baseline(output_path: Path, split: str,
                model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
                generation_kwargs: dict = {}):
    """Run baseline inference on specified split (dev/test)."""
    logging.info(f"Running baseline inference on {split} split...")
    import hashlib
    generation_kwargs_hash = hashlib.md5(str(generation_kwargs).encode()).hexdigest()
    os.makedirs(f"{output_path}/{model_name.replace('/', '_')}", exist_ok=True)
    filename = f"{output_path}/{model_name.replace('/', '_')}/baseline_{split}_{generation_kwargs_hash}.pickle"
    if os.path.exists(filename):
        logging.info(f"Loading baseline from {filename}")
        with open(filename, 'rb') as f:
            return pickle.load(f)
    data = prepare_raw_data(output_path, splits=[split], return_classifier_inputs=False)
    assert f'nb_{split}' in data, f"No {split} split found in data"
    data_split = data[f'nb_{split}']
    model, processor = load_lvlm(model_name)
    from tqdm import tqdm
    results = []
    for sample in tqdm(data_split):
        sample_copy = sample.copy()
        out = inference(sample_copy['question'], sample_copy['image'], processor, model,
                        sample=True, **generation_kwargs)
        sample_copy['pred'] = out
        del sample_copy['image']
        results.append(sample_copy)
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    return results

from collections import OrderedDict
from cd import Classifier

def register_hook(hidden_states, model):
    def get_hidden(module, input, output):
        hidden_states.append(output[0][:, -1])
        
    if 'qwen2-vl' in model.config._name_or_path.lower():
        model.model.layers[-1]._forward_hooks = OrderedDict()
        model.model.layers[-1].register_forward_hook(get_hidden)
    elif 'mistral' in model.config._name_or_path.lower() or \
        'llava-onevision' in model.config._name_or_path.lower() or \
        'llama-3.2' in model.config._name_or_path.lower():
        model.language_model.model.layers[-1]._forward_hooks = OrderedDict()
        model.language_model.model.layers[-1].register_forward_hook(get_hidden)
    else:
        raise ValueError(f"Unsupported model: {model.config._name_or_path}")

def unregister_hook(model):
    if 'qwen2-vl' in model.config._name_or_path.lower():
        model.model.layers[-1]._forward_hooks = OrderedDict()
    elif 'mistral' in model.config._name_or_path.lower() or \
        'llava-onevision' in model.config._name_or_path.lower() or \
        'llama-3.2' in model.config._name_or_path.lower():
        model.language_model.model.layers[-1]._forward_hooks = OrderedDict()
    else:
        raise ValueError(f"Unsupported model: {model.config._name_or_path}")


def load_cd_top_k_classifier(output_path: Path,
                          split: str,
                          classifier_path: str,
                          model_name: str,
                          generation_kwargs: dict,
                          k_values: list = [4, 8, 16, 32, 64, 100],
                          alpha: float = 0.5):
    """Load CD top k results from specified split."""
    generation_kwargs_hash = _filename_hash(generation_kwargs=generation_kwargs, alpha=alpha, classifier_path=classifier_path)
    results_by_k = {}
    for k in k_values:
        filename = f"{output_path}/{model_name.replace('/', '_')}/cd_classifier_k{k}_{split}_{generation_kwargs_hash}.pickle"
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found")
        logging.info(f"Loading CD top k classifier results from {filename}")
        with open(filename, 'rb') as f:
            results_by_k[k] = pickle.load(f)
    
    return results_by_k

def _filename_hash(*args, generation_kwargs: dict = {}, **kwargs):
    assert len(args) == 0, "No additional arguments are supported"
    generation_kwargs_copy = generation_kwargs.copy()
    import hashlib
    for k, v in kwargs.items():
        assert k not in generation_kwargs_copy, "cannot pass arg in kwargs and generation_kwargs"
        generation_kwargs_copy[k] = v
    hash = hashlib.md5(str(generation_kwargs_copy).encode()).hexdigest()
    return hash


def run_classifier_on_image_question(target_image,
                                     target_question,
                                     similar_images,
                                     processor,
                                     model,
                                     classifier,
                                     chunk_size: int = 20,
                                     alpha: float = 0.5,
                                     generation_kwargs: dict = {}):
    hidden_states = []
    register_hook(hidden_states, model)
    inference_loss(model, processor, [target_image], target_question)
    target_hiddens = hidden_states[0]
    target_hiddens = target_hiddens.to(device).to(torch.bfloat16)
    sim_image_hiddens = torch.Tensor().to(device)
    chunk_start = 0
    while chunk_start < len(similar_images):
        chunk_end = min(chunk_start + chunk_size, len(similar_images))
        chunk = similar_images[chunk_start:chunk_end]
        hidden_states = []
        register_hook(hidden_states, model)
        inference_loss(model, processor, chunk, target_question)
        chunk_hiddens = hidden_states[0]
        sim_image_hiddens = torch.cat((sim_image_hiddens, chunk_hiddens), dim=0)
        chunk_start += chunk_size
        
    sim_image_hiddens = sim_image_hiddens.to(torch.bfloat16)    
    unregister_hook(model)
    target_hiddens_cat = target_hiddens.repeat(len(similar_images), 1)
    cated_hiddens = torch.cat([target_hiddens_cat, sim_image_hiddens], dim=-1)
    logits = classifier(cated_hiddens).flatten()
    scores = torch.sigmoid(logits)
    pred_idx = scores.argmax().item()
    contrast_img = similar_images[pred_idx]
    return contrast_img, scores.max().item()


def load_classifier(classifier_path: str,
                    model_name: str):
    input_dim = {
        'Qwen/Qwen2-VL-2B-Instruct': 1536,
        'Qwen/Qwen2-VL-7B-Instruct': 3584,
        'llava-hf/llava-onevision-qwen2-7b-ov-hf': 3584,
        'meta-llama/Llama-3.2-11B-Vision-Instruct': 4096,
    }[model_name]
    classifier = Classifier(input_dim=input_dim)
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.eval()
    classifier.to(device).to(torch.bfloat16)
    return classifier

def run_cd_top_k_classifier(output_path: Path,
                            split: str,
                            classifier_path: str,
                            model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
                            generation_kwargs: dict = {},
                            k_values: list = [4, 8, 16, 32, 64, 100],
                            alpha: float = 0.5,
                            captioning: bool = False,
                            image_dataset: str = 'nlphuji/flickr30k'):
    """Run CD with classifier-selected contrasts on specified split."""
    logging.info(f"Running CD classifier on {split} split with k values: {k_values}")
    
    # Setup paths and check for existing results
    generation_kwargs_hash = _filename_hash(generation_kwargs=generation_kwargs,
                                            alpha=alpha,
                                            classifier_path=classifier_path,
                                            captioning=captioning,
                                            image_dataset=image_dataset)
    os.makedirs(f"{output_path}/{model_name.replace('/', '_')}", exist_ok=True)
    
    # Load data and models
    data = prepare_raw_data(output_path, splits=[split], image_dataset=image_dataset, return_classifier_inputs=True)
    assert f'nb_{split}' in data, f"No {split} split found in data"
    nb = data[f'nb_{split}']
    nb_vecs = data[f'vecs_{split}']
    index_vecs = data['index_vecs']
    img_mapping = data['img_mapping']
    index_vecs = F.normalize(index_vecs, dim=-1).to(device)
    nb_vecs = F.normalize(nb_vecs, dim=-1).to(device)
    
    model, processor = load_lvlm(model_name)
    classifier = load_classifier(classifier_path, model_name)
    
    # Compute similarities and get indices
    sim_indices = compute_sims(nb_vecs, index_vecs)
    
    # Process samples
    results_by_k = {k: [] for k in k_values}
    chunk_size = 20
    from tqdm import tqdm
    max_k = max(k_values)
    
    for i, (sample, sim_imgs) in tqdm(enumerate(zip(nb, sim_indices)), total=len(nb)):
        # Get target hidden states
        hidden_states = []
        register_hook(hidden_states, model)
        inference_loss(model, processor, [sample['image']], sample['question'])
        target_hiddens = hidden_states[0]
        target_hiddens = target_hiddens.to(device).to(torch.bfloat16)
        
        # Load similar images up to max k
        
        all_sim_images = [load_img(x.item(), img_mapping, output_path, image_dataset)
                        for x in sim_imgs[:max_k]]
        
        # Process similar images in chunks
        sim_image_hiddens = torch.Tensor().to(device)
        chunk_start = 0
        while chunk_start < len(all_sim_images):
            chunk_end = min(chunk_start + chunk_size, len(all_sim_images))
            chunk = all_sim_images[chunk_start:chunk_end]
            hidden_states = []
            register_hook(hidden_states, model)
            inference_loss(model, processor, chunk, sample['question'])
            chunk_hiddens = hidden_states[0]
            sim_image_hiddens = torch.cat((sim_image_hiddens, chunk_hiddens), dim=0)
            chunk_start += chunk_size
        
        all_sim_image_hiddens = sim_image_hiddens.to(torch.bfloat16)
        
        unregister_hook(model)
        # Process each k value
        prev_pred_idx = -1
        prev_pred = None
        
        for k in k_values:
            sim_image_hiddens = all_sim_image_hiddens[:k]
            target_hiddens_cat = target_hiddens.repeat(k, 1)
            cated_hiddens = torch.cat([target_hiddens_cat, sim_image_hiddens], dim=-1)
            
            logits = classifier(cated_hiddens).flatten()
            pred_idx = torch.sigmoid(logits).argmax().item()
            
            if pred_idx != prev_pred_idx:
                contrast_img = all_sim_images[pred_idx]
                pred = inference_cd(sample['question'] if not captioning else CAPTIONING_PROMPT, 
                                [sample['image'], contrast_img],
                                processor=processor,
                                model=model,
                                alpha=alpha,
                                generation_kwargs=generation_kwargs)
                prev_pred_idx = pred_idx
                prev_pred = pred
            else:
                contrast_img = all_sim_images[prev_pred_idx]
                pred = prev_pred
            
            sample_k = sample.copy()
            sample_k['pred'] = pred
            sample_k['contrast_img'] = contrast_img
            sample_k['contrast_img_idx'] = pred_idx
            results_by_k[k].append(sample_k)
    
    # Save results for each k
    for k in k_values:
        filename = f"{output_path}/{model_name.replace('/', '_')}/{'captioning_' if captioning else ''}cd_classifier_k{k}_{split}_{generation_kwargs_hash}.pickle"
        with open(filename, 'wb') as f:
            pickle.dump(results_by_k[k], f)
    
    return results_by_k

def run_classifier(model_path: Path, data_path: Path, output_path: Path, split: str):
    """Run classifier inference on specified split (dev/test)."""
    logging.info(f"Running classifier inference on {split} split...")
    # TODO: Implement classifier inference logic
    pass

def main():
    parser = argparse.ArgumentParser(description="Model Training and Inference Pipeline")
    parser.add_argument("--mode", type=str, required=True,
                    choices=['prepare_raw_data', 'prepare_classifier_triplets', 'train', 'baseline', 'classify'],
                    help="Operation mode")
    parser.add_argument("--output-path", type=Path, required=True,
                    help="Path to output directory")
    parser.add_argument("--model-path", type=Path,
                    help="Path to model directory (for train/classify modes)")
    parser.add_argument("--split", type=str, choices=['dev', 'test'],
                    help="Data split for inference (for baseline/classify modes)")
    parser.add_argument("--vqa-dataset", type=str, default='BaiqiL/NaturalBench',
                    help="VQA dataset to use")
    parser.add_argument("--image-dataset", type=str, default='nlphuji/flickr30k',
                    help="Image dataset to use")
    parser.add_argument("--use-augmentations", action='store_true',
                    help="Whether to use augmentations in prepare_classifier_triplets")
    parser.add_argument("--question-rephrases-path", type=str,
                    help="Path to question rephrases file (optional)")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    args.output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging()

    # Execute the requested operation
    if args.mode == 'prepare_raw_data':
        prepare_raw_data(args.vqa_dataset, args.image_dataset, args.output_path)
    elif args.mode == 'prepare_classifier_triplets':
        prepare_classifier_triplets(args.output_path, 'train',
                                use_augmentations=args.use_augmentations,
                                question_rephrases_path=args.question_rephrases_path)
    elif args.mode == 'train':
        if args.model_path is None:
            raise ValueError("--model-path is required for training mode")
        train_classifier(args.data_path, args.model_path)
    elif args.mode == 'baseline':
        if args.split is None:
            raise ValueError("--split is required for baseline mode")
        run_baseline(args.data_path, args.output_path, args.split)
    elif args.mode == 'classify':
        if args.model_path is None or args.split is None:
            raise ValueError("--model-path and --split are required for classify mode")
        run_classifier(args.model_path, args.data_path, args.output_path, args.split)

if __name__ == "__main__":
    main()
