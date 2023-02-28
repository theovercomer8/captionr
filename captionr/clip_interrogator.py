import hashlib
import math
import numpy as np
import open_clip
import os, subprocess
import pickle
import time
import torch
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
from typing import List
import logging
import requests
from thefuzz import fuzz

@dataclass 
class Config:
    captionr_config:any = None
    # models can optionally be passed in directly
    clip_model = None
    clip_preprocess = None

    # clip settings
    clip_model_name: str = 'ViT-L-14/openai'
    clip_model_path: str = None

    # interrogator settings
    cache_path: str = 'cache'
    chunk_size: int = 2048
    data_path: str = os.path.join(os.path.dirname(__file__), 'data')
    device: str = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    flavor_intermediate_count: int = 2048
    quiet: bool = False # when quiet progress bars are not shown

    fuzz_ratio: int = 50

class Interrogator():
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.config.chunk_size = 2048 if config.clip_model_name == 'ViT-L-14/openai' else 1024
        self.load_clip_model()

    def load_clip_model(self):
        start_time = time.time()
        config = self.config
        logging.info(f'Config cache path: {config.cache_path}')
        if config.clip_model is None:
            if config.clip_model_name == 'ViT-L-14/openai':
                if not os.path.exists(os.path.join(config.cache_path,'ViT-L-14_openai_flavors.pkl')):
                    r = requests.get('https://github.com/theovercomer8/captionr/raw/main/data/ViT-L-14_openai_flavors.pkl', stream=True)
                    with open(os.path.join(config.cache_path,'ViT-L-14_openai_flavors.pkl'), 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                if not os.path.exists(os.path.join(config.cache_path,'ViT-L-14_openai_artists.pkl')):
                    r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_artists.pkl', stream=True)
                    with open(os.path.join(config.cache_path,'ViT-L-14_openai_artists.pkl'), 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                if not os.path.exists(os.path.join(config.cache_path,'ViT-L-14_openai_mediums.pkl')):
                    r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_mediums.pkl', stream=True)
                    with open(os.path.join(config.cache_path,'ViT-L-14_openai_mediums.pkl'), 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                if not os.path.exists(os.path.join(config.cache_path,'ViT-L-14_openai_movements.pkl')):
                    r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_movements.pkl', stream=True)
                    with open(os.path.join(config.cache_path,'ViT-L-14_openai_movements.pkl'), 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                if not os.path.exists(os.path.join(config.cache_path,'ViT-L-14_openai_trendings.pkl')):
                    r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_trendings.pkl', stream=True)
                    with open(os.path.join(config.cache_path,'ViT-L-14_openai_trendings.pkl'), 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
            elif config.clip_model_name == 'ViT-bigG-14/openai':
                print('No cache items to preload')
                # if not os.path.exists(os.path.join(config.cache_path,'ViT-bigG-14_openai_flavors.pkl')):
                #     r = requests.get('https://github.com/theovercomer8/captionr/raw/main/data/ViT-L-14_openai_flavors.pkl', stream=True)
                #     with open(os.path.join(config.cache_path,'ViT-L-14_openai_flavors.pkl'), 'wb') as fd:
                #         for chunk in r.iter_content(chunk_size=128):
                #             fd.write(chunk)
                # if not os.path.exists(os.path.join(config.cache_path,'ViT-bigG-14_openai_artists.pkl')):
                #     r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_artists.pkl', stream=True)
                #     with open(os.path.join(config.cache_path,'ViT-L-14_openai_artists.pkl'), 'wb') as fd:
                #         for chunk in r.iter_content(chunk_size=128):
                #             fd.write(chunk)
                # if not os.path.exists(os.path.join(config.cache_path,'ViT-L-14_openai_mediums.pkl')):
                #     r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_mediums.pkl', stream=True)
                #     with open(os.path.join(config.cache_path,'ViT-L-14_openai_mediums.pkl'), 'wb') as fd:
                #         for chunk in r.iter_content(chunk_size=128):
                #             fd.write(chunk)
                # if not os.path.exists(os.path.join(config.cache_path,'ViT-L-14_openai_movements.pkl')):
                #     r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_movements.pkl', stream=True)
                #     with open(os.path.join(config.cache_path,'ViT-L-14_openai_movements.pkl'), 'wb') as fd:
                #         for chunk in r.iter_content(chunk_size=128):
                #             fd.write(chunk)
                # if not os.path.exists(os.path.join(config.cache_path,'ViT-L-14_openai_trendings.pkl')):
                #     r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_trendings.pkl', stream=True)
                #     with open(os.path.join(config.cache_path,'ViT-L-14_openai_trendings.pkl'), 'wb') as fd:
                #         for chunk in r.iter_content(chunk_size=128):
                #             fd.write(chunk)
            else:
                if not os.path.exists(os.path.join(config.cache_path,'ViT-H-14_laion2b_s32b_b79k_flavors.pkl')):
                    r = requests.get('https://github.com/theovercomer8/captionr/raw/main/data/ViT-H-14_laion2b_s32b_b79k_flavors.pkl', stream=True)
                    with open(os.path.join(config.cache_path,'ViT-H-14_laion2b_s32b_b79k_flavors.pkl'), 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                if not os.path.exists(os.path.join(config.cache_path,'ViT-H-14_laion2b_s32b_b79k_artists.pkl')):
                    r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_artists.pkl', stream=True)
                    with open(os.path.join(config.cache_path,'ViT-H-14_laion2b_s32b_b79k_artists.pkl'), 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                if not os.path.exists(os.path.join(config.cache_path,'ViT-H-14_laion2b_s32b_b79k_mediums.pkl')):
                    r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_mediums.pkl', stream=True)
                    with open(os.path.join(config.cache_path,'ViT-H-14_laion2b_s32b_b79k_mediums.pkl'), 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                if not os.path.exists(os.path.join(config.cache_path,'ViT-H-14_laion2b_s32b_b79k_movements.pkl')):
                    r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_movements.pkl', stream=True)
                    with open(os.path.join(config.cache_path,'ViT-H-14_laion2b_s32b_b79k_movements.pkl'), 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                if not os.path.exists(os.path.join(config.cache_path,'ViT-H-14_laion2b_s32b_b79k_trendings.pkl')):
                    r = requests.get('https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_trendings.pkl', stream=True)
                    with open(os.path.join(config.cache_path,'ViT-H-14_laion2b_s32b_b79k_trendings.pkl'), 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                


            clip_model_name, clip_model_pretrained_name = config.clip_model_name.split('/', 2)
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                clip_model_name, 
                pretrained=clip_model_pretrained_name, 
                precision='fp16' if config.device == 'cuda' else 'fp32',
                device=config.device,
                jit=False,
                cache_dir=config.clip_model_path
            )
            self.clip_model.to(config.device).eval()
        else:
            self.clip_model = config.clip_model
            self.clip_preprocess = config.clip_preprocess
        self.tokenize = open_clip.get_tokenizer(clip_model_name)

        sites = ['Artstation', 'behance', 'cg society', 'cgsociety', 'deviantart', 'dribble', 'flickr', 'instagram', 'pexels', 'pinterest', 'pixabay', 'pixiv', 'polycount', 'reddit', 'shutterstock', 'tumblr', 'unsplash', 'zbrush central']
        trending_list = [site for site in sites]
        trending_list.extend(["trending on "+site for site in sites])
        trending_list.extend(["featured on "+site for site in sites])
        trending_list.extend([site+" contest winner" for site in sites])

        raw_artists = _load_list(config.data_path, 'artists.txt')
        artists = [f"by {a}" for a in raw_artists]
        artists.extend([f"inspired by {a}" for a in raw_artists])

        self.artists = LabelTable(artists, "artists", self.clip_model, self.tokenize, config)
        self.flavors = LabelTable(_load_list(config.data_path, 'flavors.txt'), "flavors", self.clip_model, self.tokenize, config)
        self.mediums = LabelTable(_load_list(config.data_path, 'mediums.txt'), "mediums", self.clip_model, self.tokenize, config)
        self.movements = LabelTable(_load_list(config.data_path, 'movements.txt'), "movements", self.clip_model, self.tokenize, config)
        self.trendings = LabelTable(trending_list, "trendings", self.clip_model, self.tokenize, config)

        end_time = time.time()
        if not config.quiet:
            logging.info(f"Loaded CLIP model and data in {end_time-start_time:.2f} seconds.")

    def image_to_features(self, image: Image) -> torch.Tensor:
        images = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def filter_similar_inner(self,existing,token):
        if token == '':
            return False
        
        for s in existing:
            if fuzz.ratio(s,token) > self.config.fuzz_ratio:
                return False
        
        return True
    def filter_similar(self,existing_list):
        new_list = []

        for s in existing_list:
            if self.filter_similar_inner(new_list,s):
                new_list.append(s)

        return new_list

    def interrogate_classic(self, caption: str, image: Image, max_flavors: int=3) -> str:
        image_features = self.image_to_features(image)

        if self.config.captionr_config.clip_medium:
            medium = self.mediums.rank(image_features, 1)[0]
        else:
            medium = ''
        if self.config.captionr_config.clip_artist:
            artist = self.artists.rank(image_features, 1)[0]
        else:
            artist = ''
        
        if self.config.captionr_config.clip_trending:
            trending = self.trendings.rank(image_features, 1)[0]
        else:
            trending = ''

        if self.config.captionr_config.clip_movement:
            movement = self.movements.rank(image_features, 1)[0]
        else:
            movement = ''

        if self.config.captionr_config.clip_flavor:
            flaves = ", ".join(self.filter_similar(self.flavors.rank(image_features, max_flavors*2))[:max_flavors])
        else:
            flaves = ''

        if caption.startswith(medium) and medium != '':
            prompt = f"{caption} {artist}, {trending}, {movement}, {flaves}"
        else:
            prompt = f"{caption}, {medium} {artist}, {trending}, {movement}, {flaves}"

        return _truncate_to_fit(prompt, self.tokenize)

    def interrogate_fast(self, caption: str, image: Image, max_flavors: int = 32) -> str:
        image_features = self.image_to_features(image)
        tables = []
        if self.config.captionr_config.clip_artist:
            tables.append(self.artists)
        if self.config.captionr_config.clip_flavor:
            tables.append(self.flavors)
        if self.config.captionr_config.clip_medium:
            tables.append(self.mediums)
        if self.config.captionr_config.clip_movement:
            tables.append(self.movements)
        if self.config.captionr_config.clip_trending:
            tables.append(self.trendings)

        merged = _merge_tables(tables, self.config)
        tops = merged.rank(image_features, max_flavors*4)
        tops = self.filter_similar(tops)[:max_flavors]

        return _truncate_to_fit(caption + ", " + ", ".join(tops), self.tokenize)

    def interrogate(self, caption: str, image: Image, max_flavors: int=32) -> str:
        image_features = self.image_to_features(image)

        if self.config.captionr_config.clip_flavor:
            flaves = self.flavors.rank(image_features, self.config.flavor_intermediate_count*2)
            flaves = self.filter_similar(flaves)[:self.config.flavor_intermediate_count]
        else:
            flaves = ''
        if self.config.captionr_config.clip_medium:
            best_medium = self.mediums.rank(image_features, 1)[0]
        else:
            best_medium = ''
        if self.config.captionr_config.clip_artist:
            best_artist = self.artists.rank(image_features, 1)[0]
        else:
            best_artist = ''
        if self.config.captionr_config.clip_trending:
            best_trending = self.trendings.rank(image_features, 1)[0]
        else:
            best_trending = ''
        
        if self.config.captionr_config.clip_movement:
            best_movement = self.movements.rank(image_features, 1)[0]
        else:
            best_movement = ''

        best_prompt = caption
        best_sim = self.similarity(image_features, best_prompt)

        def check(addition: str) -> bool:
            nonlocal best_prompt, best_sim
            prompt = best_prompt + ", " + addition
            sim = self.similarity(image_features, prompt)
            if sim > best_sim:
                best_sim = sim
                best_prompt = prompt
                return True
            return False

        def check_multi_batch(opts: List[str]):
            nonlocal best_prompt, best_sim
            prompts = []
            for i in range(2**len(opts)):
                prompt = best_prompt
                for bit in range(len(opts)):
                    if i & (1 << bit):
                        prompt += ", " + opts[bit]
                prompts.append(prompt)

            t = LabelTable(prompts, None, self.clip_model, self.tokenize, self.config)
            best_prompt = t.rank(image_features, 1)[0]
            best_sim = self.similarity(image_features, best_prompt)


        check_multi_batch([best_medium, best_artist, best_trending, best_movement])

        extended_flavors = set(flaves)
        for _ in tqdm(range(max_flavors), desc="Flavor chain", disable=self.config.quiet):
            best = self.rank_top(image_features, [f"{best_prompt}, {f}" for f in extended_flavors])
            flave = best[len(best_prompt)+2:]
            if not check(flave):
                break
            if _prompt_at_max_len(best_prompt, self.tokenize):
                break
            extended_flavors.remove(flave)

        return best_prompt

    def rank_top(self, image_features: torch.Tensor, text_array: List[str]) -> str:
        text_tokens = self.tokenize([text for text in text_array]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return text_array[similarity.argmax().item()]

    def similarity(self, image_features: torch.Tensor, text: str) -> float:
        text_tokens = self.tokenize([text]).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = text_features @ image_features.T
        return similarity[0][0].item()


class LabelTable():
    def __init__(self, labels:List[str], desc:str, clip_model, tokenize, config: Config):
        self.chunk_size = config.chunk_size
        self.config = config
        self.device = config.device
        self.embeds = []
        self.labels = labels
        self.tokenize = tokenize

        hash = hashlib.sha256(",".join(labels).encode()).hexdigest()

        cache_filepath = None
        if config.cache_path is not None and desc is not None:
            os.makedirs(config.cache_path, exist_ok=True)
            sanitized_name = config.clip_model_name.replace('/', '_').replace('@', '_')
            cache_filepath = os.path.join(config.cache_path, f"{sanitized_name}_{desc}.pkl")
            if desc is not None and os.path.exists(cache_filepath):
                with open(cache_filepath, 'rb') as f:
                    try:
                        data = pickle.load(f)
                        if data.get('hash') == hash:
                            self.labels = data['labels']
                            self.embeds = data['embeds']
                    except Exception as e:
                        logging.error(f"Error loading cached table {desc}: {e}")

        if len(self.labels) != len(self.embeds):
            self.embeds = []
            chunks = np.array_split(self.labels, max(1, len(self.labels)/config.chunk_size))
            for chunk in tqdm(chunks, desc=f"Preprocessing {desc}" if desc else None, disable=self.config.quiet):
                text_tokens = self.tokenize(chunk).to(self.device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    text_features = clip_model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    text_features = text_features.half().cpu().numpy()
                for i in range(text_features.shape[0]):
                    self.embeds.append(text_features[i])

            if cache_filepath is not None:
                with open(cache_filepath, 'wb') as f:
                    pickle.dump({
                        "labels": self.labels, 
                        "embeds": self.embeds, 
                        "hash": hash, 
                        "model": config.clip_model_name
                    }, f)

        if self.device == 'cpu' or self.device == torch.device('cpu'):
            self.embeds = [e.astype(np.float32) for e in self.embeds]
    
    def _rank(self, image_features: torch.Tensor, text_embeds: torch.Tensor, top_count: int=1) -> str:
        top_count = min(top_count, len(text_embeds))
        text_embeds = torch.stack([torch.from_numpy(t) for t in text_embeds]).to(self.device)
        with torch.cuda.amp.autocast():
            similarity = image_features @ text_embeds.T
        _, top_labels = similarity.float().cpu().topk(top_count, dim=-1)
        return [top_labels[0][i].numpy() for i in range(top_count)]

    def rank(self, image_features: torch.Tensor, top_count: int=1) -> List[str]:
        if len(self.labels) <= self.chunk_size:
            tops = self._rank(image_features, self.embeds, top_count=top_count)
            return [self.labels[i] for i in tops]

        num_chunks = int(math.ceil(len(self.labels)/self.chunk_size))
        keep_per_chunk = int(self.chunk_size / num_chunks)

        top_labels, top_embeds = [], []
        for chunk_idx in tqdm(range(num_chunks), disable=self.config.quiet):
            start = chunk_idx*self.chunk_size
            stop = min(start+self.chunk_size, len(self.embeds))
            tops = self._rank(image_features, self.embeds[start:stop], top_count=keep_per_chunk)
            top_labels.extend([self.labels[start+i] for i in tops])
            top_embeds.extend([self.embeds[start+i] for i in tops])

        tops = self._rank(image_features, top_embeds, top_count=top_count)
        return [top_labels[i] for i in tops]


def _load_list(data_path: str, filename: str) -> List[str]:
    with open(os.path.join(data_path, filename), 'r', encoding='utf-8', errors='replace') as f:
        items = [line.strip() for line in f.readlines()]
    return items

def _merge_tables(tables: List[LabelTable], config: Config) -> LabelTable:
    m = LabelTable([], None, None, None, config)
    for table in tables:
        m.labels.extend(table.labels)
        m.embeds.extend(table.embeds)
    return m

def _prompt_at_max_len(text: str, tokenize) -> bool:
    tokens = tokenize([text])
    return tokens[0][-1] != 0

def _truncate_to_fit(text: str, tokenize) -> str:
    parts = text.split(', ')
    new_text = parts[0]
    for part in parts[1:]:
        if _prompt_at_max_len(new_text + part, tokenize):
            break
        new_text += ', ' + part
    return new_text