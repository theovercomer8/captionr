from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import open_clip
import torch
from blip.models.blip import blip_decoder, BLIP_Decoder
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import os
import inspect

BLIP_MODELS = {
    'base': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth',
    'large': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
}


class BLIP:
    blip_model: BLIP_Decoder = None
    device = None

    blip_image_eval_size: int = 384
    blip_model_type: str = 'large' # choose between 'base' or 'large'

    def __init__(self, device, model_name=None, beams=8, blip_max=150, blip_min=0) -> None:
        if model_name is not None:
            self.model_name = model_name
        self.device = device
        self.blip_max = blip_max
        self.blip_min = blip_min
        self.beams = beams

        blip_path = os.path.dirname(inspect.getfile(blip_decoder))
        configs_path = os.path.join(os.path.dirname(blip_path), 'configs')
        med_config = os.path.join(configs_path, 'med_config.json')
        blip_model = blip_decoder(
            pretrained=BLIP_MODELS[self.blip_model_type],
            image_size=self.blip_image_eval_size, 
            vit=self.blip_model_type, 
            med_config=med_config
        )
        blip_model.eval()
        blip_model = blip_model.to(self.device)
        self.blip_model = blip_model
        
    
    
    def caption(self,img:Image) -> str:
        size = self.blip_image_eval_size
        gpu_image = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            caption = self.blip_model.generate(
                gpu_image, 
                sample=False, 
                num_beams=self.beams, 
                max_length=self.blip_max, 
                min_length=self.blip_min
            )
        return caption[0]