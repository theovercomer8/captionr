import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

class BLIP2:
    device = None
    max_length:int

    def __init__(self, device, model_name:str=None, max_length=0) -> None:
        if model_name is not None:
            self.model_name = model_name
        self.device = device
        self.max_length = max_length
        name, model_type = self.model_name.split('/')
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch.float16).to(self.device)
        
    
    
    def caption(self,img:Image) -> str:

        inputs = self.processor(images=img, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
    
    def question(self,img:Image,question:str) -> str:
        q = f"Question: {question} Answer:"
        reply = self.model.generate({"image": img, "prompt": q})
        if len(reply) > 0:
            reply = reply[0]
        reply = reply.replace(q,'')
        return reply