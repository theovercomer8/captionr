from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
class BLIP2:
    device = None
    max_length:int

    def __init__(self, device, model_name:str=None, max_length=0) -> None:
        if model_name is not None:
            self.model_name = model_name
        self.device = device
        self.max_length = max_length
        name, model_type = self.model_name.split('/')
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.model_name,
                                                      device_map='auto',
                                                      load_in_8bit=True)
        
        
    
    def caption(self, img: Image,
                     decoding_method: str, temperature: float,
                     length_penalty: float, repetition_penalty: float, max_length: int, min_length: int, num_beams: int) -> str:

        inputs = self.processor(images=img,
                        return_tensors='pt').to(self.device, torch.float16)
        generated_ids = self.model.generate(
            pixel_values=inputs.pixel_values,
            do_sample=decoding_method == 'Nucleus sampling',
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            top_p=0.9)
        result = self.processor.batch_decode(generated_ids,
                                        skip_special_tokens=True)[0].strip()
        return result
    
    
    
    def question(self, img: Image, text: str,
                    decoding_method: str, temperature: float,
                    length_penalty: float, repetition_penalty: float, max_length: int, min_length: int, num_beams: int) -> str:

        inputs = self.processor(images=img, text=text,
                        return_tensors='pt').to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs,
                                    do_sample=decoding_method ==
                                    'Nucleus sampling',
                                    temperature=temperature,
                                    length_penalty=length_penalty,
                                    repetition_penalty=repetition_penalty,
                                    max_length=max_length,
                                    min_length=min_length,
                                    num_beams=num_beams,
                                    top_p=0.9)
        result = self.processor.batch_decode(generated_ids,
                                        skip_special_tokens=True)[0].strip()
        return result
   

     