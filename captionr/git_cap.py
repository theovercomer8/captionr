from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

class Git:
    model_name = "microsoft/git-large-r-textcaps"
    device = None
    max_length:int

    def __init__(self, device, model_name=None, max_length=0) -> None:
        if model_name is not None:
            self.model_name = model_name
        self.device = device
        self.max_length = max_length

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
    
    
    def caption(self,img:Image) -> str:
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values

        pixel_values = pixel_values.to(self.device)
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=self.max_length if self.max_length != 0 else 9999)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_caption