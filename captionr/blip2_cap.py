from lavis.models import load_model_and_preprocess
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
        self.model, self.processor, _ = load_model_and_preprocess(
            name=name, model_type=model_type, is_eval=True, device=device
        )
        
    
    
    def caption(self,img:Image) -> str:
        image = self.processor["eval"](img).unsqueeze(0).to(self.device)
        return self.model.generate({"image": image})
     