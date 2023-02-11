from PIL import Image
import open_clip
import torch


class Coca:
    model_name = "coca_ViT-L-14/mscoco_finetuned_laion2B-s13B-b90k"
    device = None
    max_length:int

    def __init__(self, device, model_name=None, max_length=0) -> None:
        if model_name is not None:
            self.model_name = model_name
        self.device = device
        self.max_length = max_length

        self.model, _, self.processor = open_clip.create_model_and_transforms(
            model_name=self.model_name.split('/')[0],
            pretrained=self.model_name.split('/')[1]
        )
        self.model.to(device)
        
    
    
    def caption(self,img:Image) -> str:
        im = self.processor(img).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            generated = self.model.generate(im)

        generated_caption = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        return generated_caption