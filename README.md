# captionr
COCA/GIT/BLIP/CLIP Caption tool as a Colab notebook and Python script

## New Feature! 
Similarity matching for tags will now eliminate dupliciate sounding tags resulting from CLIP interrogation. This affects interrogation when using the interrogate_classic and interrogate_fast methods. In addition, the uniquify_tags feature will also eliminate any similar tags. This can be useful when running CLIP flavors on existing caption files that include tags.

### CHANGELOG:
* Notebook v0.3.0 - Added similarity matching UI support
* Captionr.py v0.3.0 - Added similarity matching when uniquifying tags
* Notebook v0.2.4 - Added caption editor widget
* Captionr.py v0.2.1 - Added experimental BLIP2 questions support
* Notebook v0.2.3 - Added experimental BLIP2 questions support
* Notebook v0.2.2 - Added BLIP2 support
* Captionr.py v0.2.0 - Added BLIP2 support
* Notebook v0.2.1 - Added GDrive cell
* Notebook v0.2.0 - Refactored to support new captionr.py module
* Captionr.py v0.1.0 - Initial release of local script

Feature requests? Support? Find me on the [IlluminatiAI Discord](https://discord.gg/HqdffGgeBa)

## To use the notebook:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theovercomer8/captionr/blob/main/TO8_Captionr.ipynb)

### Wizard interface
<img width="1029" alt="image" src="https://user-images.githubusercontent.com/122644869/217708333-c80320ba-3351-4bd4-8da5-e6c14037defc.png">

### Live caption previews
<img width="336" alt="image" src="https://user-images.githubusercontent.com/122644869/217713111-9325216e-d6a2-43a5-8e1a-d56243aca5cb.png">

## To use the script:
The salesforce-lavis module required to use BLIP2 is not compatible with Python 3.9 or higher. It is recommended that you install Python 3.8 if you are currently using 3.9 or higher. It is recommended to use a tool like [Pyenv](https://github.com/pyenv/pyenv) to install/manage older Python versions if you are not experienced with doing so.

**Warning:** Using BLIP2 will take up a large amount of disk space. Users have reported as much as 45GB. In addition, the VRAM requirement is minimum 24GB.

Trying to follow the below instructions with versions newer than 3.8 will cause errors.

- Clone the repo
`git clone https://github.com/theovercomer8/captionr`

- Install requirements
`pip install -r requirements.txt`

- Execute

Sample arguments that will execute a GIT pass, followed by Coca if the fail phrases are triggered, and append CLIP flavors to multiple datasets designed for a SD 2.x model:

`python captionr.py e:\data\set1 e:\data\set2 --existing=skip --cap_length=300 --git_pass --coca_pass --model_order='git,coca' --clip_model_name=ViT-H-14/laion2b_s32b_b79k --clip_flavor --clip_max_flavors=32 --clip_method=interrogate_fast --fail_phrases="a sign that says,writing that says,that says,with the word" --uniquify_tags --prepend_text="a photo of " --device=cuda --extension=txt`

`python captionr.py --help`

```usage: Captionr [OPTIONS] [FOLDER]...

Caption a set of images

positional arguments:
  folder                One or more folders to scan for iamges. Images should be jpg/png.

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --output OUTPUT       Output to a folder rather than side by side with image files
  --existing {skip,ignore,copy,prepend,append}
                        Action to take for existing caption files (default: skip)
  --cap_length CAP_LENGTH
                        Maximum length of caption. (default: 0)
  --git_pass            Perform a GIT model pass
  --coca_pass           Perform a Coca model pass
  --blip_pass           Perform a BLIP model pass
  --model_order MODEL_ORDER
                        Perform captioning/fallback using this order (default: coca,git,blip)
  --use_blip2           Uses BLIP2 for BLIP pass. Only activated when --blip_pass also specified
  --blip2_model {blip2_t5/pretrain_flant5xxl,blip2_opt/pretrain_opt2.7b,blip2_opt/pretrain_opt6.7b,blip2_opt/caption_coco_opt2.7b,blip2_opt/caption_coco_opt6.7b,blip2_t5/pretrain_flant5xl,blip2_t5/caption_coco_flant5xl}
                        Specify the BLIP2 model to use
  --blip2_question_file BLIP2_QUESTION_FILE
                        Specify a question file to use to query BLIP2 and add answers as tags
  --blip_beams BLIP_BEAMS
                        Number of BLIP beams (default: 64)
  --blip_min BLIP_MIN   BLIP min length (default: 30)
  --blip_max BLIP_MAX   BLIP max length (default: 75)
  --clip_model_name {ViT-H-14/laion2b_s32b_b79k,ViT-L-14/openai,ViT-bigG-14/laion2b_s39b_b160k}
                        CLIP model to use. Use ViT-H for SD 2.x, ViT-L for SD 1.5 (default: ViT-H-14/laion2b_s32b_b79k)
  --clip_flavor         Add CLIP Flavors
  --clip_max_flavors CLIP_MAX_FLAVORS
                        Max CLIP Flavors (default: 8)
  --clip_artist         Add CLIP Artists
  --clip_medium         Add CLIP Mediums
  --clip_movement       Add CLIP Movements
  --clip_trending       Add CLIP Trendings
  --clip_method {interrogate,interrogate_fast,interrogate_classic}
                        CLIP method to use
  --fail_phrases FAIL_PHRASES
                        Phrases that will fail a caption pass and move to the fallback model. (default: "a sign that says,writing that says,that says,with the word")
  --ignore_tags IGNORE_TAGS
                        Comma separated list of tags to ignore
  --find FIND           Perform find and replace with --replace REPLACE
  --replace REPLACE     Perform find and replace with --find FIND
  --folder_tag          Tag the image with folder name
  --folder_tag_levels FOLDER_TAG_LEVELS
                        Number of folder levels to tag. (default: 1)
  --folder_tag_stop FOLDER_TAG_STOP
                        Do not tag folders any deeper than this path. Overrides --folder_tag_levels if --folder_tag_stop is shallower
  --uniquify_tags       Ensure tags are unique
  --fuzz_ratio FUZZ_RATIO
                        Sets the similarity ratio allowed for tags when uniquifying (default: 60.0)
  --prepend_text PREPEND_TEXT
                        Prepend text to final caption
  --append_text APPEND_TEXT
                        Append text to final caption
  --preview             Do not write to caption file. Just displays preview in STDOUT
  --use_filename        Read the existing caption from the filename, stripping all special characters/numbers
  --device {cuda,cpu}   Device to use. (default: cuda)
  --extension {txt,caption}
                        Caption file extension. (default: txt)
  --quiet
  --debug
  ```


## Special Thanks
* @cacoe for the inception of the idea. Be sure to check out his new [IlluminatiAI model v1.1](https://civitai.com/models/11193/illuminati-diffusion-v11). It slaps.
* @Kaz, @jvkas, and @PeePa for help with testing
* @Stille Willem and @NimbusFPV for the blunts

<a href="https://ko-fi.com/theovercomer8"><img width="145" alt="blunt" src="https://user-images.githubusercontent.com/122644869/218321249-d343e3b2-1600-466b-aac3-968316a75033.png"></a>

