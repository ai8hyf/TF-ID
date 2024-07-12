import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

model_id = "yifeihu/TF-ID-large" # recommended: use large models for better performance
# model_id = "yifeihu/TF-ID-base"
# model_id = "yifeihu/TF-ID-large-no-caption" # recommended: use large models for better performance
# model_id = "yifeihu/TF-ID-base-no-caption"

model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

prompt = "<OD>"

# TF-ID models were trained on digital pdf papers
image_url = "./sample_images/arxiv_2305_10853_5.png"
image = Image.open(requests.get(image_url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

print(parsed_answer)

# to visualize the generated answer, check out this colab example from Florence 2 repo: https://huggingface.co/microsoft/Florence-2-large/blob/main/sample_inference.ipynb
