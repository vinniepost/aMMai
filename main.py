"""
Main file for the project, This is the AI-agent that will use the models from the other files
"""
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import requests

# Step 1: Generate a descriptive feature summary from BLIP
def extract_image_features(image_url, processor=None, blip_model=None):
    
    # Load and preprocess the image
    image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    
    # Generate initial description
    with torch.no_grad():
        feature_ids = blip_model.generate(**inputs)
        feature_summary = processor.decode(feature_ids[0], skip_special_tokens=True)
    return feature_summary

# Step 2: Use GPT-2 to expand on the BLIP description
# Final version of generate_multimodal_caption function
def generate_multimodal_caption(image_url, max_length=100, tokenizer=None, gpt2_model=None, processor=None, blip_model=None):
    # Step 1: Get the initial description from BLIP
    feature_summary = extract_image_features(image_url, processor=processor, blip_model=blip_model)
    prompt = f"<|startoftext|>Description: {feature_summary}. Expand with more details:"

    # Encode prompt for GPT-2
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    attention_mask = torch.ones_like(input_ids)

    # Set eos_token_id as pad_token_id
    gpt2_model.config.pad_token_id = tokenizer.eos_token_id

    # Step 2: Generate a caption using sampling with repetition penalty
    with torch.no_grad():
        output = gpt2_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.8,         # Slightly lower temperature for balance
            top_k=40,                # Adjusted sampling
            top_p=0.9,
            repetition_penalty=1.2,  # Penalty to discourage repeated phrases
            early_stopping=True
        )
        caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return caption




def main(image_url="https://lp-cms-production.imgix.net/2024-05/GettyImages-1303030943.jpg?w=1440&h=810&fit=crop&auto=format&q=75"):
    # Load the BLIP model for image feature extraction
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load GPT-2 for text generation
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    caption = generate_multimodal_caption(image_url, tokenizer=tokenizer, gpt2_model=gpt2_model, processor=processor, blip_model=blip_model)
    print("Generated Caption:", caption)

if __name__ == "__main__":
    import sys
    image_url = sys.argv[1] if len(sys.argv) > 1 else "https://lp-cms-production.imgix.net/2024-05/GettyImages-1303030943.jpg?w=1440&h=810&fit=crop&auto=format&q=75"
    main(image_url)
