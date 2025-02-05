import torch
from eva_clip import create_model_and_transforms, get_tokenizer
from PIL import Image

# model_name = 'EVA01-CLIP-g-14'
# model_name = 'EVA01-CLIP-g-14-plus'
# model_name = 'EVA02-CLIP-B-16'
# model_name = 'EVA02-CLIP-bigE-14'
# model_name = 'EVA02-CLIP-bigE-14-plus'
# model_name = 'EVA02-CLIP-L-14'
# model_name = 'EVA02-CLIP-L-14-336'

pretrained = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

image_path = "./EVA/assets/eva_pipeline.png"
caption = ["a diagram", "a dog", "a cat"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
tokenizer = get_tokenizer(model_name)
model = model.to(device)

image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[0.8275, 0.1372, 0.0352]]