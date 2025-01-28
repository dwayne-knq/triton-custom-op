import torch
import clip
from PIL import Image
from pathlib import Path


# 모델과 텍스트 인코더를 로드합니다.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지를 전처리하고 텐서로 변환합니다.
image_path = "/path/to/your/image"
if Path(image_path).exists():
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 텍스트를 텐서로 변환합니다.
    text = clip.tokenize([
        "a photo of a owl",
        "a photo of a owl with a camera",
        "a photo of a camera",
        "a photo of glasses",
        "a photo of a bag"]
    ).to(device)

    # 이미지와 텍스트를 인코딩합니다.
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # 이미지와 텍스트 간의 유사도를 계산합니다.
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # 결과를 출력합니다.
    print(similarity)
else:
    print("There is no file")
