import torch
import clip

# CLIP 모델 로드
device = "cpu"  # TorchScript 변환 시 CPU에서 실행하는 것이 안전함
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()  # 추론 모드 설정

dummy_input = torch.randn(1, 3, 224, 224, device=device)

# TorchScript 변환 (Tracing 사용)
scripted_model = torch.jit.trace(model.visual, dummy_input)

scripted_model_path = "clip_model.pt"
torch.jit.save(scripted_model, scripted_model_path)

print(f"TorchScript 모델 저장 완료: {scripted_model_path}")