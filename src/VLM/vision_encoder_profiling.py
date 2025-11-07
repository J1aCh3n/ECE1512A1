from transformers import CLIPVisionModel, CLIPImageProcessor
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Load ViT-L/14 encoder
model_name = "openai/clip-vit-large-patch14"
vision_tower = CLIPVisionModel.from_pretrained(model_name).to("cuda").half()
image_processor = CLIPImageProcessor.from_pretrained(model_name)
vision_tower.eval()

# LLaVA style projection layer
vision_projector = torch.nn.Linear(1024, 4096).half().cuda()

# Dummy input image (224x224 RGB)
x = torch.randn(1, 3, 224, 224).half().cuda()

# Profile forward pass
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("vision_forward"):
        feats = vision_tower(x).pooler_output
        proj = vision_projector(feats)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
