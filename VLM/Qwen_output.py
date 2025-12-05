import os
from PIL import Image
import torch
import torch.nn as nn
from transformers import (
    CLIPVisionModel, CLIPImageProcessor,
    AutoTokenizer, AutoModelForCausalLM
)
from torchvision import transforms
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load CLIP vision encoder + processor
clip_name = "openai/clip-vit-large-patch14"
print("Loading CLIP:", clip_name)
vision_encoder = CLIPVisionModel.from_pretrained(clip_name).to(device).eval()
clip_processor = CLIPImageProcessor.from_pretrained(clip_name)

# helper to get CLIP feature (shape: [1, clip_dim])
@torch.no_grad()
def extract_clip_feature(pil_img):
    inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
    outputs = vision_encoder(**inputs)
    feat = outputs.pooler_output  # shape [1, clip_dim]
    return feat

# Define projector architectures (must match your training definitions)
class LinearProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
    def forward(self, x):
        return self.linear(x)

class LowRankProjector(nn.Module):
    def __init__(self, d_in, d_out, r):
        super().__init__()
        self.W1 = nn.Linear(d_in, r, bias=False)
        self.W2 = nn.Linear(r, d_out, bias=False)
    def forward(self, x):
        return self.W2(self.W1(x))

class MLPProjector(nn.Module):
    def __init__(self, d_in, d_out, hidden=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out)
        )
    def forward(self, x):
        return self.net(x)

class GatedProjector(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.Wp = nn.Linear(d_in, d_out)
        self.Wg = nn.Linear(d_in, d_out)
    def forward(self, x):
        gate = torch.sigmoid(self.Wg(x))
        return gate * self.Wp(x)

class LoRAProjector(nn.Module):
    def __init__(self, d_in, d_out, r=8, alpha=16):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.A = nn.Linear(d_in, r, bias=False)
        self.B = nn.Linear(r, d_out, bias=False)
        self.scaling = alpha / r
    def forward(self, x):
        return self.linear(x) + self.B(self.A(x)) * self.scaling

# Helper to instantiate projector by name (and attempt to load weights)
def build_and_load_projector(name, d_in, d_out, r=256, device=device):
    name_lower = name.lower()
    if name_lower.startswith("linear"):
        model = LinearProjector(d_in, d_out)
    elif name_lower.startswith("lowrank"):
        model = LowRankProjector(d_in, d_out, r=r)
    elif name_lower.startswith("mlp"):
        model = MLPProjector(d_in, d_out)
    elif name_lower.startswith("gated"):
        model = GatedProjector(d_in, d_out)
    elif name_lower.startswith("lora"):
        model = LoRAProjector(d_in, d_out, r=min(64, r))
    else:
        raise ValueError("Unknown projector name: " + name)

    # possible filenames to try (common variants)
    candidates = [
        f"{name}_projector.pt",
        f"{name}.pt",
        f"{name.lower()}_projector.pt",
        f"{name.lower()}.pt"
    ]
    loaded = False
    for fname in candidates:
        if os.path.exists(fname):
            state = torch.load(fname, map_location=device)
            try:
                model.load_state_dict(state)
                print(f"Loaded weights for {name} from {fname}")
                loaded = True
                break
            except Exception as e:
                # maybe the saved dict is the full model state_dict with keys
                try:
                    model.load_state_dict(state, strict=False)
                    print(f"Loaded (non-strict) weights for {name} from {fname}")
                    loaded = True
                    break
                except Exception as e2:
                    print(f"Found {fname} but failed to load exactly: {e2}")
    if not loaded:
        print(f"Warning: no weight file found for {name} among {candidates}. Using random init.")
    return model.to(device).eval()

# Load text-only Qwen2 (or other text-only model) and tokenizer
text_model_name = "Qwen/Qwen2-1.5B-Instruct"  # change if you prefer another text-only Qwen
print("Loading text LM:", text_model_name)
tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=False)
text_model = AutoModelForCausalLM.from_pretrained(text_model_name).to(device).eval()

# detect LM token embedding dimension
lm_embed = text_model.get_input_embeddings()
lm_embed_dim = lm_embed.weight.shape[1]
print("LM embedding dim:", lm_embed_dim)

# Prepare projectors list (names must match how you saved)
projector_names = ["Linear", "LowRank", "MLP", "Gated", "LoRA"]
projectors = {}
# CLIP feature dim:
clip_dim = vision_encoder.pooler_dim if hasattr(vision_encoder, "pooler_dim") else vision_encoder.config.projection_dim if hasattr(vision_encoder.config, "projection_dim") else vision_encoder.pooler_output.shape[1] if False else vision_encoder.config.hidden_size
# safer: inspect a sample
with torch.no_grad():
    dummy = torch.randn(1, 3, 224, 224).to(device)
    out = clip_processor(images=Image.new("RGB", (224,224)), return_tensors="pt").to(device)
# Instead just infer by doing a real forward on a small tensor:
with torch.no_grad():
    sample_inputs = clip_processor(images=Image.new("RGB",(224,224)), return_tensors="pt").to(device)
    sample_out = vision_encoder(**sample_inputs)
    clip_dim = sample_out.pooler_output.shape[1]
print("Detected CLIP feature dim:", clip_dim)

for name in projector_names:
    p = build_and_load_projector(name, d_in=clip_dim, d_out=lm_embed_dim, r=256, device=device)
    projectors[name] = p

# Utility: prepend projector embedding to token embeddings and generate
@torch.no_grad()
def generate_with_projector(p_model, pil_img, question, num_vis_tokens=1, max_new_tokens=80):
    # 1) get CLIP feature
    clip_feat = extract_clip_feature(pil_img)  # [1, clip_dim]
    # 2) get projector feature mapped to LM embedding dim
    proj_feat = p_model(clip_feat.to(device))   # [1, lm_embed_dim]
    # 3) decide how many visual tokens to produce
    #    simple: repeat the vector num_vis_tokens times -> [1, num_vis_tokens, lm_embed_dim]
    vis_embeds = proj_feat.unsqueeze(1).repeat(1, num_vis_tokens, 1).to(device)
    # 4) tokenize the question text
    toks = tokenizer(question, return_tensors="pt").to(device)
    input_ids = toks["input_ids"]
    attn = toks.get("attention_mask", torch.ones_like(input_ids)).to(device)
    # 5) get token embeddings from LM embed table
    token_embeds = text_model.get_input_embeddings()(input_ids)  # [1, seq_len, lm_embed_dim]
    # 6) concat: [vis_embeds, token_embeds]
    inputs_embeds = torch.cat([vis_embeds, token_embeds], dim=1)
    # 7) build attention mask (1s for vis tokens)
    vis_mask = torch.ones((inputs_embeds.size(0), vis_embeds.size(1)), dtype=attn.dtype).to(device)
    attention_mask = torch.cat([vis_mask, attn], dim=1)
    # 8) generate using inputs_embeds (some HF models require passing attention_mask)
    outputs = text_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic; change if you want diversity
        temperature=0.2,
        eos_token_id=tokenizer.eos_token_id
    )
    # decode
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Demo: use uploaded image at /content/Confusing_fig.jpg
img_path = "/content/Confusing_fig.jpg"
if not os.path.exists(img_path):
    print("ERROR: demo image not found at", img_path)
else:
    demo_img = Image.open(img_path).convert("RGB")
    demo_img = demo_img.resize((224,224))
    prompt = "Describe this image."

    print("\n=== Qwen2 (text-only) with different projectors ===\n")
    for name, proj in projectors.items():
        print(f"--- {name} projector ---")
        try:
            ans = generate_with_projector(proj, demo_img, prompt, num_vis_tokens=1, max_new_tokens=80)
            print(ans)
        except Exception as e:
            print("Failed for", name, ":", e)

    print("\nDone.")
