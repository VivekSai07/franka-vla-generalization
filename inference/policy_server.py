import argparse
import base64
import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

app = FastAPI()
policy = None
preprocessor = None
postprocessor = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class InferenceRequest(BaseModel):
    front_image_b64: str
    wrist_image_b64: str
    top_image_b64: str  
    state: list[float]
    instruction: str

def decode_image(b64_str):
    """Decodes Base64 JPEG back into a LeRobot-ready Tensor"""
    image_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(image_data)).convert("RGB")
    img_arr = np.array(img)
    
    # Convert (H, W, C) uint8 to (1, C, H, W) float32 [0, 1]
    img_tensor = torch.from_numpy(img_arr).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    return img_tensor.to(DEVICE)

@app.post("/predict")
def predict(req: InferenceRequest):
    # Map cameras exactly as defined in your --rename_map during training
    raw_obs_dict = {
        "observation.images.camera1": decode_image(req.front_image_b64),
        "observation.images.camera2": decode_image(req.wrist_image_b64),
        "observation.images.camera3": decode_image(req.top_image_b64),
        "observation.state": torch.tensor(req.state, dtype=torch.float32).unsqueeze(0).to(DEVICE),
        "task": [req.instruction]
    }
    
    with torch.inference_mode():
        # 1. Preprocess: Tokenizes text, normalizes images & state
        processed_obs = preprocessor(raw_obs_dict)
        
        # 2. Predict: Feed the processed data to SmolVLA
        action_tensor = policy.select_action(processed_obs)
        
        # 3. Postprocess: LeRobot expects just the tensor, not a dict!
        unnormalized_action = postprocessor(action_tensor)
        final_action = unnormalized_action.squeeze(0).cpu().numpy().tolist()
        
    return {"action": final_action}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=str, required=True, help="Path to your trained model")
    args = parser.parse_args()

    print(f"🤖 Loading trained SmolVLA policy from: {args.policy_path}")
    
    # Load Model
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.eval()
    policy.to(DEVICE)
    
    print("⚙️ Loading Data Preprocessors & Postprocessors...")
    
    # Load the pipelines (Added device=DEVICE to safely override hardcoded configs)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, 
        pretrained_path=args.policy_path,
        device=DEVICE
    )
    
    print("✅ Policy & Processors loaded! Starting inference server on http://localhost:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")