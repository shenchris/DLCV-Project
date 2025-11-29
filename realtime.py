# # usbipd bind  --busid  4-1
# # usbipd attach --wsl --busid 4-1
# # usbipd detach --all
import torch 
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sys
import os
from datasets import S2T_Dataset_online
import cv2
import numpy as np
from pathlib import Path
import utils
from models import Uni_Sign
from rtmlib import Wholebody
from collections import deque
import threading  # <--- NEW: For background processing
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
You convert sign language gloss into a natural English sentence.
Input: gloss tokens, separated by spaces, in all caps.
Output: ONE fluent English sentence. No explanation.
"""

def gloss_to_sentence_from_list(tokens):
    gloss = " ".join(tokens)
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": gloss},
        ],
        max_output_tokens=64,
    )
    return response.output[0].content[0].text



# --- CONFIGURATION ---
MAX_LEN = 30          
PREDICT_EVERY = 5     
DEVICE_POSE = "cpu"   
DEVICE_MODEL = "cuda" 
STABILITY_COUNT = 3   # <--- NEW: Require 3 consecutive same predictions

# --- SETUP PATHS & MODEL ---
sys.path.insert(0, str(Path('.').resolve()))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Load Uni-Sign (Translation) Model
print("Loading Uni-Sign Model (GPU)...")
checkpoint_path = "out/stage3_finetuning/best_checkpoint.pth"
seed = 42
parser = utils.get_args_parser()
args = parser.parse_args([]) 
args.seed = seed
args.finetune = checkpoint_path
args.output_dir = './output'
args.rgb_support = False
utils.set_seed(args.seed)

model = Uni_Sign(args=args)
model.cuda()

if args.finetune and os.path.exists(args.finetune):
    print(f"Loading checkpoint...")
    state_dict = torch.load(args.finetune, map_location='cpu')['model']
    model.load_state_dict(state_dict, strict=True)

model.eval()
model.to(torch.bfloat16)
print("Uni-Sign Loaded.")

# 2. Load Pose Model (Global - Loaded ONCE)
print("Loading Pose Model (CPU)...")
pose_model = Wholebody(
    to_openpose=False,
    mode="lightweight",
    backend="onnxruntime",
    device=DEVICE_MODEL
)
print("Pose Model Loaded.")

# --- OPTIMIZED FUNCTIONS ---

def get_pose_fast(frame):
    # (保持不變)
    keypoints, scores = pose_model(frame)
    h, w, _ = frame.shape
    norm_kps = keypoints / np.array([w, h])[None, None]
    return {'keypoints': [norm_kps], 'scores': [scores]}

# --- MODIFIED INFERENCE FUNCTION ---
def run_inference_bg(pose_data, model, args, result_container, prediction_history):
    """
    Runs in a background thread.
    Args:
        result_container: List to update final display text.
        prediction_history: Deque storing recent raw predictions.
    """
    global words
    try:
        # Prepare Dataset
        online_data = S2T_Dataset_online(args=args)
        online_data.rgb_data = None 
        online_data.pose_data = pose_data

        loader = DataLoader(
            online_data, batch_size=1, 
            collate_fn=online_data.collate_fn, sampler=torch.utils.data.SequentialSampler(online_data)
        )

        with torch.no_grad():
            for src_input, tgt_input in loader:
                # Move to GPU
                for k, v in src_input.items():
                    if isinstance(v, torch.Tensor):
                        src_input[k] = v.to(torch.bfloat16).cuda()

                stack_out = model(src_input, tgt_input)
                output = model.generate(stack_out, max_new_tokens=50, num_beams=1)
                
                # Decode
                tokenizer = model.mt5_tokenizer
                pad_id = tokenizer.eos_token_id
                pad = torch.ones(150 - len(output[0]), device="cuda") * pad_id
                raw_sent = tokenizer.decode(torch.cat([output[0], pad.long()]), skip_special_tokens=True)
                
                print(f"Raw Prediction: {raw_sent}") # Debug print
                # --- NEW: STABILITY LOGIC ---
                # 1. Add new prediction to history
                prediction_history.append(raw_sent)

                # 2. Check if buffer is full and all elements are identical
                if len(prediction_history) == prediction_history.maxlen:
                    # Convert to set to count unique elements. If len(set) == 1, all are same.
                    if len(set(prediction_history)) == 1:
                        final_text = prediction_history[0]
                        
                        # Only update if it's different from what's currently shown (optional optimization)
                        if result_container[0] != final_text:
                            
                            if final_text == "yes":
                                if len(words)> 0:
                                    print(gloss_to_sentence_from_list(words))
                                words = []
                            result_container[0] = final_text
                            words.append(final_text)
                            print(f">>> CONFIRMED OUTPUT: {final_text}")
                    else:
                        print(f"--- Unstable: {list(prediction_history)}")

    except Exception as e:
        print(f"Inference Error: {e}")

# --- MAIN LOOP ---

cap = cv2.VideoCapture(0)
# (Camera setup code...)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

kps_buffer = deque(maxlen=MAX_LEN)
scores_buffer = deque(maxlen=MAX_LEN)

# --- NEW BUFFER FOR STABILITY ---
# 用來儲存最近 3 次的預測結果
raw_prediction_buffer = deque(maxlen=STABILITY_COUNT) 

frame_counter = 0
current_translation = ["Waiting..."] 
is_inferencing = False

print(f"Starting Real-time System (Stability Count: {STABILITY_COUNT})...")

try:
    words = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_counter += 1

        # 1. Fast Pose Extraction
        pose_res = get_pose_fast(frame)
        kps_buffer.append(pose_res['keypoints'][0])
        scores_buffer.append(pose_res['scores'][0])

        # 2. Smart Inference Trigger
        if len(kps_buffer) == MAX_LEN and (frame_counter % PREDICT_EVERY == 0):
            # Check if previous thread is alive
            if 't' in locals() and t.is_alive():
                pass # GPU busy, skip
            else:
                # Create snapshot
                data_snapshot = {
                    'keypoints': list(kps_buffer), 
                    'scores': list(scores_buffer)
                }
                
                # Start thread (Pass the NEW raw_prediction_buffer)
                t = threading.Thread(
                    target=run_inference_bg, 
                    args=(data_snapshot, model, args, current_translation, raw_prediction_buffer)
                )
                t.start()

        # 3. Visualization
        cv2.rectangle(frame, (0, 440), (640, 480), (0, 0, 0), -1)
        
        # 根據是否穩定顯示不同顏色 (可選)
        # 這裡我們只顯示 current_translation，它只會在穩定時被更新
        cv2.putText(frame, current_translation[0], (10, 470), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Real Time SLT", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    print("Stopping...")
finally:
    cap.release()
    cv2.destroyAllWindows()