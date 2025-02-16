import os
import sys
from speech_tokenizer.utils import extract_speech_token, WhisperVQEncoder
from transformers import WhisperFeatureExtractor
from flow_inference import AudioDecoder
from datasets import Dataset, Audio
import pandas as pd
import os
import jsonlines
from pathlib import Path
from tqdm import tqdm
sys.path.insert(0, "./cosyvoice")
sys.path.insert(0, "./third_party/Matcha-TTS")
# Load the tokenizer model and feature extractor
device='cuda'
tokenizer_path = './glm-4-voice-tokenizer'
flow_path = "./glm-4-voice-decoder"
flow_config = os.path.join(flow_path, "config.yaml")
flow_checkpoint = os.path.join(flow_path, 'flow.pt')
hift_checkpoint = os.path.join(flow_path, 'hift.pt')
whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to('cuda')
feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
                                hift_ckpt_path=hift_checkpoint,
                                device=device)

with jsonlines.open('text_conversion/TW_Attraction_dataset.jsonl') as reader:
    data = list(reader)
    
def extract_chat_id(filename):
    number_part = filename[:6]
    return int(number_part)

def extract_user_id(filename):
     # 使用 split 分割檔名
    parts = filename.split('_')
    
    # 取得 User_X 部分
    user_part = f"{parts[1]}_{parts[2].split('.')[0]}"  # 得到 "User_0"
    
    # 取得數字並加1
    user_num = int(parts[2].split('.')[0])  # 從 "0.wav" 中取得 "0"
    new_user_num = user_num + 1
    
    # 組合新的 User 部分
    new_user_part = f"User_{new_user_num}"
    new_label_part = f"Machine_{new_user_num}"
    
    return new_user_part, new_label_part

def get_machine_audio_path(chat_id, label_user_id, machine_dir):
    # 將 chat_id 轉換為六位數字字串
    formatted_chat_id = str(chat_id).zfill(6)
    
    # 建立機器回應的檔名格式
    parts = label_user_id.split('_')
    machine_num = str(int(parts[-1])-1)
    machine_filename = f"{formatted_chat_id}_Machine_{machine_num}.wav"
    # 組合完整路徑
    machine_path = os.path.join(machine_dir, machine_filename)
    return machine_path if os.path.exists(machine_path) else None


def create_audio_dataset(audio_dir):
    audio_files = []
    machine_dir = "/work/twsurgy726/machine/TW_Attraction_dataset"
    
    # 先收集所有 wav 檔案並排序
    all_wav_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.wav'):
                all_wav_files.append((file, os.path.join(root, file)))
    
    # 根據檔名排序
    all_wav_files.sort(key=lambda x: x[0])  # x[0] 是檔名
    
    for file, full_path in tqdm(all_wav_files):
            
        chat_id = extract_chat_id(file)
        input_user_id, label_user_id = extract_user_id(file)

        machine_path = get_machine_audio_path(chat_id, label_user_id, machine_dir)
        if machine_path is None:
            continue
        speech_unit = extract_speech_token(whisper_model, feature_extractor, [machine_path])[0]

        try:
            audio_files.append({
                'audio_path': full_path,
                'input_text': data[chat_id][input_user_id],
                'label_text': data[chat_id][label_user_id],
                'label_codec': speech_unit
            })
        except:
            continue
        
    
    df = pd.DataFrame(audio_files)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column('audio_path', Audio())
    
    return dataset

if __name__ == "__main__":
    ds = create_audio_dataset(Path("/work/twsurgy726/user/TW_Attraction_dataset/"))
    ds.save_to_disk("./GLM-4_unit_local_ds")