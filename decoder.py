import os
import sys
from flow_inference import AudioDecoder
import torchaudio
import torch
import uuid
import wandb

sys.path.insert(0, "./GLM-4-Voice/cosyvoice")
sys.path.insert(0, "./GLM-4-Voice/third_party/Matcha-TTS")
class GLMvocoder:
    def __init__(self, device):
        base_path = 'GLM-4-Voice'
        flow_path = "glm-4-voice-decoder"
        flow_config = os.path.join(base_path, flow_path, "config.yaml")
        flow_checkpoint = os.path.join(base_path, flow_path, 'flow.pt')
        hift_checkpoint = os.path.join(base_path, flow_path, 'hift.pt')
        self.device = device
        self.audio_decoder = AudioDecoder(config_path=flow_config, flow_ckpt_path=flow_checkpoint,
                                    hift_ckpt_path=hift_checkpoint,
                                    device=device)
        
    def token_to_wav(self, tts_tokens):
        this_uuid = str(uuid.uuid4())
        prompt_speech_feat = torch.zeros(1, 0, 80).to(self.device)
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(self.device)
        is_finalize = False
        with torch.no_grad():
            tts_speech, tts_mel = self.audio_decoder.token2wav(tts_tokens, uuid=this_uuid,
                                                                        prompt_token=flow_prompt_speech_token.to(self.device),
                                                                        prompt_feat=prompt_speech_feat.to(self.device),
                                                                        finalize=is_finalize)
        return tts_speech.cpu()
