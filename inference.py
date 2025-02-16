import torch
import soundfile as sf
from model import SpeechUnitModel
from transformers import AutoTokenizer, AutoModelForCausalLM, MimiModel, AutoFeatureExtractor


def inference(input_text, model, mimi_vocoder, max_length=150):
    # process input_text into index
    data_input = [{'role': 'assistant', "content": input_text}]
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    input_ids = tokenizer.apply_chat_template(
            data_input,
            add_generation_prompt=False,
            return_tensors="pt"
        )
    # delete template index
    input_ids = input_ids[:, 5:]
    _, seq_length = input_ids.shape
    input_ids = input_ids.to('cuda')
    
    audio_ids = torch.full((8, 1), 2048).to('cuda')
    add_tensor = torch.zeros_like(audio_ids).to('cuda')
    for i in range(1, 8):
        add_tensor[i, :] = 2050 * (i)
    with torch.no_grad():
        for i in range(1, max_length):
            if i > seq_length:
                padding = torch.full((1,1), tokenizer.eos_token_id, dtype=torch.long).to('cuda')
                input_ids = torch.cat([input_ids, padding], dim=1)
            outputs = model(input_ids=input_ids[:, :i], audio_ids=audio_ids)
            # Avoid other layer decode eos
            outputs[:,1:,:,2049] = float('-inf')
            current_audio_ids = outputs.squeeze()[:,-1].argmax(-1).unsqueeze(-1)
            # Stop when output is eos (2049)
            if current_audio_ids[0] == 2049:
                break
            current_audio_ids = current_audio_ids+add_tensor
            audio_ids = torch.cat([audio_ids, current_audio_ids], dim=-1)

    # delete bos token (first timestep)
    audio_ids = audio_ids[:, 1:]
    # process unit to original mimi codec
    add_tensor = torch.zeros_like(audio_ids)
    for i in range(1, 8):
        add_tensor[i, :] = 2050 * (i)
    audio_ids = audio_ids - add_tensor
    audio_ids = audio_ids.unsqueeze(0)
    with torch.no_grad():
        # input dimension: (bs_size, 8, seq_len)
        audio_values = mimi_vocoder.decode(audio_ids)[0]
    return audio_values

if __name__ == '__main__':
    mimi_model = MimiModel.from_pretrained("kyutai/mimi", device_map='cuda')
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    model_name = "meta-llama/Meta-Llama-3-8B"
    base_model = AutoModelForCausalLM.from_pretrained( 
                    model_name,  
                    device_map="cpu",
                    torch_dtype="float",
                    trust_remote_code=True,  
                    # attn_implementation="flash_attention_2"
                )
    model = SpeechUnitModel(base_model)  # Example output_dim
    checkpoint_path = './checkpoints/checkpoint_epoch_40.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model = model.to('cuda')

    input_text = "發現新大陸"
    audio_values = inference(input_text, model, mimi_model)
    sf.write("test_output.wav", audio_values.cpu().detach().squeeze().numpy(), 24000)