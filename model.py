from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel, AutoConfig
from transformers.models.llama.modeling_llama import LlamaModel, LlamaRotaryEmbedding, LlamaConfig

# Extract only the first three layers from Llama3's base model
class LlamaSubModel(nn.Module):
    def __init__(self, model_name, num_layers=6):
        config = LlamaConfig()
        config.num_hidden_layers = 6
        super(LlamaSubModel, self).__init__()
        base_model = AutoModelForCausalLM.from_pretrained( 
                model_name,  
                device_map="cpu",
                torch_dtype="float32",
                trust_remote_code=True,  
                # attn_implementation="flash_attention_2"
            )
        self.embed_tokens = base_model.model.embed_tokens
        original_vocab_size, embed_dim = self.embed_tokens.weight.shape

        # 2048 codebook * 8 head + 2 (EOS + PAD)
        self.audio_embed = nn.Embedding(16386, embed_dim)
        nn.init.xavier_uniform_(self.audio_embed.weight.data)

        self.token_weights = nn.Parameter(torch.ones(8))
        self.layers = torch.nn.ModuleList(base_model.model.layers[:num_layers])

        # Use causal attention to allow streaming input
        for i in range(len(self.layers)):
            self.layers[i].self_attn.is_causal = True

        self.norm = base_model.model.norm
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(self, input_ids, audio_ids=None, attention_mask=None, position_ids=None, position_embeddings=None):
        '''
        Be aware that the length of input_ids should be the same as audio_ids.
        '''
        if position_ids is None:
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # hidden_states shape: (batch_size, seq_len, hidden_dim)
        hidden_states = self.embed_tokens(input_ids)

        if audio_ids is not None:
            # audio_ids shape: (batch_size, 8, seq_len)
            add_tensor = torch.zeros_like(audio_ids)
            for i in range(1, audio_ids.shape[0]):
                add_tensor[i] = 2048 * (i)
            audio_ids = audio_ids + add_tensor
            audio_embedding = self.audio_embed(audio_ids)   # shape: (batch_size, 8, seq_len, 4096)
            weight_audio = torch.sum(audio_embedding * self.token_weights.view(1, -1, 1, 1), dim=1)

            hidden_states = hidden_states + weight_audio


        if position_embeddings is None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            
        extended_attention_mask = None
        if attention_mask is not None:
            extended_attention_mask = self._extend_attention_mask(attention_mask, hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=extended_attention_mask, position_embeddings=position_embeddings
            )[0]

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def _extend_attention_mask(self, attention_mask, device):
        return (1.0 - attention_mask[:, None, None, :]) * -10000.0

class SpeechUnitModel(PreTrainedModel, GenerationMixin):
    def __init__(self, config, model_name="meta-llama/Meta-Llama-3-8B", output_dim=2050, num_heads=8):
        super().__init__(config)
        self.ref_model = LlamaSubModel(model_name)  # Pre-trained model (first 3 layers)

        # create 8 head to predict mimi codec
        self.heads = nn.ModuleList([nn.Linear(4096, output_dim) for _ in range(num_heads)])

    def forward(self, input_ids, audio_ids, attention_mask=None):
        outputs = self.ref_model(input_ids=input_ids, audio_ids=audio_ids, attention_mask=attention_mask)
        logits = [head(outputs[:, -1, :]) for head in self.heads]
        # Output dimension: (batch size, 8, 2048) 8 -> mimi codec, 2048 -> codebook size
        logits = torch.stack(logits, dim=1)
        return logits


if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B"

    # Initialize model with extracted weights, and set output dim as mimi codebook size (2048)
    speech_model = SpeechUnitModel()  # Example output_dim

    prompt = "hello world!"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_token, attention_mask = tokenizer(prompt, return_tensors='pt').values()
    from datasets import load_dataset
    ds = load_dataset("anthony-wss/Soundon-tts", streaming="True")
    for data in ds['train']:
        test_text = data['text']
        test_case = torch.tensor(data['unit'])
        print(torch.tensor(data['unit']).shape)
        break
    test_unit = test_case[:, :4]
    test_unit = test_unit.unsqueeze(0)
    base_model = AutoModelForCausalLM.from_pretrained( 
                model_name,  
                device_map="cpu",
                torch_dtype="float",
                trust_remote_code=True,  
                # attn_implementation="flash_attention_2"
            )
    first_three_layers = {}
    for key, value in base_model.model.state_dict().items():
        if key.startswith("embed") or key.startswith("layers.0.") or key.startswith("layers.1.") or key.startswith("layers.2.") or key.startswith("layers.3.") or key.startswith("layers.4.") or key.startswith("layers.5.") or key.startswith("norm"):
            first_three_layers[key] = value
    speech_model.ref_model.load_state_dict(first_three_layers, strict=False)  # 'strict=False' allows partial loading
    speech_model = speech_model.to('cuda')
    input_token, attention_mask = tokenizer(prompt, return_tensors='pt').values()
    outputs = speech_model(input_token.to('cuda'), audio_ids=test_unit.to('cuda'), attention_mask=attention_mask.to('cuda'))
    predicted_tokens = torch.argmax(outputs, dim=-1)
    print(predicted_tokens)