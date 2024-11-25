from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaModel, LlamaRotaryEmbedding, LlamaConfig

# Extract only the first three layers from Llama3's base model
class LlamaSubModel(nn.Module):
    def __init__(self, base_model):
        config = LlamaConfig()
        super(LlamaSubModel, self).__init__()
        self.embed_tokens = base_model.model.embed_tokens
        self.layers = torch.nn.ModuleList(base_model.model.layers[:3])
        self.norm = base_model.model.norm
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

    def forward(self, input_ids, attention_mask=None, position_ids=None, position_embeddings=None):
        if position_ids is None:
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        hidden_states = self.embed_tokens(input_ids)
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

class SpeechUnitModel(nn.Module):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", output_dim=2048, num_heads=8):
        super(SpeechUnitModel, self).__init__()
        model = AutoModelForCausalLM.from_pretrained( 
                model_name,  
                device_map="cpu",
                torch_dtype="float32",
                trust_remote_code=True,  
                # attn_implementation="flash_attention_2"
            )
    
        first_three_layers = {}
        for key, value in model.model.state_dict().items():
            if key.startswith("embed") or key.startswith("layers.0.") or key.startswith("layers.1.") or key.startswith("layers.2.") or key.startswith("norm"):
                first_three_layers[key] = value

        llama_sub_model = LlamaSubModel(base_model=model)
        self.base_model = llama_sub_model  # Pre-trained model (first 3 layers)
        for param in self.base_model.parameters():
            param.requires_grad = False
        # Dynamically create multiple heads
        self.heads = nn.ModuleList([nn.Linear(4096, output_dim) for _ in range(num_heads)])


    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = [head(outputs) for head in self.heads]
        # Output dimension: (batch size, 8, seq_len, 2048) 8 -> mimi codec, 2048 -> codebook size
        logits = torch.stack(logits, dim=1)
        return logits


if __name__ == "__main__":
    model_name = "meta-llama/Meta-Llama-3-8B"

    # Initialize model with extracted weights, and set output dim as mimi codebook size (2048)
    speech_model = SpeechUnitModel()  # Example output_dim

    prompt = "hello world!"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_token, attention_mask = tokenizer(prompt, return_tensors='pt').values()
    outputs = speech_model(input_token.to('cuda'), attention_mask=attention_mask.to('cuda'))
    predicted_tokens = torch.argmax(outputs, dim=-1)
    print(predicted_tokens)