from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel, AutoConfig
from transformers.models.llama.modeling_llama import LlamaModel, LlamaRotaryEmbedding, LlamaConfig
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Extract only the first three layers from Llama3's base model
class SpeechUnitModel(nn.Module):
    def __init__(self, base_model, llama_layers=3, output_dim=2050, num_heads=8, model_id="meta-llama/Llama-3.2-3B-Instruct"):
        super(SpeechUnitModel, self).__init__()
        
        # Configuration and base model initialization
        config = LlamaConfig()
        config.num_hidden_layers = llama_layers
        # Embedding layers
        self.embed_tokens = base_model.model.embed_tokens
        original_vocab_size, embed_dim = self.embed_tokens.weight.shape
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.model_id = model_id

        # (2048 + 2 (EOS + BOS) )codebook * 8 head , 2048 as begin-of-audio, 2049 as end-of-audio
        self.codebook_size = output_dim * num_heads
        self.audio_embed = nn.Embedding(self.codebook_size, embed_dim)
        nn.init.xavier_uniform_(self.audio_embed.weight.data)

        self.token_weights = nn.Parameter(torch.ones(num_heads))

        # Transformer layers
        self.layers = torch.nn.ModuleList(base_model.model.layers[:llama_layers])
        for i in range(len(self.layers)):
            self.layers[i].self_attn.is_causal = True

        self.norm = base_model.model.norm
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        # Prediction heads
        self.heads = nn.ModuleList([nn.Linear(embed_dim, output_dim) for _ in range(num_heads)])

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
            # audio_ids shape: (num_heads, seq_len)
            audio_embedding = self.audio_embed(audio_ids)   # shape: (num_heads, seq_len, embed_dim)
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

        # Prediction heads
        logits = [head(hidden_states) for head in self.heads]
        logits = torch.stack(logits, dim=1)  # Output dimension: (batch size, 8, output_dim)
        return logits

    def _extend_attention_mask(self, attention_mask, device):
        return (1.0 - attention_mask[:, None, None, :]) * -10000.0
    
    def inference(self, input_text, vocoder, max_length=150):
        # process input_text into index
        data_input = [{'role': 'assistant', "content": input_text}]
        model_id = self.model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        input_ids = tokenizer.apply_chat_template(
                data_input,
                add_generation_prompt=False,
                return_tensors="pt"
            )
        blank_input_ids = tokenizer.apply_chat_template(
            [{'role': 'assistant', "content": ""}],
            add_generation_prompt=False,
            return_tensors="pt"
        )
        # Duplicate to a batch
        print("[DEBUG] shape of input_ids:", input_ids.shape)
        print("[DEBUG] shape of blank ids", blank_input_ids.shape)
        print("[DEBUG] First 10 tokens:", input_ids[0, :10])
        print("[DEBUG] First 10 tokens (blank):", blank_input_ids)
        
        # delete template index
        input_ids = input_ids[:, 5:]
        _, seq_length = input_ids.shape
        input_ids = input_ids.to(next(self.parameters()).device)
        
        print("[DEBUG] shape of input_ids after template index removal:", input_ids.shape)
        audio_ids = torch.full((self.num_heads, 1), (self.output_dim - 2), device=next(self.parameters()).device)
        add_tensor = torch.zeros_like(audio_ids)
        for i in range(1, self.num_heads):
            add_tensor[i, :] = self.output_dim * (i)
        with torch.no_grad():
            for i in range(1, max_length):
                if i > seq_length:
                    padding = torch.full((1,1), tokenizer.eos_token_id, dtype=torch.long, device=next(self.parameters()).device)
                    input_ids = torch.cat([input_ids, padding], dim=1)
                print("[DEBUG] shape of ids passed to model:", input_ids[:, :i].shape)
                print("[DEBUG] shape of audio_ids passed to model:", audio_ids.shape)
                outputs = self(input_ids=input_ids[:, :i], audio_ids=audio_ids)
                # Avoid other layer decode eos
                print("[DEBUG] shape of outputs:", outputs.shape)
                if self.output_dim > 1:
                    outputs[:,1:,:,self.num_heads - 1] = float('-inf')
                print("[DEBUG] shape of outputs after masking:", outputs.shape)
                # Print future shape
                print("[DEBUG] shape of squeeze:", outputs.squeeze(0).shape)
                print("[DEBUG] shape of squeeze:", outputs.squeeze(0)[:,-1].shape)
                print("[DEBUG] shape of squeeze:", outputs.squeeze(0)[:,-1].argmax(-1).shape)
                print("[DEBUG] shape of squeeze:", outputs.squeeze(0)[:,-1].argmax(-1).unsqueeze(-1).shape)
                current_audio_ids = outputs.squeeze(0)[:,-1].argmax(-1).unsqueeze(-1)
                print("[DEBUG] shape of current_audio_ids:", current_audio_ids.shape)
                # Stop when output is eos (2049)
                print("[DEBUG] current_audio_ids:", current_audio_ids)
                if current_audio_ids[0] == self.num_heads - 1:
                    break
                current_audio_ids = current_audio_ids+add_tensor
                audio_ids = torch.cat([audio_ids, current_audio_ids], dim=-1)
        print("[DEBUG] Final shape of audio_ids:", audio_ids.shape)
        # delete bos token (first timestep)
        audio_ids = audio_ids[:, 1:]
        # process unit to original mimi codec
        add_tensor = torch.zeros_like(audio_ids)
        for i in range(1, self.num_heads):
            add_tensor[i, :] = self.output_dim * (i)
        audio_ids = audio_ids - add_tensor
        if self.num_heads > 1:
            audio_ids = audio_ids.unsqueeze(0)
        print("[DEBUG] Final shape of audio_ids after processing:", audio_ids.shape)
        with torch.no_grad():
            # input dimension: (bs_size, 8, seq_len)
            audio_values = vocoder.decode(audio_ids)[0]
        return audio_values


class MockVocoder:
    def decode(self, token_ids):
        # Simply return a random audio tensor based on the token IDs shape
        # In a real scenario, this would convert token IDs to audio waveform
        batch_size = token_ids.shape[0]
        length = token_ids.shape[2] * 256  # Assuming each token represents ~256 samples
        return [torch.randn(length) for _ in range(batch_size)]


def main():
    parser = argparse.ArgumentParser(description="Test SpeechUnitModel")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", 
                        help="Model ID to use as base model")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the speech synthesis model.",
                        help="Text to synthesize into speech")
    parser.add_argument("--output", type=str, default="output", 
                        help="Output directory for test results")
    parser.add_argument("--layers", type=int, default=3,
                        help="Number of LLaMA layers to use")
    parser.add_argument("--max_length", type=int, default=150,
                        help="Maximum length for audio token generation")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization for the base model")
    parser.add_argument("--use_mock", action="store_true", default=True,
                        help="Use mock vocoder (since real one is unavailable)")
    
    args = parser.parse_args()
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    import os
    os.makedirs(args.output, exist_ok=True)
    
    # Load model with or without quantization
    print(f"Loading base model: {args.model}")
    try:
        if args.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using a smaller model as fallback...")
        fallback_model = "facebook/opt-125m"
        base_model = AutoModelForCausalLM.from_pretrained(fallback_model).to(device)
    
    # Initialize speech model
    print(f"Initializing SpeechUnitModel with {args.layers} layers")
    speech_model = SpeechUnitModel(
        base_model=base_model,
        llama_layers=args.layers,
        output_dim=2050,
        num_heads=8,
        model_id=args.model
    ).to(device)
    
    speech_model.eval()
    print(f"Model created with {sum(p.numel() for p in speech_model.parameters())/1e6:.2f}M parameters")
    
    # Use mock vocoder for testing
    vocoder = MockVocoder()
    
    # Test model with provided text
    print(f"Testing with text: '{args.text}'")
    try:
        with torch.inference_mode():
            audio_values = speech_model.inference(
                input_text=args.text,
                vocoder=vocoder,
                max_length=args.max_length
            )
        
        # Visualize and save the generated audio
        plt.figure(figsize=(10, 4))
        plt.plot(audio_values[:1000].cpu().numpy())
        plt.title("First 1000 samples of generated audio")
        plt.savefig(f"{args.output}/generated_audio.png")
        
        # Save the generated audio (as numpy array since we don't have actual audio)
        np.save(f"{args.output}/audio_values.npy", audio_values.cpu().numpy())
        
        print(f"Generated audio with {len(audio_values)} samples")
        print(f"Results saved to {args.output} directory")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        
        # Try a direct forward pass to diagnose issues
        print("\nTesting direct forward pass...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            sample_input = tokenizer("Test input", return_tensors="pt").input_ids.to(device)
            batch_size, seq_len = sample_input.shape
            
            dummy_audio_ids = torch.randint(0, 2050, (8, seq_len)).to(device)
            
            outputs = speech_model(
                input_ids=sample_input,
                audio_ids=dummy_audio_ids
            )
            
            print(f"Forward pass successful. Output shape: {outputs.shape}")
        except Exception as e2:
            print(f"Forward pass also failed: {e2}")
    
    print("Test completed!")

if __name__ == "__main__":
    main()
