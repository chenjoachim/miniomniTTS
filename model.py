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
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

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
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        # input_ids = tokenizer.apply_chat_template(
        #         data_input,
        #         add_generation_prompt=False,
        #         return_tensors="pt"
        #     )
        # # delete template index
        # input_ids = input_ids[:, 5:]
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
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
                    padding = torch.full((1,1), self.tokenizer.eos_token_id, dtype=torch.long, device=next(self.parameters()).device)
                    input_ids = torch.cat([input_ids, padding], dim=1)
                outputs = self(input_ids=input_ids[:, :i], audio_ids=audio_ids)
                # Avoid other layer decode eos
                if self.output_dim > 1:
                    outputs[:,1:,:,self.output_dim - 1] = float('-inf')
                # Print future shape
                current_audio_ids = outputs.squeeze(0)[:,-1].argmax(-1).unsqueeze(-1)
                 # Stop when output is eos (2049)
                if current_audio_ids[0] == self.output_dim - 1:
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
        # print(audio_ids)
        # audio_ids = torch.tensor([[ 12881, 15667, 12829, 8102, 12821, 12433, 11534, 14630, 12166, 5760, 3745, 10949, 1596, 2084, 11355, 14755, 14713, 795, 15825, 13491, 2513, 596, 12573, 7142, 6865, 3214, 1552, 1907, 4270, 12969, 16303, 5231, 2018, 2794, 4701, 7640, 293, 15294, 14505, 5386, 6746, 11047, 7859, 308, 7912, 9495, 13181, 8989, 1722, 4105, 2073, 596, 7299, 13472, 4380, 2051, 10017, 11103, 6265, 1774, 10175, 3882, 15084, 9223, 1517, 6582, 14811, 14619, 7198, 14510, 8442, 13436, 6388, 8878, 14255, 4380, 11538, 4986, 6906, 12308, 128, 7093, 5777, 7040, 6978, 11848, 3229, 10075, 1106, 6457, 10594, 15668, 9029, 5901, 9392, 12587, 7567, 16187, 3289, 4846, 2752, 10726, 12563, 5454, 4228, 11783, 5889, 501, 16039, 15263, 5211, 8112, 3572, 16265, 12282, 8414, 8878, 142, 5970, 10654, 16148, 10686, 15903, 2471, 11083, 14318, 13543, 5365, 9490, 3835, 8204, 10495, 15177, 8529, 8022, 12989, 10888, 6132, 1934, 11157, 1155, 5834, 8184, 15848, 62, 8418, 3976, 1825, 10428, 16194, 2961, 8287, 3856, 1719, 14658, 5666, 14790, 1502, 30, 5004, 5606, 15666, 6168, 14910, 16121, 9062, 6150, 8227, 8093, 2350, 2350, 15411, 15411, 15411, 15411, 15411, 15411, 15411, 15411, 12072 ]], device=next(self.parameters()).device)
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

