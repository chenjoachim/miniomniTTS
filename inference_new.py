import torch
import argparse
import os
import sys
import uuid
import torchaudio
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the SpeechUnitModel from model.py
from model import SpeechUnitModel
# Import the GLMvocoder from decoder.py
from decoder import GLMvocoder
from transformers.models.llama.modeling_llama import LlamaModel, LlamaRotaryEmbedding, LlamaConfig, LlamaDecoderLayer
import torch.nn as nn

def initialize_speech_unit_model(checkpoint_path=None, llama_components_path="llama_partial.pt", 
                                output_dim=2050, num_heads=8, model_id="meta-llama/Llama-3.2-3B-Instruct"):
    # Load the extracted LLaMa components
    print(f"Loading LLaMa components from {llama_components_path}")
    llama_components = torch.load(llama_components_path, map_location="cpu")
    
    # Get embedding dimensions
    vocab_size, embed_dim = llama_components["embed_tokens.weight"].shape
    num_layers = len(set([int(k.split('.')[1]) for k in llama_components.keys() 
                         if k.startswith("layers.")]))
    
    print(f"Found {num_layers} layers with embedding dimension {embed_dim}")
    
    # Create LLaMa config
    config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    config.hidden_size = embed_dim
    config.num_hidden_layers = num_layers
    
    
    
    class LlamaModelStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
            self.embed_tokens.weight.data.copy_(llama_components["embed_tokens.weight"])
            
            self.layers = nn.ModuleList([LlamaDecoderLayer(config, i) for i in range(num_layers)])
            
            self.norm = nn.LayerNorm(embed_dim, bias=False)
            self.norm.weight.data.copy_(llama_components["norm.weight"])
    
    class LlamaStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = LlamaModelStub()
            self.config = config
    
    # Create the base model
    base_model = LlamaStub()
    
    # Create the SpeechUnitModel
    speech_model = SpeechUnitModel(
        base_model=base_model,
        llama_layers=num_layers,
        output_dim=output_dim,
        num_heads=num_heads,
        model_id=model_id  # Just a placeholder
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading SpeechUnitModel checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # Handle possible key prefixes
        if "model_state_dict" in checkpoint:
            speech_model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            speech_model.load_state_dict(checkpoint["state_dict"])
        else:
            speech_model.load_state_dict(checkpoint)
    
    print("SpeechUnitModel initialization complete!")
    return speech_model

def run_inference(args):
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load base model with quantization if specified
    print(f"Loading base model: {args.model}")
    # base_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    llama_components_path = f"{args.model.split('/')[1]}-layer-{args.layers}.pt"
    speech_model = initialize_speech_unit_model(
        llama_components_path=llama_components_path,
        checkpoint_path=args.checkpoint,
        output_dim=16386,
        num_heads=1,
        model_id=args.model
    ).to(device)
    # Initialize speech model
    
    print(f"Initialized SpeechUnitModel with {args.layers} layers")
    # speech_model = SpeechUnitModel(
    #     checkpoint_path=args.checkpoint,
    #     llama_layers=args.layers,
    #     output_dim=16386,
    #     num_heads=1,
    #     model_id=args.model
    # ).to(device)
    
    # checkpoint = torch.load(args.checkpoint, map_location=device)
    # speech_model.load_state_dict(checkpoint)
    
    speech_model.eval()
    print(f"Model created with {sum(p.numel() for p in speech_model.parameters())/1e6:.2f}M parameters")
    
    # Initialize the vocoder
    if args.use_mock:
        from model import MockVocoder
        vocoder = MockVocoder()
    else:
        print("Initializing GLMvocoder...")
        vocoder = GLMvocoder(device)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run inference with the provided text
    print(f"Running inference with text: '{args.text}'")
    try:
        with torch.inference_mode():
            audio_values = speech_model.inference(
                input_text=args.text,
                vocoder=vocoder,
                max_length=args.max_length
            )
        
        # Save the audio output
        if args.use_mock:
            # For mock vocoder, save numpy array
            import numpy as np
            output_path = f"{args.output}/generated_audio.npy"
            np.save(output_path, audio_values.cpu().numpy())
        else:
            # For real vocoder, save as wav file
            output_path = f"{args.output}/generated_audio.wav"
            torchaudio.save(output_path, audio_values.unsqueeze(0), args.sample_rate)
        
        print(f"Generated audio saved to {output_path}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
    
    print("Inference completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech Synthesis Inference")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", 
                        help="Model ID to use as base model")
    parser.add_argument("--text", type=str, default="雖然角色扮演遊戲的劇情走向有一定的框架，但是玩家可以通過自己的選擇和行動來影響劇情的細節，使得劇情走向更有變數和趣味性。有些遊戲甚至會根據玩家的選擇和行動而有多個結局，讓玩家可以體驗到不同的故事。",
                        help="Text to synthesize into speech")
    parser.add_argument("--output", type=str, default="output", 
                        help="Output directory for generated audio")
    parser.add_argument("--layers", type=int, default=3,
                        help="Number of LLaMA layers to use")
    parser.add_argument("--max_length", type=int, default=150,
                        help="Maximum length for audio token generation")
    parser.add_argument("--use_mock", action="store_true", 
                        help="Use mock vocoder instead of GLMvocoder")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint file")
    parser.add_argument("--sample_rate", type=int, default=24000,
                        help="Sample rate of the result")
    
    args = parser.parse_args()
    run_inference(args)

