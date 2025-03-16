from transformers import LlamaForCausalLM
import torch

def extract_llama_components(model_id="meta-llama/Llama-3.2-3B-Instruct", num_layers=3):
    
    
    # Load only the required components
    model = LlamaForCausalLM.from_pretrained(model_id)
    output_path = f"{model_id.split('/')[1]}-layer-{num_layers}.pt"
    
    # Extract only what we need
    components = {
        "embed_tokens.weight": model.model.embed_tokens.weight.clone(),
        "norm.weight": model.model.norm.weight.clone()
    }
    
    # Extract just the layers we need
    for i in range(num_layers):
        for name, param in model.model.layers[i].named_parameters():
            components[f"layers.{i}.{name}"] = param.clone()
    
    # Save the extracted components
    torch.save(components, output_path)
    print(f"Saved extracted components to {output_path}")
    
    # Free memory
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    extract_llama_components(model_id="meta-llama/Llama-3.2-3B-Instruct", num_layers=3)