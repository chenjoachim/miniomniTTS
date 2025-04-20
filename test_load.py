import torch
import argparse

def load_checkpoint(checkpoint_path, save_key=None, save_path=None):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Checkpoint loaded successfully from {checkpoint_path}")
        for key, value in checkpoint.items():
            print(f"{key}: {type(value)}")
            if isinstance(value, torch.Tensor):
                print(f" - shape: {value.shape}")
            else:
                print(f" - value: {value}")
            if "token_weights" in key:
                print(f" - token_weights: {value}")
        if save_key and save_path and save_key in checkpoint:
            model_state_dict = {}
            model_state_dict[save_key] = checkpoint[save_key]
            torch.save(model_state_dict, save_path)
            print(f"Saved {save_key} to {save_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print(f"Checkpoint is not a dict.")
        print("The checkpoint has type:", type(checkpoint))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--save_key', type=str, help='Key to save from the checkpoint')
    parser.add_argument('--save_path', type=str, help='Path to save the extracted tensor')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    load_checkpoint(checkpoint_path, args.save_key, args.save_path)