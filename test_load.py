import torch
import argparse

def load_checkpoint(checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Checkpoint loaded successfully from {checkpoint_path}")
        for key, value in checkpoint.items():
            print(f"{key}: {type(value)}")
            if isinstance(value, torch.Tensor):
                print(f" - shape: {value.shape}")
            else:
                print(f" - value: {value}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    load_checkpoint(checkpoint_path)