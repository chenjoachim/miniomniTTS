import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class MimiUnitDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        """
        Args:
            hf_dataset: Input dataset containing 'text' and 'unit'.
            tokenizer: Tokenizer instance (e.g., LLaMA3 tokenizer).
            max_length (int): Maximum tokenized sequence length.
        """
        self.dataset = hf_dataset  # Convert iterable dataset to a list for indexing
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract 'text' and 'unit' from the dataset
        item = self.dataset[idx]

        # Process the text into Llama3 index
        data_input = [{'role': 'assistant', "content": item['text']}]

        # Process the audio unit from list to tensor, and add index to each dimension
        labels = torch.tensor(item['unit'], dtype=torch.long)
        input_ids = self.tokenizer.apply_chat_template(
            data_input,
            add_generation_prompt=False,
            return_tensors="pt"
        )
        # Cut llama generate template index
        input_ids = input_ids[:, 5:]
        if labels.shape[1] > input_ids.shape[1]+1:
            padding = torch.full((1, labels.shape[1] - input_ids.shape[1]+1), self.tokenizer.eos_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding], dim=1)
        else:
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                # data have been processed
                pass

        add_tensor = torch.zeros_like(labels)
        for i in range(1, 8):
            add_tensor[i, :] = 2050 * (i)
        labels = labels + add_tensor
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }
    
def mimi_collate_fn(batch):
    """
    Collate function for dynamically padding input_ids, attention_mask, and labels (units).

    Args:
        batch (list): List of samples from the dataset.

    Returns:
        dict: Dictionary containing padded input_ids, attention_mask, and labels.
    """
    # Extract input_ids, attention_mask, and labels
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_mask = [torch.tensor(item['attention_mask']) for item in batch]
    labels = [item['labels'] for item in batch]

    # Determine the maximum length in the current batch
    max_length = max(seq.size(0) for seq in input_ids)

    # Pad input_ids and attention_mask to the maximum length in the batch
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Ensure input_ids and attention_mask are trimmed to the batch max length
    padded_input_ids = padded_input_ids[:, :max_length]
    padded_attention_mask = padded_attention_mask[:, :max_length]

    # Pad labels (variable-length tensors)
    max_channels = max(label.size(0) for label in labels)
    max_label_length = max(label.size(1) for label in labels)

    padded_labels = torch.zeros(len(labels), max_channels, max_label_length)  # Initialize with zeros
    for i, label in enumerate(labels):
        padded_labels[i, :label.size(0), :label.size(1)] = label  # Copy the actual label values

    padded_labels = padded_labels.type(torch.LongTensor)

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': padded_labels
    }

def filter_dataset(hf_dataset, tokenizer):
    """
    返回一個 filter 函數，用於過濾資料集
    """
    def is_valid_item(item):
        try:
            data_input = [{'role': 'assistant', "content": item['text']}]
            labels = torch.tensor(item['unit'], dtype=torch.long)
            
            input_ids = tokenizer.apply_chat_template(
                data_input,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            input_ids = input_ids[:, 5:]
            
            return labels.shape[1] > input_ids.shape[1] + 1
            
        except Exception:
            return False
    
    # 使用 Dataset.filter 而不是存儲在記憶體中
    filtered_dataset = hf_dataset.filter(is_valid_item, num_proc=4)
    
    print(f"Original dataset size: {len(hf_dataset)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    print(f"Removed {len(hf_dataset) - len(filtered_dataset)} items")

if __name__ == "__main__":
    # Initialize the dataset
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    from itertools import islice

    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("anthony-wss/Soundon-tts", streaming="True")

    def collect_and_save(iterable_dataset, num_samples=256):
        """
        Collects a specified number of samples from an IterableDataset and saves them as a Hugging Face Dataset.

        Args:
            iterable_dataset (IterableDataset): Input IterableDataset to sample from.
            output_path (str): Path to save the Hugging Face Dataset.
            num_samples (int): Number of samples to collect.
        """
        # Collect `num_samples` items from the iterable dataset
        collected_samples = list(islice(iterable_dataset, num_samples))

        # Convert to a Hugging Face Dataset
        hf_dataset = Dataset.from_list(collected_samples)
        return hf_dataset
    test_ds = collect_and_save(ds['train'], num_samples=256)
    dataset = MimiUnitDataset(test_ds, tokenizer)

    # Wrap with a DataLoader for batching
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=mimi_collate_fn)
    for data in data_loader:
        print(data['input_ids'].shape)
        print('labels')
        print(data['attention_mask'])
        print(data['labels'].shape)
        print(data['labels'])
        break
