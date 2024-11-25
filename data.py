import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

class MimiUnitDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        """
        Args:
            hf_dataset: Input dataset containing 'text' and 'unit'.
            tokenizer: Tokenizer instance (e.g., LLaMA3 tokenizer).
            max_length (int): Maximum tokenized sequence length.
        """
        self.dataset = list(hf_dataset)  # Convert iterable dataset to a list for indexing
        self.tokenizer = tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract 'text' and 'unit' from the dataset
        item = self.dataset[idx]
        text = item['text']
        unit = item['unit']

        # Tokenize the text
        tokenized = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Extract input_ids and attention_mask
        input_ids = tokenized['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = tokenized['attention_mask'].squeeze(0)  # Remove batch dimension

        # Convert 'unit' to tensor
        label = torch.tensor(unit, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label
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
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad input_ids and attention_mask
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Pad labels (variable-length tensors)
    max_channels = max(label.size(0) for label in labels)
    max_length = 512

    padded_labels = torch.zeros(len(labels), max_channels, max_length)  # Initialize with zeros
    for i, label in enumerate(labels):
        padded_labels[i, :label.size(0), :label.size(1)] = label  # Copy the actual label values
    padded_labels = padded_labels.type(torch.LongTensor)

    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': padded_labels
    }

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
