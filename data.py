import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

class MimiUnitDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512, num_layers=8, codebook_size=2048, column_names=['text', 'unit']):
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
        self.codebook_size = codebook_size
        self.num_layers = num_layers
        self.text_column = column_names[0]
        self.unit_column = column_names[1]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract 'text' and 'unit' from the dataset
        item = self.dataset[idx]

        # Process the text into Llama3 index
        # data_input = [{'role': 'assistant', "content": item[self.text_column]}]
        text = item[self.text_column]
        input_ids = self.tokenizer.encode(text, return_tensors="pt")

        # Process the audio unit from list to tensor, and add index to each dimension
        labels = torch.tensor(item[self.unit_column], dtype=torch.long)
        
        input_ids = self.tokenizer.encode()

        # If label is 1D, add a dimension
        if len(labels.shape) == 1:
            labels = labels.unsqueeze(0)
        
        # Handle mismatch (e.g., skip or pad/truncate)
        if labels.shape[0] != self.num_layers:
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                # If this is the last item, create a placeholder with correct dimensions
                labels = torch.zeros((self.num_layers, labels.shape[1]), dtype=torch.long)

        # If the label is longer than the input_ids, add padding
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
        for i in range(1, self.num_layers):
            add_tensor[i, :] = (self.codebook_size + 2) * (i)
        labels = labels + add_tensor
        
        attention_mask = torch.ones_like(input_ids)
    
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
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
    input_ids = [item['input_ids'].squeeze(0) for item in batch]  # Remove batch dimension
    attention_mask = [item['attention_mask'].squeeze(0) for item in batch]
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

def filter_dataset(hf_dataset, tokenizer, text_column='text', unit_column='unit', num_layers=8):
    """
    Returns a filtered dataset based on valid items.
    
    Args:
        hf_dataset: The Hugging Face dataset to filter
        tokenizer: The tokenizer to use for processing text
        text_column (str): The column name containing the text data
        unit_column (str): The column name containing the unit data
        num_layers (int): Expected number of layers in unit data
    """
    def is_valid_item(item):
        try:
            # Check if required columns exist
            if text_column not in item or unit_column not in item:
                return False
            
            text = item[text_column]
            data_input = [{'role': 'assistant', "content": text}]
            labels = torch.tensor(item[unit_column], dtype=torch.long)
            
            # Handle 1D labels case
            if len(labels.shape) == 1:
                # If we expect only 1 layer, this is valid
                if num_layers != 1:
                    return False
                # Reshape to 2D with first dimension as 1
                labels = labels.unsqueeze(0)
            # If we expect a specific number of layers, verify
            elif labels.shape[0] != num_layers:
                return False
            
            # input_ids = tokenizer.apply_chat_template(
            #     data_input,
            #     add_generation_prompt=False,
            #     return_tensors="pt"
            # )
            # input_ids = input_ids[:, 5:]
            
            input_ids = tokenizer.encode(text, return_tensors="pt")
            
            return labels.shape[1] > input_ids.shape[1] + 1
            
        except Exception as e:
            # Optionally log the exception for debugging
            # print(f"Error processing item: {e}")
            return False
    
    # Use Dataset.filter to filter items
    filtered_dataset = hf_dataset.filter(is_valid_item, num_proc=4)
    
    print(f"Original dataset size: {len(hf_dataset)}")
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    print(f"Removed {len(hf_dataset) - len(filtered_dataset)} items")
    
    return filtered_dataset

if __name__ == "__main__":
    # Initialize the dataset
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    from itertools import islice

    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("Allen172/GLM-4_codec_dataset", streaming=True)
    
    # Configure parameters
    num_layers = 1  # Number of layers
    codebook_size = 16384  # Codebook size

    def collect_and_save(iterable_dataset, num_samples=256):
        """
        Collects a specified number of samples from an IterableDataset and saves them as a Hugging Face Dataset.
        """
        collected_samples = list(islice(iterable_dataset, num_samples))
        hf_dataset = Dataset.from_list(collected_samples)
        return hf_dataset
        
    test_ds = collect_and_save(ds['train'], num_samples=64)
    
    # Pass the parameters to the dataset
    dataset = MimiUnitDataset(test_ds, tokenizer, num_layers=num_layers, codebook_size=codebook_size,
                              column_names=['label_text', 'label_codec'])

    # Wrap with a DataLoader for batching
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=mimi_collate_fn)
    
    for data in data_loader:
        print(f"Input shape: {data['input_ids'].shape}")
        print(f"Labels shape: {data['labels'].shape}")
        print(f"Number of layers: {data['labels'].shape[1]}")
        print(f"Sample labels from first item:")
        for layer in range(min(3, data['labels'].shape[1])):  # Print first 3 layers only
            print(f"Layer {layer}: {data['labels'][0, layer, :10]}")  # First 10 values
        break
