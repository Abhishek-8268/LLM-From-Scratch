import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            # Convert to PyTorch tensors (the format the AI needs)
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # Returns the total number of rows in the dataset
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Returns a single row (input and target)
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


if __name__ == "__main__":
    
    # 1. Read the text file
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 2. Test with Batch Size 1 (To see the sliding window effect)
    print("--- Test 1: Batch Size 1, Stride 1 ---")
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print("First batch:\n", first_batch)

    second_batch = next(data_iter)
    print("Second batch:\n", second_batch)

    # 3. Test with Batch Size 8 (To see a real training batch)
    print("\n--- Test 2: Batch Size 8, Stride 4 ---")
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)