import tiktoken
import os

# 1. Setup the BPE Tokenizer
# We use 'gpt2' encoding as a standard starting point for LLMs

tokenizer = tiktoken.get_encoding("gpt2")

file_path = "the-verdict.txt"

if not os.path.exists(file_path):
    print(f"Error: Could not find {file_path}. Please check the file location.")
else:
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

enc_text = tokenizer.encode(raw_text)

# Print results
print("------------------------------------------------")
print(f"Total characters in file: {len(raw_text)}")
print(f"Total tokens after BPE:   {len(enc_text)}")
print("------------------------------------------------")
print(f"First 10 tokens: {enc_text[:10]}")

