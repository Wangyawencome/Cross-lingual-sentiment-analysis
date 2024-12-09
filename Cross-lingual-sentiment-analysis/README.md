## 文件和目录说明
- **`README.md`**: Introduction to the project and usage instructions
- **`src/`**: Source code of the project
- **`test.py`**: Testing script
- **`data/`**: Project data
  - **`data/raw/split_data.py`**: Constructs training and testing datasets
  - **`data/semantic_similarity_search.py`**: Semantic similarity search module
  - **`data/sentiment_instruction_generator.py`**:  Sentiment instruction generator
- **`examples/`**: Project configuration files
  - **`examples/train_lora/llama3_lora_sft.yaml`**: LoRA fine-tuning configuration
  - **`examples/merge_lora/llama3_lora_sft.yaml`**: LoRA merging configuration.
  - **`examples/inference/llama3_lora_sft.yaml`**: Inference configuration

- **`requirements.txt`**:  Lists the Python dependencies required for the project


## Commands

The following three commands perform LoRA fine-tuning, inference, and merging for the Llama3-8B-Instruct model:

`llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml`

`llamafactory-cli chat examples/inference/llama3_lora_sft.yaml`

`llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml`
