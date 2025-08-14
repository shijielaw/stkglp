# ST-KGLP: Combining Structural and Textual Knowledge for Knowledge Graph Link Prediction via Large Language Models

This is the PyTorch implementation for ST-KGLP.




## Architecture

The project contains files:

```
ST-KGLP/
├── main.py              # Main entry point for training, testing, and evaluation
├── finetune.py          # Model fine-tuning with LoRA
├── inference.py         # Model inference and prediction
├── eval.py             # Evaluation metrics computation
├── combiner.py         # knowledge aligner and query-aware adaptive weighting
├── utils/              # Utility functions and tools
│   ├── prompter.py     # Prompt template management
│   ├── tools.py        # General utilities and KGE processing
│   └── callbacks.py    # Training callbacks
├── prompts/            # Prompt templates
├── lora/               # LoRA weights storage
├── log/                # Logs and results
└── data/               # Dataset storage
```



## Environment Requirement

The code has been tested running under Python 3.10.16 on Linux. 

The required packages are as follows:

```
datasets==2.19.1
fire==0.6.0
matplotlib==3.7.5
numpy==1.23.1
peft==0.3.0
torch==2.2.0
tqdm==4.66.4
transformers==4.40.1
modelscope==1.28.2
```



## Download LLMs

```bash
cd ./llms
python llm_downloader.py
```



## Data Preparation

Place your dataset in the `data/` directory with the following structure:

```
data/
└── {dataset_name}/
    ├── train.json          # Training samples
    ├── valid.json          # Validation samples
    ├── test.json           # Test samples
    ├── entity_embedding.npy    # Entity embeddings
    └── relation_embedding.npy  # Relation embeddings
```



## Run Model

### - Training and Testing

Set all training parameters and run:

```bash
python main.py
```

### - Testing Only

Set `args.run_mode = "test"` and run:

```bash
python main.py
```



##  Configuration Parameters

### - Training Parameters
- `--num_epochs`: Number of training epochs
- `--batch_size`: Global batch size
- `--micro_batch_size`: Per-GPU batch size
- `--learning_rate`: Learning rate for optimization
- `--cutoff_len`: Maximum sequence length

### - LoRA Parameters
- `--lora_rank`: Rank of LoRA adaptation
- `--lora_alpha`: Alpha parameter for LoRA
- `--lora_dropout`: Dropout rate for LoRA layers
- `--lora_target_modules`: Target modules for LoRA adaptation

### - Model Parameters
- `--llm_path`: Path to pre-trained LLM
- `--llm_hidden_size`: Hidden dimension of LLM
- `--adapter_hidden_dim`: Hidden dimension of knowledge aligner



## Output Structure

After training, the model generates:

```
lora/{dataset_name}/run_epoch{num_epochs}/     # LoRA weights
log/res/{dataset_name}/                        # Results and responses
├── responses_epoch{num_epochs}.txt            # Model responses
├── results_epoch{num_epochs}.json             # Structured results
└── evaluation_epoch{num_epochs}.txt           # Evaluation metrics
```


