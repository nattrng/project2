from transformers import GPTNeoXForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, TrainingArguments, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.optim import Adam
from PythiaBinaryDataset import PythiaBinaryDataset
from NoShuffleTrainer import NoShuffleTrainer
from custom_cross_entropy import SynonymCrossEntropy
import wandb
import yaml

wandb.login()


def load_config(path: str):
    try:
        with open(path, 'r') as config:
            config = yaml.safe_load(config) #returns dict btw
            return config
    except FileNotFoundError:
        print("Config File Not Found. Check your path!!")

config_dict = load_config('/Users/nathan/Documents/Development/project2/pythia_160m_deduped_config.yaml')

def create_neighbor_lookup_table(embed_matrix, num_neighbors):
    seq_len = embed_matrix.shape[0]
    scaffold = torch.ones((embed_matrix.shape[0], num_neighbors))
    for token_idx in range(seq_len):
        target = embed_matrix[token_idx]
        norms = torch.norm(embed_matrix - target, dim=-1, p=2)
        norms[token_idx] = float('inf')
        indices = norms.topk(num_neighbors, largest=False, sorted=True)[1] # chooses smallest, sorts from smallest to largest
        scaffold[token_idx] = indices 
    return scaffold.int()

wandb.init(
    project="pythia-160m-test-training-run", 
    name="Test-Run-01",
    config=config_dict
)

train_iters = config_dict['train-iters']
seq_len = config_dict['seq-length']
global_batch_size = 1024 # attempts to simulate using micro batches

micro_batch_size = 8 # done for grad accum
grad_accum_steps = global_batch_size // micro_batch_size

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

other_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")
embed_matrix = other_model.get_input_embeddings().weight
del other_model 

model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-160m-deduped", revision="step0", cache_dir="./pythia-160m-deduped/step0")
model.to(device)



#tokenizer unnecessary, i think document.bin comes preprocessed with the targets idk



optimizer = Adam(
    model.parameters(), 
    lr = config_dict['optimizer']['params']['lr'], 
    betas = config_dict['optimizer']['params']['betas'], 
    eps = config_dict['optimizer']['params']['eps'], 
    weight_decay = config_dict['weight-decay']
    )

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_training_steps = train_iters,
    num_warmup_steps = int(train_iters * config_dict['warmup'])
)

dataset = PythiaBinaryDataset("./pythia_data/document.bin", seq_len)


training_arguments = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=grad_accum_steps,
    max_steps=train_iters,  
    logging_steps=1,
    report_to="wandb",
    save_steps=1000,
    fp16=torch.cuda.is_available(),
    bf16=torch.mps.is_available()
)

trainer = NoShuffleTrainer(
    model=model, 
    args=training_arguments,
    train_dataset=dataset,
    optimizers=(optimizer, scheduler),
    # compute_loss_func = SynonymCrossEntropy(embed_matrix=embed_matrix)
)

trainer.train()

