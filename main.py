import torch 
import transformers 
import lm_eval 
from datasets import load_dataset 
from transformers import AutoTokenizer, LlamaForCausalLM 
import numpy as np 
from datasets import concatenate_datasets 
from torch.utils.data import DataLoader 
from typing import List, Literal, Optional, Tuple, Union 
import argparse 

### Parsing the arguments ### 
parser = argparse.ArgumentParser(description = "CommonSense Reasoning with generation and chain-of-thoughts") 
parser.add_argument("--tasks", type = str) 
parser.add_argument("--model", type = str) 
parser.add_argument("--device", type = str) 

args = parser.parse_args() 
tasks = args.tasks.split(",") 

### Loading the tokenizer and the model ### 
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct") 
if tokenizer.pad_token is not None: 
    print("tokenizer has pad token {}".format(tokenizer.pad_token)) 
else: 
    tokenizer.pad_token = tokenizer.eos_token 
    print("We now use eos_token as pad token") 
tokenizer.padding_side = "left" 
# model = LlamaForCausalLM.from_pretrained(args.model, device_map = args.device, torch_dtype = torch.bfloat16) 
model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map = args.device, torch_dtype = torch.bfloat16) 

### Loading the datasets ### 
def get_dataset(datasetname): 
    # loading the manually written cot prompt 
    cotprompt: str = None 
    with open("{}_cot_prompts.txt".format(datasetname), "r") as file: 
        cotprompt = file.read() 
    if datasetname == "csqa": 
        # loading the actual dataset 
        dataset = load_dataset("tau/commonsense_qa", split = "test") 
        def encodewithtokenizer(example): 
            options = example["choices"]["text"] 
            inputtext = "Q: {}\nOptions: (a) {} (b) {} (c) {} (d) {} (e) {}\nA:".format(example["question"], options[0], options[1], options[2], options[3], options[4]) 
            return tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False) 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
    elif datasetname == "strategyqa": 
        dataset = load_dataset("tasksource/bigbench", "strategyqa", split = "validation") 
        def encodewithtokenizer(example): 
            inputtext = "Q: Yes or No: {}".format(example["inputs"][3 :]) 
            return tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False) 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
    elif datasetname == "date": 
        dataset = load_dataset("tasksource/bigbench", "date_understanding") 
        dataset = concatenate_datasets([dataset["train"], dataset["validation"]]) 
        def encodewithtokenizer(example): 
            inputtext = example["inputs"] 
            return tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False) 
        dataset = dataset[10: ] 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
    elif datasetname == "sports": 
        dataset = load_dataset("tasksource/bigbench", "sports_understanding") 
        dataset = concatenate_datasets([dataset["train"], dataset["validation"]]) 
        def encodewithtokenizer(example): 
            inputtext = "Q: {}".format(example["inputs"]) 
            return tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False) 
        dataset = dataset[10: ] 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
    else: 
        raise ValueError("Unknown dataset {}".format(datasetname)) 
        
    dataloader = DataLoader(dataset, batch_size = 1, shuffle = False) 
    return dataloader, cotprompt 

### Generate with custom termination condition ### 
class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )

for task in tasks: 
    dataloader, cotprompt = get_dataset(task) 
    promptids = tokenizer(cotprompt, return_tensors = "pt", truncation = True, padding = False)["input_ids"] 
    promptids = torch.tensor(promptids, dtype = torch.long) 
    for batch in dataloader: 
        input_ids = batch["input_ids"] 
        input_ids = torch.cat([promptids, input_ids], dim = 1) 
        input_ids = input_ids.to(model.device) 
        stop_criteria = stop_sequences_criteria(tokenizer, "Q:", input_ids.shape[1], input_ids.shape[0]) 
        
        outputs = model.generate(
            input_ids = input_ids, 
            max_length = 500, 
            stopping_criteria = stop_criteria, 
            pad_token_id = tokenizer.pad_token_id, 
            do_sample = False, 
        ) 
        print(tokenizer.decode(outputs[0])) 
        exit(0) 
