import torch 
from torch.utils.data.distributed import DistributedSampler 
import torch.distributed as dist 
import transformers 
from accelerate import Accelerator 
import lm_eval 
from datasets import load_dataset 
from transformers import AutoTokenizer, LlamaForCausalLM 
import numpy as np 
from datasets import concatenate_datasets 
from torch.utils.data import DataLoader 
from typing import List, Literal, Optional, Tuple, Union 
import argparse 
from tqdm import tqdm 
from termcolor import colored 

### Parsing the arguments ### 
parser = argparse.ArgumentParser(description = "CommonSense Reasoning with generation and chain-of-thoughts") 
parser.add_argument("--tasks", type = str) 
parser.add_argument("--model", type = str) 
parser.add_argument("--device", type = str, default = None) 
parser.add_argument("--limit", type = int, default = None) 

accelerator = Accelerator() 

# Check if we are in a distributed setup
is_distributed = accelerator.distributed_type != "NO"

args = parser.parse_args() 
tasks = args.tasks.split(",") 

if args.device is None: 
    # args.device = "cuda" if torch.cuda.is_available() else "cpu" 
    if is_distributed: 
        args.device = "cuda:{}".format(accelerator.process_index) if torch.cuda.is_available() else "cpu" 
    else: 
        args.device = "cuda" if torch.cuda.is_available() else "cpu" 

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
if is_distributed: 
    model = accelerator.prepare(model) 

### Loading the datasets ### 
def get_dataset(datasetname, is_distributed = False, requirements = ""): 
    # loading the manually written cot prompt 
    cotprompt: str = None 
    with open("{}_cot_prompts{}.txt".format(datasetname, requirements), "r") as file: 
        cotprompt = file.read() 
        cotprompt = cotprompt.replace("\\n", "") 
        cotprompt = cotprompt.replace("\\", "") 
    if datasetname == "csqa": 
        # loading the actual dataset 
        if args.limit is None: 
            dataset = load_dataset("tau/commonsense_qa", split = "validation") 
        else: 
            dataset = load_dataset("tau/commonsense_qa", split = "validation[:{}]".format(args.limit)) 
        def encodewithtokenizer(example): 
            options = example["choices"]["text"] 
            inputtext = "Q: {}\nOptions: (a) {} (b) {} (c) {} (d) {} (e) {}\nA:".format(example["question"], options[0], options[1], options[2], options[3], options[4]) 
            return tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False, add_special_tokens = False) 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
        print("length of dataset: ", len(dataset)) 
    elif datasetname == "strategyqa": 
        if args.limit is None: 
            dataset = load_dataset("tasksource/bigbench", "strategyqa", split = "validation") 
        else: 
            dataset = load_dataset("tasksource/bigbench", "strategyqa", split = "validation[:{}]".format(args.limit)) 
        def encodewithtokenizer(example): 
            inputtext = "Q: Yes or No: {}".format(example["inputs"][3 :]) 
            return tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False, add_special_tokens = False) 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
    elif datasetname == "date": 
        dataset = load_dataset("tasksource/bigbench", "date_understanding") 
        dataset = concatenate_datasets([dataset["train"], dataset["validation"]]) 
        def encodewithtokenizer(example): 
            inputtext = example["inputs"] 
            return tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False, add_special_tokens = False) 
        dataset = dataset.select(range(10, len(dataset))) 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
    elif datasetname == "sports": 
        dataset = load_dataset("tasksource/bigbench", "sports_understanding") 
        dataset = concatenate_datasets([dataset["train"], dataset["validation"]]) 
        def encodewithtokenizer(example): 
            inputtext = "Q: {}".format(example["inputs"]) 
            return tokenizer(inputtext, return_tensors = "pt", truncation = True, padding = False, add_special_tokens = False) 
        dataset = dataset.select(range(10, len(dataset))) 
        dataset = dataset.map(encodewithtokenizer, num_proc = 8) 
    else: 
        raise ValueError("Unknown dataset {}".format(datasetname)) 
    
    if is_distributed: 
        distributedsampler = DistributedSampler(dataset, num_replicas = accelerator.num_processes, rank = accelerator.process_index) 
        dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, sampler = distributedsampler) 
    else: 
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

def criteriaoutput(datasetname, outputs, inputexample): 
    if datasetname == "csqa": 
        expectedanswer = inputexample["answerKey"][0].lower() 
        generatedtext = tokenizer.decode(outputs) 
        indexpinned = generatedtext.find("So the answer is ") 
        indexperiod = generatedtext.find(".", indexpinned) 
        # answer = generatedtext[indexpinned + len("So the answer is ") : indexperiod] 
        answer = generatedtext[indexperiod - 2] 
        # expectedanswer = batch["answerKey"][0].lower() 
        if accelerator.is_main_process: 
            if answer == expectedanswer: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "green")) 
            else: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "red")) 
        return int(answer == expectedanswer) 
    elif datasetname == "strategyqa": 
        expectedanswer = inputexample["multiple_choice_targets"][inputexample["multiple_choice_scores"].index(1)][0].lower() 
        generatedtext = tokenizer.decode(outputs) 
        indexpinned = generatedtext.find("So the answer is ") 
        indexperiod = generatedtext.find(".", indexpinned) 
        answer = generatedtext[indexpinned + len("So the answer is ") : indexperiod] 
        if accelerator.is_main_process: 
            if answer == expectedanswer: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "green")) 
            else: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "red")) 
        return int(answer == expectedanswer) 
    elif datasetname == "date": 
        expectedanswer = inputexample["targets"][0][0] 
        generatedtext = tokenizer.decode(outputs) 
        indexpinned = generatedtext.find("So the answer is ") 
        indexperiod = generatedtext.find(".", indexpinned) 
        answer = generatedtext[indexpinned + len("So the answer is ") : indexperiod] 
        resultoutput = False 
        if answer == expectedanswer: 
            resultoutput = True 
        else: 
            segsanswer = answer.split("/") 
            segsexpectedanswer = expectedanswer.split("/") 
            if len(segsanswer) != len(expectedanswer): 
                resultoutput = False 
            else: 
                accumulate = True 
                for i in range(3): 
                    if segsexpectedanswer[i][0] == '0': 
                        segsexpectedanswer[i] = segsexpectedanswer[i][1 : ] 
                    if segsanswer[i][0] == '0': 
                        segsanswer[i] = segsanswer[i][1 : ] 
                    accumulate = accumulate and (segsanswer[i] == segsexpectedanswer[i]) 
                    print("answer {} expected {} accumulate {}".format(segsanswer[i], segsexpectedanswer[i], accumulate)) 
                resultoutput = accumulate 
        if accelerator.is_main_process: 
            if resultoutput: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "green")) 
            else: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "red")) 
        return int(resultoutput) 
    elif datasetname == "sports": 
        expectedanswer = inputexample["targets"][0][0] 
        generatedtext = tokenizer.decode(outputs) 
        indexpinned = generatedtext.find("So the answer is ") 
        indexperiod = generatedtext.find(".", indexpinned) 
        answer = generatedtext[indexpinned + len("So the answer is ") : indexperiod] 
        if accelerator.is_main_process: 
            if answer == expectedanswer: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "green")) 
            else: 
                print(colored("Answer {} expected {}".format(answer, expectedanswer), "red")) 
        return int(answer == expectedanswer) 
    else: 
        raise ValueError("Unknown dataset {}".format(datasetname)) 

print("tasks {}".format(tasks)) 
countaccum = {} 
for task in tasks: 
    # dataloader, cotprompt = get_dataset(task, requirements = "_5shot") 
    dataloader, cotprompt = get_dataset(task, is_distributed = is_distributed, requirements = "") 
    promptids = tokenizer(cotprompt, return_tensors = "pt", truncation = True, padding = False)["input_ids"] 
    promptids = torch.tensor(promptids, dtype = torch.long).to(args.device) 
    totalexamples = 0 
    correctanswers = 0 
    
    '''
    # make the kv cache 
    outputs = model(
        input_ids = promptids, 
        use_cache = True, 
        return_dict = True, 
    ) 
    kv_cache = outputs.past_key_values 
    ''' 
    
    for i, batch in enumerate(tqdm(dataloader)): 
        # print("answer found {}".format("answerKey" in batch.keys())) 
        # print(batch["answerKey"][0]) 
        # print(len(batch["answerKey"])) 
        # exit(0) 
        input_ids = batch["input_ids"] 
        input_ids = torch.tensor(input_ids, dtype = torch.long) 
        input_ids = input_ids.to(args.device) 
        if accelerator.is_main_process: 
            print(tokenizer.decode(input_ids[0])) 
        input_ids = torch.cat([promptids, input_ids], dim = 1) 
        input_ids = input_ids.to(args.device) 
        stop_criteria = stop_sequences_criteria(tokenizer, "Q:", input_ids.shape[1], input_ids.shape[0]) 
        if is_distributed: 
            outputs = model.module.generate(
                input_ids = input_ids, 
                attention_mask = None, 
                # max_length = input_ids.shape[1] + 20, 
                max_length = input_ids.shape[1] + 200, 
                use_cache = True, 
                stopping_criteria = stop_criteria, 
                pad_token_id = tokenizer.pad_token_id, 
                do_sample = False, 
                # past_key_values = kv_cache, 
            ) 
        else: 
            outputs = model.generate(
                input_ids = input_ids, 
                attention_mask = None, 
                max_length = input_ids.shape[1] + 200, 
                use_cache = True, 
                stopping_criteria = stop_criteria, 
                pad_token_id = tokenizer.pad_token_id, 
                do_sample = False, 
            ) 
        # print(tokenizer.decode(outputs[0])) 
        if accelerator.is_main_process: 
            print(tokenizer.decode(outputs[0][input_ids.shape[1] :])) 
        generatedtext = tokenizer.decode(outputs[0][input_ids.shape[1] :]) 
        checkcriteria = criteriaoutput(task, outputs[0][input_ids.shape[1] :], batch) 
        totalexamples += 1 
        correctanswers += checkcriteria 
        if accelerator.is_main_process: 
            print("Total examples: {} Correct answers: {}".format(totalexamples, correctanswers)) 
        
    dist.all_reduce(torch.tensor(totalexamples, device = args.device), op = dist.ReduceOp.SUM) 
    dist.all_reduce(torch.tensor(correctanswers, device = args.device), op = dist.ReduceOp.SUM) 
    countaccum[task] = [totalexamples, correctanswers, correctanswers / totalexamples] 

if accelerator.is_main_process: 
    # formatting the output 
    print("Task\tTotal\tCorrect\tSolve Rate") 
    for task in tasks: 
        print("{}\t{}\t{}\t{}".format(task, countaccum[task][0], countaccum[task][1], countaccum[task][2])) 
