# accelerate launch --main_process_port 29502 --num_processes 4 main.py --tasks csqa --model meta-llama/Meta-Llama-3-8B-Instruct --griffin=True --check=True --kernel_size 16 --spr 0.5 --thr 0.1 
accelerate launch --main_process_port 29502 --num_processes 4 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats --limit 50 
# python main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive 
# python main.py --tasks aqua --model meta-llama/Llama-2-7b-chat-hf --shotfive 
