# accelerate launch --main_process_port 29502 --num_processes 4 main.py --tasks csqa --model meta-llama/Meta-Llama-3-8B-Instruct --griffin=True --check=True --kernel_size 16 --spr 0.5 --thr 0.1 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats --check --kernel_size 16 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 
# python main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --widthtree 8 
# accelerate launch --main_process_port 29501 --num_processes 6 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --widthtree 1 
# accelerate launch --main_process_port 29501 --num_processes 6 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --widthtree 2 
# accelerate launch --main_process_port 29501 --num_processes 6 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --widthtree 4 
accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks strategyqa --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive 
accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks strategyqa --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats --spr 0.4 
accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks strategyqa --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --spr 0.4 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats --check --kernel_size 16 
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 
# treewidth=(2) 
# for treew in ${treewidth[@]}; do
#     accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats --check --kernel_size 16 --widthtree $treew --patternstrict --filteractiveenabled 
# done

# treewidth=(2) 
# for treew in ${treewidth[@]}; do
#     accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --widthtree $treew --filteractiveenabled 
# done
# accelerate launch --main_process_port 29501 --num_processes 8 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin 
# accelerate launch --main_process_port 29501 --num_processes 6 main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --griffin --check --kernel_size 16 --widthtree 4 
# python main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive --cats 
# python main.py --tasks aqua --model meta-llama/Meta-Llama-3-8B-Instruct --shotfive 
# python main.py --tasks aqua --model meta-llama/Llama-2-7b-chat-hf --shotfive 
