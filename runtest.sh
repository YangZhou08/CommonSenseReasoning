# accelerate launch --main_process_port 29502 --num_processes 4 main.py --tasks csqa --model meta-llama/Meta-Llama-3-8B-Instruct --griffin=True --check=True --kernel_size 16 --spr 0.5 --thr 0.1 
python main.py --tasks csqa --model meta-llama/Meta-Llama-3-8B-Instruct --griffin=True --check=True --kernel_size 16 --spr 0.5 --thr 0.1 
