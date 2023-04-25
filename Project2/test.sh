nvidia-smi --query-gpu=power.draw,temperature.gpu,memory.used --format=csv --loop-ms=1000 > ./experiments/pytorch/experiment-test.csv &
serverPID=$!
date +%Y-%m-%d_%H-%M-%S.%N > ./log
python ./src/training_tensorflow.py --dataset=CIFAR10 --resnet_size="resnet50" > ./log
date +%Y-%m-%d_%H-%M-%S.%N > ./log
kill serverPID