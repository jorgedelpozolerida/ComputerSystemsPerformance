nvidia-smi --query-gpu=power.draw,temperature.gpu,memory.used --format=csv --loop-ms=1000 > ./experiments/pytorch/experiment-test.csv &
serverPID=$!
python ./src/training_tensorflow.py --dataset=CIFAR10 --resnet_size="resnet50"
kill serverPID