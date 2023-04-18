for i in {1..8}
do
    nvidia-smi --query-gpu=power.draw,temperature.gpu,memory.used --format=csv --loop-ms=100 > ./experiments/tensorflow/experiment-$i.csv
    & python ./training_pytorch.py
done

for i in {1..8}
do
    nvidia-smi --query-gpu=power.draw,temperature.gpu,memory.used --format=csv --loop-ms=100 > ./experiments/pytorch/experiment-$i.csv
    & python ./training_tensorflow.py
done