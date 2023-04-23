#!/bin/bash

# variable 
experiment_runs=(1 2 3 4)
datasets=("SVHN" "CIFAR10" "CIFAR100")
models=("resnet50" "resnet101" "resnet152")

# fixed
batch_sizes=(128)
epochs=(10)
devices=("cpu")


# TENSORFLOW
framework="tensorflow"
for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for device in "${devices[@]}"
        do
            for batch_size in "${batch_sizes[@]}"
            do
                for epoch in "${epochs[@]}"
                do
                    for run in "${experiment_runs[@]}"
                    do
                        timestamp=$(date +%Y-%m-%d_%H-%M-%S-%3N)

                        filename_energy="run-${run}_device-${device}_epoch-${epoch}_batchsize-${batch_size}_framework-${framework}_dataset-${dataset}_model-${model}_ENERGY_${timestamp}"
                        filename_python="run-${run}_device-${device}_epoch-${epoch}_batchsize-${batch_size}_framework-${framework}_dataset-${dataset}_model-${model}_MODEL"

                        ./src/monitor_CPU.sh > "./experiments/${device}/${framework}/${filename_energy}.csv"  &
                        serverPID=$!
                        sleep 2
                        # python ./src/training_tensorflow.py --dataset "${dataset}" --resnet_size "${model}" --device "${device}" --batch_size "${batch_size}" --epochs "${epoch}" > "./experiments/${device}/${framework}/${filename_python}.csv" &
                        kill $serverPID
                        # wait
                    done
                done
            done
        done
    done
done

# PYTORCH
framework="pytorch"
for dataset in "${datasets[@]}"
do
    for model in "${models[@]}"
    do
        for device in "${devices[@]}"
        do
            for batch_size in "${batch_sizes[@]}"
            do
                for epoch in "${epochs[@]}"
                do
                    for run in "${experiment_runs[@]}"
                    do
                        timestamp=$(date +%Y-%m-%d_%H-%M-%S-%3N)

                        filename_energy="run-${run}_device-${device}_epoch-${epoch}_batchsize-${batch_size}_framework-${framework}_dataset-${dataset}_model-${model}_ENERGY_${timestamp}"
                        filename_python="run-${run}_device-${device}_epoch-${epoch}_batchsize-${batch_size}_framework-${framework}_dataset-${dataset}_model-${model}_MODEL"

                        ./src/monitor_CPU.sh > "./experiments/${device}/${framework}/${filename_energy}.csv"  &
                        serverPID=$!
                        sleep 2
                        # python ./src/training_pytorch.py --dataset "${dataset}" --resnet_size "${model}" --device "${device}" --batch_size "${batch_size}" --epochs "${epoch}" > "./experiments/${device}/${framework}/${filename_python}.csv" &
                        kill serverPID
                    done
                done
            done
        done
    done
done