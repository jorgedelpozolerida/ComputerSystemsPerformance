#!/bin/bash
for i in {1..8}
do
    echo "Threads;Hash_Bits;Running Time (ms)" > ./experiments/independent_output/experiment_$i.csv
    for hashbits in {1..22}
    do
        for threads in {1,2,4,8}
        do
            perf stat -o ./experiments/independent_output/perf_experiment_$i.txt ./independent_output $hashbits $threads >> ./experiments/independent_output/experiment_$i.csv
            #./independent_output $hashbits $threads >> ./experiments/independent_output/experiment_$i.csv
        done
    done
done

for i in {1..8}
do
    echo "Threads;Hash_Bits;Running Time (ms)" > ./experiments/concurrent_output/experiment_$i.csv
    for hashbits in {1..22}
    do
        for threads in {1,2,4,8}
        do
            perf stat -o ./experiments/concurrent_output/perf_experiment_$i.txt ./concurrent_output $hashbits $threads >> ./experiments/concurrent_output/experiment_$i.csv
            #./concurrent_output $hashbits $threads >> ./experiments/concurrent_output/experiment_$i.csv
        done
    done
done