#include <iostream>
#include "includes/utils.hpp"
#include <string.h>
#include <thread>
#include <vector>
#include <chrono>
#include <fstream>

const int INPUT_SIZE = 2000000;

struct process_parameters{
    int start; 
    int end; 
    int hash_bits; 
    std::vector<std::vector<std::tuple<u64, u64>>> *buffer;
};

// u64 is defined in utils.hpp - it is an alias for usigned long long
u64 *generate_input()
{
    u64 *generated = new u64[INPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        generated[i] = (((u64)RAND() << 48) ^ ((u64)RAND() << 35) ^ ((u64)RAND() << 22) ^
                        ((u64)RAND() << 9) ^ ((u64)RAND() >> 4));
    }

    return generated;
}

// we might want to pass the buffer by reference
void process_partition(u64 *data, int start, int end, int hash_bits, std::vector<std::vector<std::tuple<u64, u64>>> *buffer)
{
    utils util_obj; 

    for (size_t i = start; i < end; i++)
    {
        int hash = util_obj.hash(data[i], hash_bits);
        std::tuple<int, u64> t = std::make_tuple(hash, data[i]);

        buffer->at(hash).push_back(t);
    }
}

int64_t run_experiment(int hash_bits, int num_threads, u64* &input)
{
    const int partition_size = INPUT_SIZE / num_threads;

    std::vector<std::thread> threads;
    process_parameters* thread_parameters = new process_parameters[num_threads];
    std::vector<std::vector<std::vector<std::tuple<u64, u64>>>> thread_buffers;

    for (int i = 0; i < num_threads; ++i)
    {
        std::vector<std::vector<std::tuple<u64, u64>>>* output_buffer = new std::vector<std::vector<std::tuple<u64, u64>>>();

        // create "partitions"
        u64 max_partition_hash = utils::max_partition_hash_static(hash_bits);
        for (u64 i = 0; i <= max_partition_hash; i++)
        {
            std::vector<std::tuple<u64, u64>> partition_buffer;
            output_buffer->push_back(partition_buffer);
        }

        // set which thread processes what part of the data
        size_t start = i * partition_size;
        size_t end = (i + 1) * partition_size;
        if (i == num_threads - 1)
        {
            end = INPUT_SIZE;
        }

        process_parameters params;
        params.buffer = output_buffer;
        params.end = end;
        params.start = start;
        thread_parameters[i] = params;
    }

    // measuring the time
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < num_threads; i++)
    {
        // create the thread
        std::thread thread(process_partition, input, thread_parameters[i].start, thread_parameters[i].end, hash_bits, std::ref(thread_parameters[i].buffer));
        threads.push_back(std::move(thread));
    }
    
    for (std::thread &thread : threads)
    {
        thread.join();
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::chrono::milliseconds elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds);

    // clean-up
    for (size_t i = 0; i < num_threads; i++)
    {
        delete thread_parameters[i].buffer;
    }
    delete[] thread_parameters;

    return elapsed_ms.count();
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    u64* input = generate_input();

    int hash_bits = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    int64_t exp = run_experiment(hash_bits, num_threads, input);
    std::cout << num_threads << ";" << hash_bits << ";" << exp << "\n";

    delete[] input;
}