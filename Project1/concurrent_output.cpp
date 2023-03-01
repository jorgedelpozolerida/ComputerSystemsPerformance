#include <iostream>
#include "includes/utils.hpp"
#include <thread>
#include <vector>
#include <chrono>
#include <fstream>
#include <mutex>
#include <atomic>
#include <fstream>

const int INPUT_SIZE = 2000000;
const int MAX_NUM_THREADS = 32;
const int MAX_HASH_BITS = 32;

std::atomic<int> *sharedIndices;

// cocurrency primitives needed - the index does not need to atomic because it's incremented while it's locked
std::mutex *mut;

// u64 is defined in utils.hpp - it is an alias for usigned long long
u64* generate_input()
{
    u64* generated = new u64[INPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        generated[i] = (((u64)RAND()<<48) ^ ((u64)RAND()<<35) ^ ((u64)RAND()<<22) ^
                            ((u64)RAND()<< 9) ^ ((u64)RAND()>> 4));
    }

    return generated;
}

// we might want to pass the buffer by reference
void process_partition(u64* data, int start, int end, int hash_bits, std::tuple<u64, u64>** buffer, int max_part_s)
{
    utils util_obj; 

    for(size_t i = start; i < end; i++)
    {
        int hash = util_obj.hash(data[i], hash_bits);
        std::tuple<int, u64> t = std::make_tuple(hash, data[i]);

        std::unique_lock<std::mutex> lock(mut[hash]);
        buffer[hash][sharedIndices[hash]] = t;
        sharedIndices[hash] += 1;

        lock.unlock();
    }
}

int64_t run_experiment(int hash_bits, int num_threads, u64* &input)
{
    // maximum hash value
    int max_partition_hash = utils::max_partition_hash_static(hash_bits);

    // random data to be partitioned
    const int thread_divide_size = INPUT_SIZE / num_threads;
    int partition_buffer_size = (INPUT_SIZE/(max_partition_hash))*hash_bits;
    std::vector<std::thread> threads;

    std::cout << "Threads: " << num_threads <<"\n";
    std::cout << "Maximum output size: " << max_partition_hash <<"\n";

    if(partition_buffer_size < 5){
        partition_buffer_size = 5;
    }

    std::cout << "Maximum partition buffer size: " << partition_buffer_size <<"\n";
    std::cout << "Maximum: " << INPUT_SIZE/(max_partition_hash) <<"\n";

    // reset the shared index and create the buffer
    sharedIndices = new std::atomic<int>[max_partition_hash+1];
    mut = new std::mutex[max_partition_hash+1];

    std::tuple<u64, u64>** output_buffer = new std::tuple<u64, u64>*[max_partition_hash+1]; // max hash includes 0, so we do +1
    for(int i = 0; i <= max_partition_hash; i++){
        output_buffer[i] = new std::tuple<u64, u64>[partition_buffer_size];
        sharedIndices[i] = 0;
    }

    // create threads
    for (int i = 0; i < num_threads; ++i)
    {
        // set which thread processes what part of the data
        size_t start = i * thread_divide_size;
        size_t end = (i + 1) * thread_divide_size;
        if (i == num_threads - 1)
        {
            end = INPUT_SIZE;
        }

        // create the thread
        std::thread thread(process_partition, input, start, end, hash_bits, std::ref(output_buffer), partition_buffer_size);
        threads.push_back(std::move(thread));
    }

    // measuring the time
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_threads; ++i)
    {
        threads[i].join();
    }
    auto end = std::chrono::steady_clock::now();
    
    // calculate the elapsed time
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::chrono::milliseconds elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_seconds);
    
    std::cout << "elapsed time: " << elapsed_ms.count() << "ms\n";

    int highest = 0;
    for(int i = 0; i <= max_partition_hash; i++){
        delete[] output_buffer[i];

        // if(sharedIndices[i] > highest){
        //     highest = sharedIndices[i];
        // }
    }

    std::cout << "Highest index: " << highest << "\n\n";

    delete[] output_buffer;
    delete[] sharedIndices;
    delete[] mut;
    
    return elapsed_ms.count();
}

int main()
{
    srand(time(NULL));
    u64* input = generate_input();

    for (int experiment = 1; experiment <= 8; experiment += 1) 
    {
        std::string filename = "./experiments/concurrent_output/experiment_" + std::to_string(experiment) + ".csv";
        std::ofstream fout(filename);

        fout << "Threads;Hash_Bits;Running Time (ms)\n";

        for (int hash_bits = 21; hash_bits <= MAX_HASH_BITS; hash_bits += 1) 
        {
            std::cout << " HASH BITS: " << hash_bits <<"\n";

            for (int num_threads = 1; num_threads <= MAX_NUM_THREADS; num_threads *= 2) 
            {
                int64_t exp = run_experiment(hash_bits, num_threads, input);
                fout << num_threads << ";" << hash_bits << ";" << exp << "\n";
            }
        }

        fout.close();
    }

    delete[] input;
}