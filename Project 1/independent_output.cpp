#include <iostream>
#include "includes/utils.hpp"
#include <string.h>
#include <thread>
#include <vector>

const int INPUT_SIZE = 8192;
const int NUM_THREADS = 1;
const int HASH_BITS = 8;

// u64 is defined in utils.hpp - it is an alias for usigned long long
u64* generate_input()
{
    u64* generated = new u64[INPUT_SIZE];
    srand(INPUT_SIZE); // set a seed

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        generated[i] = (((u64)RAND()<<48) ^ ((u64)RAND()<<35) ^ ((u64)RAND()<<22) ^
                            ((u64)RAND()<< 9) ^ ((u64)RAND()>> 4));
    }

    return generated;
}

void process_partition(u64* data, int start, int end, std::vector<std::vector<std::tuple<u64, u64>>> &partitions)
{
    for(size_t i = start; i < end; i++)
    {
        int hash = utils::hash(data[i], HASH_BITS);
        std::tuple tuple = std::make_tuple(hash, data[i]);
        std::cout << "(" << hash << ", " << data[i] << ")\n";
        
        // lock
        partitions.at(hash).push_back(tuple);
        // unlock
    }
}

int main()
{
    u64* input = generate_input();
    const int partition_size = INPUT_SIZE / NUM_THREADS;

    // our hash will be at most 18 bits
    std::vector<std::vector<std::tuple<u64, u64>>> partitions;
    std::vector<std::thread> threads;

    // create "partitions"
    int max_partition_hash = utils::max_partition_hash(HASH_BITS);
    for (size_t i = 0; i < max_partition_hash; i++)
    {
        std::vector<std::tuple<u64, u64>> partition;
        partitions.push_back(partition);
    }
    

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        // set which thread processes what part of the data
        size_t start = i * partition_size;
        size_t end = (i + 1) * partition_size;
        if (i == NUM_THREADS - 1)
        {
            end = INPUT_SIZE;
        }

        // create the thread
        std::thread thread(process_partition, input, start, end, std::ref(partitions));
        threads.push_back(std::move(thread));
    }

    for (std::thread &thread : threads)
    {
        thread.join();
        std::cout << "thread done\n";
    }

    delete input;

    std::string s = "";
    std::cin >> s;
}