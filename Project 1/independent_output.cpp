#include <iostream>
#include "includes/utils.hpp"
#include <string.h>
#include <thread>
#include <vector>

// 8192 bytes are 8 KiB
const int INPUT_SIZE = 8192;
const int NUM_THREADS = 4;

unsigned char* generate_input()
{
    unsigned char* generated = new unsigned char[INPUT_SIZE];
    srand(INPUT_SIZE); // set a seed

    for (int i = 0; i < INPUT_SIZE; i++)
    {
        generated[i] = (unsigned char)(rand());
    }

    return generated;
}

void process_partition(unsigned char* data, int start, int end, std::vector<std::tuple<unsigned int, unsigned char>> &partition)
{
    for(size_t i = start; i < end; i++)
    {
        int hash = utils::hash(data[i], NUM_THREADS);
        partition.push_back(std::tuple<int, unsigned char>(hash, data[i]));
    }
}

int main()
{
    unsigned char* input = generate_input();
    const int partition_size = INPUT_SIZE / NUM_THREADS;

    // our hash will be at most 18 bits
    std::vector<std::vector<std::tuple<int, unsigned char>>>* partitions = new std::vector<std::vector<std::tuple<int, unsigned char>>>();
    std::vector<std::thread> threads;

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        // set which thread processes what part of the data
        size_t start = i * partition_size;
        size_t end = (i + 1) * partition_size;
        if (i == NUM_THREADS - 1)
        {
            end = INPUT_SIZE;
        }
        
        // create the partition
        std::vector<std::tuple<int, unsigned char>> partition;
        // partitions->push_back(partition);

        // create the thread
        std::thread thread(process_partition, input, start, end, std::ref(partition));
        //threads.push_back(thread);
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    std::string s = "";
    std::cin >> s;

    delete input;
    delete partitions;
}