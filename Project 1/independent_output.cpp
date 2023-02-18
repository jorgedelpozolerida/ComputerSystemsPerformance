#include <iostream>
#include "includes/utils.hpp"
#include <string.h>
#include <thread>
#include <vector>
#include <chrono>
#include <fstream>

const int INPUT_SIZE = 2000000;
// const int NUM_THREADS = 8;
const int HASH_BITS = 1;

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

// we might want to pass the buffer by reference
void process_partition(u64* data, int start, int end, int hash_bits,  std::vector<std::vector<std::tuple<u64, u64>>> buffer)
{
    for(size_t i = start; i < end; i++)
    {
        int hash = utils::hash(data[i], hash_bits);
        std::tuple<int, u64> t = std::make_tuple(hash, data[i]);
        //std::cout << "(" << hash << ", " << data[i] << ")\n";
        
        buffer.at(hash).push_back(t);
    }
}

int main()
{   
    for (int experiment = 1; experiment <= 8; experiment += 1) 
    {
        std::fstream fout;
        
        // Create a new file to store updated data
        
        std::string filename = "Project 1/experiments/independent_output/experiment_" + std::to_string(experiment) + ".csv";
        fout.open(filename, std::ios::out);

        fout << " ; "<< 1 << "; "<< 2 << "; "<< 4 << "; "<< 8 << "; "<< 16 << "; "<< 32 << "\n";

        for (int HASH_BITS = 1; HASH_BITS <= 18; HASH_BITS += 1) 
        {
            std::cout << " HASH BITS: " << HASH_BITS <<"\n";

            fout << HASH_BITS << "; ";
            for (int NUM_THREADS = 1; NUM_THREADS <= 32; NUM_THREADS *= 2) 
            {

                u64* input = generate_input();
                const int partition_size = INPUT_SIZE / NUM_THREADS;

                std::cout << "Threads: " << NUM_THREADS <<"\n";

                std::vector<std::thread> threads;
                std::vector<std::vector<std::vector<std::tuple<u64, u64>>>> thread_buffers;

                for (int i = 0; i < NUM_THREADS; ++i)
                {
                    std::vector<std::vector<std::tuple<u64, u64>>> output_buffer;

                    // create "partitions"
                    int max_partition_hash = utils::max_partition_hash(HASH_BITS);
                    for (size_t i = 0; i <= max_partition_hash; i++)
                    {
                        std::vector<std::tuple<u64, u64>> partition_buffer;
                        output_buffer.push_back(partition_buffer);
                    }

                    // set which thread processes what part of the data
                    size_t start = i * partition_size;
                    size_t end = (i + 1) * partition_size;
                    if (i == NUM_THREADS - 1)
                    {
                        end = INPUT_SIZE;
                    }

                    // create the thread
                    std::thread thread(process_partition, input, start, end, HASH_BITS, output_buffer);
                    threads.push_back(std::move(thread));
                }

                // measuring the time
                auto start = std::chrono::steady_clock::now();
                for (std::thread &thread : threads)
                {
                    thread.join();
                }
                auto end = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed_seconds = end-start;

                std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n\n";

                delete input;
                fout << elapsed_seconds.count() << "; ";

                // std::string s = "";
                // std::cin >> s;
            }
            fout << "\n";

        }  

        fout.close();
        // delete fout;
    }
 
}