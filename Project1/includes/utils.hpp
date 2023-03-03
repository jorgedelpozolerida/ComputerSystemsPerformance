#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include<cstdlib>
#include <chrono>
#include <time.h>
#include "picosha.h"
#include <cstring>
#include <bitset>


// ensure only 15-bits
#define RAND() (rand() & 0x7fff)

// u64 is 8 bytes of size
typedef unsigned long long u64;

class utils
{
public:
    utils(/* args */);
    ~utils();

    u64 hash(u64 input, int hash_bits){
        std::string hash = picosha2::hash256_hex_string(std::to_string(input));
        u64 num = 0;

        // get only the first 32 bytes of the hash
        for(int i = 0; i < 32; i++){
            // shift each character 2 bits and add it to the hash with XOR
            u64 temp = (u64)hash.at(i);
            num = num ^ (temp << (2*i));
        }

        // mask the hash to fit our max hash value
        return num & (max_partition_hash(hash_bits));
    }

    u64 max_partition_hash(int hash_bits){
        return (0xffffffffffffffff << hash_bits) ^ 0xffffffffffffffff;
    }

    static u64 max_partition_hash_static(int hash_bits){
        return (0xffffffffffffffff << hash_bits) ^ 0xffffffffffffffff;
    }
};

// constructor
utils::utils(/* args */){
}

// destructor
utils::~utils(){}
#endif