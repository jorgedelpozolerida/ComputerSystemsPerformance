#include<cstdlib>
#include <chrono>
#include <time.h>

// ensure only 15-bits
#define RAND() (rand() & 0x7fff)

// u64 is 8 bytes of size
typedef unsigned long long u64;

class utils
{
private:
    /* data */
public:
    utils(/* args */);
    ~utils();

    // given an input, output a hash of the output that is 0 >= max_size
    // h(c, m) = ((ac+b) mod p) mod m
    u64 hash(u64 input, int hash_bits){
        u64 num = (((u64)RAND()<<48) ^ ((u64)RAND()<<35) ^ ((u64)RAND()<<22) ^
                   ((u64)RAND()<< 9) ^ ((u64)RAND()>> 4));

        return num % max_partition_hash(hash_bits);
    }

    u64 max_partition_hash(int hash_bits){
        return (0xffffffffffffffff << hash_bits) ^ 0xffffffffffffffff;
    }

    static u64 max_partition_hash_static(int hash_bits){
        return (0xffffffffffffffff << hash_bits) ^ 0xffffffffffffffff;
    }
};

// constructor
utils::utils(/* args */){}

// destructor
utils::~utils(){}