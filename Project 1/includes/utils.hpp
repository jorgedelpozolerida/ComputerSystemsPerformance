#include<cstdlib>
#include <chrono>

class utils
{
private:
    /* data */
public:
    utils(/* args */);
    ~utils();

    // given an input, output a hash of the output that is 0 >= max_size
    // h(c, m) = ((ac+b) mod p) mod m
    static unsigned short hash(unsigned char input, int max_size){
        unsigned int p = INT_MAX; // large prime number
        unsigned int a = (p-1) * ((double)rand()) / RAND_MAX;
        unsigned int b = (p-1) * ((double)rand()) / RAND_MAX;
        int c = (int)input;

        return (unsigned short)((a*c + b) % p) % max_size;
    }
};

// constructor
utils::utils(/* args */){}

// destructor
utils::~utils(){}