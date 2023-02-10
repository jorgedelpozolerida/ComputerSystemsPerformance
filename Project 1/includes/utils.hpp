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
    static int hash(char input, int max_size){
        int p = INT_MAX-1; // large prime number
        int a = (p-1) * ((double)rand()) / RAND_MAX;
        int b = (p-1) * ((double)rand()) / RAND_MAX;
        int c = (int)input;

        return ((a*c + b) % p) % max_size;
    }
};

// constructor
utils::utils(/* args */){}

// destructor
utils::~utils(){}