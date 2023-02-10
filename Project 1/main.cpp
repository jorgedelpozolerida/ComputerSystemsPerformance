#include "includes/thread_pool.hpp"
#include "includes/utils.hpp"
#include <string.h>

// 8192 bytes are 8 KiB
const int INPUT_SIZE = 8192;

unsigned char* generate_input(){
    unsigned char* generated = new unsigned char[INPUT_SIZE];

    for(int i = 0; i < INPUT_SIZE; i++){
        std::cout << (unsigned char)((INT32_MAX/i)^(INT32_MAX/i)) << " ";
        //generated[i] = (unsigned char)((INT32_MAX/i)^(INT32_MAX/i));
    }

    return generated;
}

int main(){
    unsigned char* input = generate_input();
    int num_partitions = 8;

    // for(int i = 0; i < INPUT_SIZE; i++){
    //     std::cout << "(" << input[i] << ", " << utils::hash('c', num_partitions) << ")" << " ";
    // }

    std::cout << std::endl;

    std::string s = "";
    std::cin >> s;
    
    // delete the reference to the input
    delete input;
}