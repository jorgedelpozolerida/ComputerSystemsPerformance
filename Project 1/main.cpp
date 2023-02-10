#include "includes/thread_pool.hpp"
#include "includes/utils.hpp"

int main(){
    int test1 = utils::hash('c', 5);
    int test2 = utils::hash('c', 5);
    int test3 = utils::hash('c', 5);
    int test4 = utils::hash('c', 5);

    std::cout << test1 << std::endl;
    std::cout << test2 << std::endl;
    std::cout << test3 << std::endl;
    std::cout << test4 << std::endl;
}