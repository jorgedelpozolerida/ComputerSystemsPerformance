#include <chrono>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <vector>

int NUM_THREADS = 4;
std::mutex iomutex;

void work(int thread_index) {
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  while (1) {
    {
      // Thread-safe access to cout using a mutex:
      std::lock_guard<std::mutex> iolock(iomutex);
      std::cout << "Thread #" << thread_index << ": on CPU " 
                << sched_getcpu() << "\n";
    }
    //while(1) ; //comment-in if you want to see things more visibly with htop
    std::this_thread::sleep_for(std::chrono::milliseconds(900));
  }
}

int main(int argc, const char** argv) {
  std::vector<std::thread> threads(NUM_THREADS);
  for (int i = 0; i < NUM_THREADS; ++i) {
    threads[i] = std::thread(work, i);

    // Create a cpu_set_t object representing a set of CPUs.
    // Clear it and mark only CPU i as set.
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
  }

  for (auto& t : threads) {
    t.join();
  }
}
