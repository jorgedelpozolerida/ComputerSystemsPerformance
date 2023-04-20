#include <chrono>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <thread>
#include <vector>

int NUM_INCREMENTS = 10000;
std::mutex countermutex;
int counter = 0;

using namespace std;
using namespace std::chrono;

void work(int thread_index) {
  for (int i = 0; i < NUM_INCREMENTS; i++) {
    lock_guard<mutex> counterlock(countermutex);
    counter++;
  }
}

int main(int argc, const char** argv) {
  cout << "Example usage:" << endl << "./affinitynumaexample 4 0 8 16 24" << endl;

  if (argc < 2) {
      cout << "Require atleast 2 arguments, number of threads and then CPU id for every thread." << endl;
      return -1;
  }
  auto n_threads = atoi(argv[1]);

  if (argc < n_threads + 1){
    cout << "CPU id must be specified for every thread" << endl;
    return -1;
  }

  cout << "Starting number of threads: " << n_threads << endl;

  auto start = high_resolution_clock::now();

  vector<thread> threads(n_threads);
  for (int i = 0; i < n_threads; ++i) {
    cout << "CPU: " << atoi(argv[2 + i]) << endl;
    threads[i] = thread(work, i);

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(atoi(argv[2 + i]), &cpuset);
    int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      cerr << "Error calling pthread_setaffinity_np: " << rc << endl;
    }
  }

  for (auto& t : threads) {
    t.join();
  }

  auto stop = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(stop - start);
  cout << "Counter value: " << counter << endl;
  cout << "Duration: " << duration.count() << " microseconds" << endl;
}
