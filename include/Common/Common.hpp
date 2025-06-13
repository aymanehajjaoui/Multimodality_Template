/*Common.hpp*/

#pragma once

#define WITH_CMSIS_NN 1
#define ARM_MATH_DSP 1
#define ARM_NN_TRUNCATE

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <deque>
#include <chrono>
#include <atomic>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <dirent.h>
#include <semaphore.h>

#include "rp.h"

#define DATA_SIZE 16384
#define QUEUE_MAX_SIZE 1000000
#define DISK_SPACE_THRESHOLD 0.2 * 1024 * 1024 * 1024

//#define acq_priority 1
//#define write_csv_priority 1
//#define write_dac_priority 1
#define model_priority 20
//#define log_csv_priority 1
//#define log_dac_priority 1
//#define log_tcp_priority 1

#define SHM_COUNTERS "/channel_counters"

extern bool save_data_csv;
extern bool save_data_dac;
extern bool save_output_csv;
extern bool save_output_dac;
extern bool save_output_tcp;

extern std::atomic<bool> stop_acquisition;
extern std::atomic<bool> stop_program;

extern pid_t pid1,pid2;

struct shared_counters_base_t {
    std::atomic<int> acquire_count;
    std::atomic<int> model_count;
    std::atomic<int> write_count_csv;
    std::atomic<int> write_count_dac;
    std::atomic<int> log_count_csv;
    std::atomic<int> log_count_dac;
    std::atomic<int> log_count_tcp;
    std::atomic<uint64_t> trigger_time_ns;
    std::atomic<uint64_t> end_time_ns;
    std::atomic<int> ready_barrier;
};