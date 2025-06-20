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
#define acq_priority 1
#define write_csv_priority 1
#define write_dac_priority 1
#define model_priority 20
#define log_csv_priority 1
#define log_dac_priority 1
#define log_tcp_priority 1

constexpr size_t DISK_SPACE_THRESHOLD = static_cast<size_t>(0.2 * 1024 * 1024 * 1024);

extern std::atomic<bool> stop_acquisition;
extern std::atomic<bool> stop_program;

extern bool save_data_csv, save_data_dac, save_data_tcp, save_output_csv, save_output_dac, save_output_tcp;