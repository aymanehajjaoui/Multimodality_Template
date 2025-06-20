/*SystemUtils.hpp*/

#pragma once

#include <sys/statvfs.h>
#include <chrono>
#include <thread>
#include <sys/stat.h>
#include <iomanip>
#include <csignal>
#include <dirent.h>
#include <iostream>
#include <string>
#include "Common.hpp"
#include "Channel.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"

extern Channel1 *g_channel1_ptr;
extern Channel2 *g_channel2_ptr;

bool is_disk_space_below_threshold(const char *path, double threshold);
bool set_thread_priority(std::thread &th, int priority);
bool set_thread_affinity(std::thread &th, int core_id);
bool set_current_thread_affinity(int core_id);
void signal_handler(int sig);
void print_duration(const std::string &label, uint64_t start_ns, uint64_t end_ns);
void folder_manager(const std::string &folder_path);
bool ask_user_preferences(bool &save_data_csv, bool &save_data_dac, bool &save_data_tcp,
                          bool &save_output_csv, bool &save_output_dac, bool &save_output_tcp);

template <typename In, typename Out>
void print_channel_stats(const Channel<In, Out> &channel)
{
    std::cout << "\n====================================\n\n";

    print_duration(channel.name,
                   channel.trigger_time_ns.load(),
                   channel.end_time_ns.load());

    std::cout << std::left << std::setw(60) << "Total data acquired:" << channel.acquire_count.load() << '\n';

    if (save_data_csv)
        std::cout << std::left << std::setw(60) << "Lines written to CSV:" << channel.write_count_csv.load() << '\n';
    if (save_data_dac)
        std::cout << std::left << std::setw(60) << "Lines written to DAC:" << channel.write_count_dac.load() << '\n';
    if (save_data_tcp)
        std::cout << std::left << std::setw(60) << "Lines sent over TCP:" << channel.write_count_tcp.load() << '\n';

    std::cout << std::left << std::setw(60) << "Total model inferences:" << channel.model_count.load() << '\n';

    if (save_output_csv)
        std::cout << std::left << std::setw(60) << "Model results to CSV:" << channel.log_count_csv.load() << '\n';
    if (save_output_dac)
        std::cout << std::left << std::setw(60) << "Model results to DAC:" << channel.log_count_dac.load() << '\n';
    if (save_output_tcp)
        std::cout << std::left << std::setw(60) << "Model results to TCP:" << channel.log_count_tcp.load() << '\n';

    std::cout << "\n====================================\n";
}
