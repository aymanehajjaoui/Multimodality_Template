/*SystemUtils*/

#include "SystemUtils.hpp"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <type_traits>

Channel1 *g_channel1_ptr = nullptr;
Channel2 *g_channel2_ptr = nullptr;

bool is_disk_space_below_threshold(const char *path, double threshold)
{
    struct statvfs stat;
    if (statvfs(path, &stat) != 0)
    {
        std::cerr << "Error getting filesystem statistics." << std::endl;
        return false;
    }

    double available_space = stat.f_bsize * stat.f_bavail;
    return available_space < threshold;
}

bool set_thread_priority(std::thread &th, int priority)
{
    struct sched_param param;
    param.sched_priority = priority;

    if (pthread_setschedparam(th.native_handle(), SCHED_FIFO, &param) != 0)
    {
        std::cerr << "Failed to set thread priority to " << priority << std::endl;
        return false;
    }
    return true;
}

bool set_thread_affinity(std::thread &th, int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset) != 0)
    {
        std::cerr << "Failed to set thread affinity to Core " << core_id << std::endl;
        return false;
    }
    return true;
}

bool set_current_thread_affinity(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0)
    {
        std::cerr << "Failed to set current thread affinity to Core " << core_id << std::endl;
        return false;
    }
    return true;
}

void signal_handler(int sig)
{
    if (sig == SIGINT)
    {
        std::cout << "^CSIGINT received, initiating graceful shutdown..." << std::endl;
        stop_program.store(true);
        stop_acquisition.store(true);
        std::cin.setstate(std::ios::failbit);

        if (g_channel1_ptr)
        {
            sem_post(&g_channel1_ptr->data_sem_csv);
            sem_post(&g_channel1_ptr->data_sem_dac);
            sem_post(&g_channel1_ptr->data_sem_tcp);
            sem_post(&g_channel1_ptr->model_sem);
            sem_post(&g_channel1_ptr->result_sem_csv);
            sem_post(&g_channel1_ptr->result_sem_dac);
            sem_post(&g_channel1_ptr->result_sem_tcp);
        }

        if (g_channel2_ptr)
        {
            sem_post(&g_channel2_ptr->data_sem_csv);
            sem_post(&g_channel2_ptr->data_sem_dac);
            sem_post(&g_channel2_ptr->data_sem_tcp);
            sem_post(&g_channel2_ptr->model_sem);
            sem_post(&g_channel2_ptr->result_sem_csv);
            sem_post(&g_channel2_ptr->result_sem_dac);
            sem_post(&g_channel2_ptr->result_sem_tcp);
        }
    }
}

void print_duration(const std::string &label, uint64_t start_ns, uint64_t end_ns)
{
    auto duration_ns = end_ns > start_ns ? end_ns - start_ns : 0;
    auto duration_ms = duration_ns / 1'000'000;

    auto minutes = duration_ms / 60000;
    auto seconds = (duration_ms % 60000) / 1000;
    auto ms = duration_ms % 1000;

    std::cout << std::left << std::setw(40) << label + " acquisition time:"
              << minutes << " min " << seconds << " sec " << ms << " ms\n";
}

void folder_manager(const std::string &folder_path)
{
    namespace fs = std::filesystem;

    try
    {
        fs::path dir_path(folder_path);

        if (fs::exists(dir_path))
        {
            for (const auto &entry : fs::directory_iterator(dir_path))
            {
                try
                {
                    fs::remove_all(entry);
                }
                catch (const fs::filesystem_error &e)
                {
                    std::cerr << "Failed to delete: " << entry.path() << " - " << e.what() << std::endl;
                }
            }
        }
        else
        {
            if (!fs::create_directories(dir_path))
            {
                std::cerr << "Failed to create directory: " << folder_path << std::endl;
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
}

bool ask_user_preferences(bool &save_data_csv, bool &save_data_dac, bool &save_data_tcp,
                          bool &save_output_csv, bool &save_output_dac, bool &save_output_tcp)
{
    int max_attempts = 3;

    for (int attempt = 1; attempt <= max_attempts; ++attempt)
    {
        int choice;
        std::cout << "Save acquired data?\n"
                  << "1. CSV only\n2. DAC only\n3. TCP only\n4. CSV + DAC\n5. All\n6. None\n"
                  << "Enter choice (1-6): ";
        std::cin >> choice;

        if (choice >= 1 && choice <= 6)
        {
            save_data_csv = (choice == 1 || choice == 4 || choice == 5);
            save_data_dac = (choice == 2 || choice == 4 || choice == 5);
            save_data_tcp = (choice == 3 || choice == 5);
            break;
        }

        std::cerr << "Invalid input. Try again.\n";
        if (attempt == max_attempts)
            return false;
    }

    for (int attempt = 1; attempt <= max_attempts; ++attempt)
    {
        int choice;
        std::cout << "\nSave model output?\n"
                  << "1. CSV\n2. DAC\n3. TCP\n4. CSV + DAC\n5. All\n6. None\n"
                  << "Enter choice (1-6): ";
        std::cin >> choice;

        if (choice >= 1 && choice <= 6)
        {
            save_output_csv = (choice == 1 || choice == 4 || choice == 5);
            save_output_dac = (choice == 2 || choice == 4 || choice == 5);
            save_output_tcp = (choice == 3 || choice == 5);
            return true;
        }

        std::cerr << "Invalid input. Try again.\n";
        if (attempt == max_attempts)
            return false;
    }

    return true;
}
