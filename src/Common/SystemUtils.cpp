/*SystemUtils*/

#include "SystemUtils.hpp"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <type_traits>
#include <cctype>

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
    auto ask_yes_no = [](const std::string &prompt) -> bool {
        char input;
        while (true)
        {
            std::cout << prompt;
            std::cin >> input;
            input = std::tolower(input);

            if (input == 'y')
                return true;
            else if (input == 'n')
                return false;
            else
                std::cerr << "Invalid input. Please enter 'y' or 'n'.\n";
        }
    };

    std::cout << "Configure acquired data export:\n";
    save_data_csv = ask_yes_no("Save to CSV? (y/n): ");
    save_data_dac = ask_yes_no("Save to DAC? (y/n): ");
    save_data_tcp = ask_yes_no("Save to TCP? (y/n): ");

    std::cout << "\nConfigure model output export:\n";
    save_output_csv = ask_yes_no("Save to CSV? (y/n): ");
    save_output_dac = ask_yes_no("Save to DAC? (y/n): ");
    save_output_tcp = ask_yes_no("Save to TCP? (y/n): ");

    if (save_data_dac && save_output_dac)
    {
        std::cerr << "\nConflict detected: DAC cannot be used for both acquired data and model output.\n";
        bool assign_to_signal = ask_yes_no("Do you want to assign DAC to acquired data? (y/n): ");

        if (assign_to_signal)
        {
            save_output_dac = false;
            std::cout << "DAC assigned to acquired data. Disabled for model output.\n";
        }
        else
        {
            save_data_dac = false;
            std::cout << "DAC assigned to model output. Disabled for acquired data.\n";
        }
    }

    return true;
}