/*ModelWriterCSV.cpp*/

#include "ModelWriterCSV.hpp"
#include <iostream>
#include <type_traits>
#include <cstdio>

template <typename T>
void write_output(FILE *file, int index, const T &value, double time_ms)
{
    if constexpr (std::is_integral<T>::value)
        fprintf(file, "%d,%d,%.6f\n", index, static_cast<int>(value), time_ms);
    else if constexpr (std::is_floating_point<T>::value)
        fprintf(file, "%d,%.6f,%.6f\n", index, value, time_ms);
    else
        fprintf(file, "%d,%d,%.6f\n", index, static_cast<int>(value), time_ms);
}

template <typename In, typename Out>
void log_results_csv(Channel<In, Out> &channel, const std::string &filename)
{
    try
    {
        FILE *output_file = fopen(filename.c_str(), "w");
        if (!output_file)
        {
            std::cerr << "Error opening output file: " << filename << "\n";
            return;
        }

        int output_index = 1;

        while (true)
        {
            if (is_disk_space_below_threshold("/", DISK_SPACE_THRESHOLD))
            {
                std::cerr << "ERR: Disk space below threshold. Stopping Writing." << std::endl;
                stop_program.store(true);
                break;
            }

            if (sem_wait(&channel.result_sem_csv) != 0)
            {
                if (errno == EINTR && stop_program.load())
                    break;
                continue;
            }

            if (stop_program.load() && channel.result_buffer_csv.empty())
                break;

            while (!channel.result_buffer_csv.empty())
            {
                const auto &result = channel.result_buffer_csv.front();
                write_output(output_file, output_index++, result.output[0], result.computation_time);
                fflush(output_file);
                channel.result_buffer_csv.pop_front();
                channel.log_count_csv.fetch_add(1, std::memory_order_relaxed);
            }

            if (stop_program.load() && channel.processing_done && channel.acquisition_done && channel.result_buffer_csv.empty())
                break;
        }

        fclose(output_file);
        std::cout << "Logging inference results on CSV thread on " << channel.name << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in log_results_csv: " << e.what() << std::endl;
    }
}

template void log_results_csv<input_type1, output_type1>(Channel<input_type1, output_type1> &, const std::string &);
template void log_results_csv<input_type2, output_type2>(Channel<input_type2, output_type2> &, const std::string &);
