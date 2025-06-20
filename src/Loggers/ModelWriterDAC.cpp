/* ModelWriterDAC.cpp */

#include "ModelWriterDAC.hpp"
#include <iostream>
#include <unistd.h>

template <typename In, typename Out>
void log_results_dac(Channel<In, Out> &channel, rp_channel_t rp_ch)
{
    try
    {
        int output_index = 1;
        while (true)
        {
            if (sem_wait(&channel.result_sem_dac) != 0)
            {
                if (errno == EINTR && stop_program.load())
                    break;
                continue;
            }

            if (stop_program.load() && channel.result_buffer_dac.empty())
                break;

            while (!channel.result_buffer_dac.empty())
            {
                const auto &result = channel.result_buffer_dac.front();

                float voltage = OutputToVoltage(result.output[0]);
                voltage = std::clamp(voltage, -1.0f, 1.0f);
                if (rp_GenAmp(rp_ch, voltage) != RP_OK)
                {
                    std::cerr << "DAC write failed on " << channel.name << std::endl;
                }

                channel.result_buffer_dac.pop_front();
                channel.log_count_dac.fetch_add(1, std::memory_order_relaxed);
                ++output_index;
            }

            if (stop_program.load() && channel.processing_done && channel.acquisition_done && channel.result_buffer_dac.empty())
                break;
        }

        std::cout << "Logging DAC results on " << channel.name << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in log_results_dac: " << e.what() << std::endl;
    }
}

template void log_results_dac<input_type1, output_type1>(Channel<input_type1, output_type1> &, rp_channel_t);
template void log_results_dac<input_type2, output_type2>(Channel<input_type2, output_type2> &, rp_channel_t);
