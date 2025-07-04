/*DataWriterDAC.hpp*/

#pragma once

#include "DAC.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"
#include <iostream>
#include <algorithm>

extern std::atomic<bool> stop_program;

template <typename In, typename Out>
void write_data_dac(Channel<In, Out> &channel)
{
    try
    {
        while (true)
        {
            if (sem_wait(&channel.data_sem_dac) != 0)
            {
                if (errno == EINTR)
                {
                    if (stop_program.load() && channel.acquisition_done && channel.data_queue_dac.empty())
                        break;
                    continue;
                }
            }

            while (!channel.data_queue_dac.empty())
            {
                auto part = channel.data_queue_dac.front();
                channel.data_queue_dac.pop();

                for (size_t k = 0; k < channel.InputLength; ++k)
                {
                    float voltage = OutputToVoltage(part->data[k][0]);
                    voltage = std::clamp(voltage, -1.0f, 1.0f);
                    rp_GenAmp(channel.rp_channel, voltage);
                }

                channel.write_count_dac.fetch_add(1, std::memory_order_relaxed);
            }

            if (stop_program.load() && channel.acquisition_done && channel.data_queue_dac.empty())
                break;
        }

        std::cout << "Data writing on DAC thread on " << channel.name << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in write_data_dac: " << e.what() << std::endl;
    }
}