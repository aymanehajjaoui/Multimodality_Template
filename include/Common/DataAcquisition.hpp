/*DataAcquisition.hpp*/

#pragma once

#include "ADC.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"
#include "SystemUtils.hpp"
#include <iostream>

template <typename In, typename Out>
void acquire_data(Channel<In, Out>& channel)
{
    try
    {
        std::cout << "Waiting for trigger on " << channel.name << "..." << std::endl;

        while (!channel.channel_triggered && !stop_acquisition.load())
        {
            if (rp_AcqGetTriggerStateCh(channel.rp_channel, &channel.state) != RP_OK)
            {
                std::cerr << "rp_AcqGetTriggerStateCh failed on " << channel.name << std::endl;
                exit(-1);
            }

            if (channel.state == RP_TRIG_STATE_TRIGGERED)
            {
                channel.channel_triggered = true;
                std::cout << "Trigger detected on " << channel.name << "!" << std::endl;
                channel.trigger_time = std::chrono::steady_clock::now();
                channel.trigger_time_ns.store(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        channel.trigger_time.time_since_epoch()).count());
            }
        }

        if (!channel.channel_triggered)
        {
            std::cerr << "INFO: Acquisition stopped before trigger detected on " << channel.name << "." << std::endl;
            stop_acquisition.store(true);
            return;
        }

        std::cout << "Starting data acquisition on " << channel.name << std::endl;

        uint32_t pw = 0;
        constexpr uint32_t samples_per_chunk = Channel<In, Out>::InputLength;
        uint32_t chunk_size = samples_per_chunk;

        if (rp_AcqAxiGetWritePointerAtTrig(channel.rp_channel, &pw) != RP_OK)
        {
            std::cerr << "Error getting write pointer at trigger for channel " << channel.name << std::endl;
            exit(-1);
        }

        uint32_t pos = pw;

        while (!stop_acquisition.load())
        {
            if (is_disk_space_below_threshold("/", DISK_SPACE_THRESHOLD))
            {
                std::cerr << "ERR: Disk space below threshold. Stopping acquisition." << std::endl;
                stop_acquisition.store(true);
                break;
            }

            uint32_t pwrite = 0;
            if (rp_AcqAxiGetWritePointer(channel.rp_channel, &pwrite) == RP_OK)
            {
                int64_t distance = (pwrite >= pos) ? (pwrite - pos) : (DATA_SIZE - pos + pwrite);

                if (distance < 0)
                {
                    std::cerr << "ERR: Negative distance calculated on " << channel.name << std::endl;
                    continue;
                }

                if (distance >= DATA_SIZE)
                {
                    std::cerr << "ERR: Overrun detected on " << channel.name << std::endl;
                    stop_acquisition.store(true);
                    return;
                }

                if (distance >= samples_per_chunk)
                {
                    int16_t buffer_raw[samples_per_chunk];
                    if (rp_AcqAxiGetDataRaw(channel.rp_channel, pos, &chunk_size, buffer_raw) != RP_OK)
                    {
                        std::cerr << "rp_AcqAxiGetDataRaw failed on " << channel.name << std::endl;
                        continue;
                    }

                    auto part = std::make_shared<typename Channel<In, Out>::DataPart>();
                    channel.convert_raw_data(buffer_raw, part->data, samples_per_chunk);

                    pos += samples_per_chunk;
                    if (pos >= DATA_SIZE)
                        pos -= DATA_SIZE;

                    if (save_data_csv)
                    {
                        channel.data_queue_csv.push(part);
                        sem_post(&channel.data_sem_csv);
                    }

                    if (save_data_dac)
                    {
                        channel.data_queue_dac.push(part);
                        sem_post(&channel.data_sem_dac);
                    }

                    if (save_data_tcp)
                    {
                        channel.data_queue_tcp.push(part);
                        sem_post(&channel.data_sem_tcp);
                    }

                    channel.model_queue.push(part);
                    sem_post(&channel.model_sem);

                    channel.acquire_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }

        channel.end_time = std::chrono::steady_clock::now();
        channel.end_time_ns.store(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                channel.end_time.time_since_epoch()).count());

        channel.acquisition_done = true;

        if (save_data_csv)
            sem_post(&channel.data_sem_csv);
        if (save_data_dac)
            sem_post(&channel.data_sem_dac);

        sem_post(&channel.model_sem);

        std::cout << "Acquisition thread on " << channel.name << " exiting..." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception in acquire_data for channel: " << e.what() << std::endl;
    }
}
