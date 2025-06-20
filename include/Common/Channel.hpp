/*Channel.hpp*/

#pragma once

#include <queue>
#include <deque>
#include <memory>
#include <string>
#include <semaphore.h>
#include <chrono>
#include <atomic>
#include <cmath>
#include <type_traits>

#include "rp.h"

template <typename InputType, typename OutputType>
class Channel
{
public:
    static_assert(std::is_array<InputType>::value, "InputType must be an array type.");
    static constexpr size_t InputLength = std::extent<InputType, 0>::value;
    static_assert(InputLength > 0, "InputType must have a non-zero extent.");

    static constexpr uint32_t Decimation = static_cast<uint32_t>(125000.0 / static_cast<double>(InputLength));

    struct DataPart
    {
        InputType data;
    };

    struct Result
    {
        OutputType output;
        double computation_time;
    };

    std::string name;
    rp_channel_t rp_channel;

    std::queue<std::shared_ptr<DataPart>> data_queue_csv, data_queue_dac, data_queue_tcp, model_queue;
    std::deque<Result> result_buffer_csv, result_buffer_dac, result_buffer_tcp;

    sem_t data_sem_csv, data_sem_dac, data_sem_tcp, model_sem, result_sem_csv, result_sem_dac, result_sem_tcp;
    rp_acq_trig_state_t state;
    std::chrono::steady_clock::time_point trigger_time, end_time;

    std::atomic<bool> channel_triggered{false}, acquisition_done{false}, processing_done{false};
    std::atomic<uint64_t> trigger_time_ns{0}, end_time_ns{0};

    std::string ip;
    std::atomic<int> acquire_count, model_count, write_count_csv, write_count_dac, write_count_tcp;
    std::atomic<int> log_count_csv, log_count_dac, log_count_tcp;

    Channel()
    {
        sem_init(&data_sem_csv, 0, 0);
        sem_init(&data_sem_dac, 0, 0);
        sem_init(&data_sem_tcp, 0, 0);
        sem_init(&model_sem, 0, 0);
        sem_init(&result_sem_csv, 0, 0);
        sem_init(&result_sem_dac, 0, 0);
        sem_init(&result_sem_tcp, 0, 0);
        reset_counters();

        rp_channel = (name == "Channel1") ? RP_CH_1 : RP_CH_2;
    }

    ~Channel()
    {
        sem_destroy(&data_sem_csv);
        sem_destroy(&data_sem_dac);
        sem_destroy(&data_sem_tcp);
        sem_destroy(&model_sem);
        sem_destroy(&result_sem_csv);
        sem_destroy(&result_sem_dac);
        sem_destroy(&result_sem_tcp);
    }

    static constexpr uint32_t get_decimation()
    {
        return Decimation;
    }
    
    virtual void runmodel(const InputType input, OutputType output) = 0;

    void reset_counters()
    {
        acquire_count = 0;
        model_count = 0;
        write_count_csv = 0;
        write_count_dac = 0;
        write_count_tcp = 0;
        log_count_csv = 0;
        log_count_dac = 0;
        log_count_tcp = 0;
    }

    template <typename T, size_t N>
    inline void convert_raw_data(const int16_t *src, T (&dst)[N][1], size_t count)
    {
        for (size_t i = 0; i < count && i < N; ++i)
        {
            if constexpr (std::is_same<T, float>::value)
                dst[i][0] = static_cast<float>(src[i]) / 8192.0f;
            else if constexpr (std::is_same<T, int8_t>::value)
                dst[i][0] = static_cast<int8_t>(std::round(src[i] / 64.0f));
            else if constexpr (std::is_same<T, int16_t>::value)
                dst[i][0] = src[i];
            else
                static_assert(!sizeof(T *), "Unsupported data type in convert_raw_data.");
        }
    }
};
