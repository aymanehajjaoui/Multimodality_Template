/*Common_CH2.hpp*/

#pragma once

#include "../Common/Common.hpp"
#include "../model1/include/model.h"

#define DECIMATION_CH2 (125000 / MODEL_INPUT_DIM_0)

struct data_part_CH2_t
{
    input_t data;
};

struct model_result_CH2_t
{
    output_t output;
    double computation_time;
};

struct shared_counters_CH2_t : public shared_counters_base_t {};

struct Channel_CH2
{
    std::queue<std::shared_ptr<data_part_CH2_t>> data_queue_csv;
    std::queue<std::shared_ptr<data_part_CH2_t>> data_queue_dac;
    std::queue<std::shared_ptr<data_part_CH2_t>> model_queue;
    std::deque<model_result_CH2_t> result_buffer_csv;
    std::deque<model_result_CH2_t> result_buffer_dac;
    std::deque<model_result_CH2_t> result_buffer_tcp;

    sem_t data_sem_csv;
    sem_t data_sem_dac;
    sem_t model_sem;
    sem_t result_sem_csv;
    sem_t result_sem_dac;
    sem_t result_sem_tcp;

    rp_acq_trig_state_t state;

    std::chrono::steady_clock::time_point trigger_time_point;
    std::chrono::steady_clock::time_point end_time_point;

    bool acquisition_done = false;
    bool processing_done = false;
    bool channel_triggered = false;

    shared_counters_CH2_t *counters = nullptr;
    std::atomic<uint64_t> trigger_time_ns{0};
    std::atomic<uint64_t> end_time_ns{0};

    rp_channel_t channel_id;

    std::string ip;
};

extern Channel_CH2 channel2;

template <typename T>
inline void convert_raw_data_CH2(const int16_t *src, T dst[MODEL_INPUT_DIM_0][1], size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        if constexpr (std::is_same<T, float>::value)
        {
            dst[i][0] = static_cast<float>(src[i]) / 8192.0f;
        }
        else if constexpr (std::is_same<T, int8_t>::value)
        {
            dst[i][0] = static_cast<int8_t>(std::round(src[i] / 64.0f));
        }
        else if constexpr (std::is_same<T, int16_t>::value)
        {
            dst[i][0] = src[i];
        }
        else
        {
            static_assert(!sizeof(T *), "Unsupported data type in convert_raw_data.");
        }
    }
}

