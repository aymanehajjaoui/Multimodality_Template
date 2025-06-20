/* ModelProcessing.cpp */

#include "ModelProcessing.hpp"
#include <iostream>
#include <chrono>
#include <type_traits>
#include <algorithm>

extern bool save_output_csv;
extern bool save_output_dac;
extern bool save_output_tcp;

template <typename T, size_t N>
void sample_norm(T (&data)[N][1])
{
    using base_t = std::remove_cv_t<std::remove_reference_t<decltype(data[0][0])>>;

    base_t min_val = data[0][0];
    base_t max_val = data[0][0];

    for (size_t i = 1; i < N; ++i)
    {
        if (data[i][0] < min_val)
            min_val = data[i][0];
        if (data[i][0] > max_val)
            max_val = data[i][0];
    }

    base_t range = max_val - min_val;
    if (range == 0)
        range = 1;

    for (size_t i = 0; i < N; ++i)
    {
        if constexpr (std::is_floating_point<base_t>::value)
            data[i][0] = (data[i][0] - min_val) / static_cast<base_t>(range);
        else
            data[i][0] = static_cast<base_t>(((data[i][0] - min_val) * 512) / range);
    }
}

template <typename In, typename Out>
void model_inference(Channel<In, Out> &channel)
{
    try
    {
        std::cout << "Starting Inference on " << channel.name << "..." << std::endl;

        while (true)
        {
            if (sem_wait(&channel.model_sem) != 0)
            {
                if (errno == EINTR && stop_program.load())
                    break;
                continue;
            }

            if (stop_program.load() && channel.acquisition_done && channel.model_queue.empty())
                break;

            while (!channel.model_queue.empty())
            {
                std::shared_ptr<typename Channel<In, Out>::DataPart> part = channel.model_queue.front();
                channel.model_queue.pop();

                typename Channel<In, Out>::Result result;
                auto start = std::chrono::high_resolution_clock::now();
                channel.runmodel(part->data, result.output);
                auto end = std::chrono::high_resolution_clock::now();

                result.computation_time = std::chrono::duration<double, std::milli>(end - start).count();

                if (save_output_csv)
                {
                    channel.result_buffer_csv.push_back(result);
                    sem_post(&channel.result_sem_csv);
                }

                if (save_output_dac)
                {
                    channel.result_buffer_dac.push_back(result);
                    sem_post(&channel.result_sem_dac);
                }

                if (save_output_tcp)
                {
                    channel.result_buffer_tcp.push_back(result);
                    sem_post(&channel.result_sem_tcp);
                }

                channel.model_count.fetch_add(1, std::memory_order_relaxed);
            }

            if (stop_program.load() && channel.acquisition_done && channel.model_queue.empty())
                break;
        }

        channel.processing_done = true;
        if (save_output_csv)
            sem_post(&channel.result_sem_csv);
        if (save_output_dac)
            sem_post(&channel.result_sem_dac);
        if (save_output_tcp)
            sem_post(&channel.result_sem_tcp);

        std::cout << "Model inference thread on " << channel.name << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in model_inference: " << e.what() << std::endl;
    }
}

template <typename In, typename Out>
void model_inference_mod(Channel<In, Out> &channel)
{
    try
    {
        while (true)
        {
            if (sem_wait(&channel.model_sem) != 0)
            {
                if (errno == EINTR && stop_program.load())
                    break;
                continue;
            }

            if (stop_program.load() && channel.acquisition_done && channel.model_queue.empty())
                break;

            while (!channel.model_queue.empty())
            {
                std::shared_ptr<typename Channel<In, Out>::DataPart> part = channel.model_queue.front();
                channel.model_queue.pop();

                sample_norm(part->data);

                typename Channel<In, Out>::Result result;
                auto start = std::chrono::high_resolution_clock::now();
                channel.runmodel(part->data, result.output);
                auto end = std::chrono::high_resolution_clock::now();

                result.computation_time = std::chrono::duration<double, std::milli>(end - start).count();

                if (save_output_csv)
                {
                    channel.result_buffer_csv.push_back(result);
                    sem_post(&channel.result_sem_csv);
                }

                if (save_output_dac)
                {
                    channel.result_buffer_dac.push_back(result);
                    sem_post(&channel.result_sem_dac);
                }

                if (save_output_tcp)
                {
                    channel.result_buffer_tcp.push_back(result);
                    sem_post(&channel.result_sem_tcp);
                }

                channel.model_count.fetch_add(1, std::memory_order_relaxed);
            }

            if (stop_program.load() && channel.acquisition_done && channel.model_queue.empty())
                break;
        }

        channel.processing_done = true;
        if (save_output_csv)
            sem_post(&channel.result_sem_csv);
        if (save_output_dac)
            sem_post(&channel.result_sem_dac);
        if (save_output_tcp)
            sem_post(&channel.result_sem_tcp);

        std::cout << "Model inference mod thread on " << channel.name << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in model_inference_mod: " << e.what() << std::endl;
    }
}

template void model_inference<input_type1, output_type1>(Channel<input_type1, output_type1> &);
template void model_inference<input_type2, output_type2>(Channel<input_type2, output_type2> &);
template void model_inference_mod<input_type1, output_type1>(Channel<input_type1, output_type1> &);
template void model_inference_mod<input_type2, output_type2>(Channel<input_type2, output_type2> &);
