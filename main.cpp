/* main.cpp */

#include <iostream>
#include <thread>
#include <csignal>
#include "Common.hpp"
#include "SystemUtils.hpp"
#include "DataAcquisition.hpp"
#include "ModelProcessing.hpp"
#include "ModelWriterCSV.hpp"
#include "ModelWriterDAC.hpp"
#include "ModelWriterTCP.hpp"
#include "DataWriterCSV.hpp"
#include "DataWriterDAC.hpp"
#include "DataWriterTCP.hpp"
#include "TCP.hpp"
#include "DAC.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"
#include <barrier>

std::barrier sync_start(2);

Channel1 channel1;
Channel2 channel2;

std::atomic<bool> stop_program(false);
std::atomic<bool> stop_acquisition(false);

bool save_data_csv = false;
bool save_data_dac = false;
bool save_data_tcp = false;
bool save_output_csv = false;
bool save_output_dac = false;
bool save_output_tcp = false;

template <typename In, typename Out>
void run_channel(Channel<In, Out> &channel, rp_channel_t rp_ch, int core_id,
                 const std::string &data_csv, const std::string &output_csv,
                 int tcp_data_port, int tcp_result_port)
{
    set_current_thread_affinity(core_id);
    sync_start.arrive_and_wait();

    std::thread acq_thread([&]()
                           { acquire_data<In, Out>(channel); });

    //set_thread_priority(acq_thread,acq_priority);

    std::thread model_thread([&]()
                             { model_inference<In, Out>(channel); });

    set_thread_priority(model_thread,model_priority);

    std::thread write_thread_csv, write_thread_dac, write_thread_tcp;
    std::thread log_thread_csv, log_thread_dac, log_thread_tcp;

    if (save_data_csv)
        write_thread_csv = std::thread([&]()
                                       { write_data_csv<In, Out>(channel, data_csv); });

    if (save_data_dac)
        write_thread_dac = std::thread([&]()
                                       { write_data_dac<In, Out>(channel); });

    if (save_data_tcp)
        write_thread_tcp = std::thread([&]()
                                       { write_data_tcp<In, Out>(channel, tcp_data_port); });

    if (save_output_csv)
        log_thread_csv = std::thread([&]()
                                     { log_results_csv<In, Out>(channel, output_csv); });

    if (save_output_dac)
        log_thread_dac = std::thread([&]()
                                     { log_results_dac<In, Out>(channel, rp_ch); });

    if (save_output_tcp)
        log_thread_tcp = std::thread([&]()
                                     { log_results_tcp<In, Out>(channel, tcp_result_port); });

    if (acq_thread.joinable())
        acq_thread.join();
    if (model_thread.joinable())
        model_thread.join();

    if (save_data_csv && write_thread_csv.joinable())
        write_thread_csv.join();

    if (save_data_dac && write_thread_dac.joinable())
        write_thread_dac.join();

    if (save_data_tcp && write_thread_tcp.joinable())
        write_thread_tcp.join();

    if (save_output_csv && log_thread_csv.joinable())
        log_thread_csv.join();

    if (save_output_dac && log_thread_dac.joinable())
        log_thread_dac.join();

    if (save_output_tcp && log_thread_tcp.joinable())
        log_thread_tcp.join();
}

int main()
{
    Channel1 channel1;
    Channel2 channel2;
    if (rp_Init() != RP_OK)
    {
        std::cerr << "Rp API init failed!" << std::endl;
        return -1;
    }

    initialize_acq();
    if (save_data_dac || save_output_dac)
    {
        initialize_DAC();
    }
    folder_manager("DataOutput");
    folder_manager("ModelOutput");
    ask_user_preferences(save_data_csv, save_data_dac, save_data_tcp, save_output_csv, save_output_dac, save_output_tcp);

    g_channel1_ptr = &channel1;
    g_channel2_ptr = &channel2;
    std::signal(SIGINT, signal_handler);

    std::thread thread1([&]()
                        { run_channel<input_type1, output_type1>(
                              channel1, RP_CH_1, 0,
                              "DataOutput/data_ch1.csv",
                              "ModelOutput/output_ch1.csv",
                              4000, 5000); });

    std::thread thread2([&]()
                        { run_channel<input_type2, output_type2>(
                              channel2, RP_CH_2, 1,
                              "DataOutput/data_ch2.csv",
                              "ModelOutput/output_ch2.csv",
                              4001, 5001); });

    thread1.join();
    thread2.join();

    print_channel_stats(channel1);
    print_channel_stats(channel2);

    return 0;
}