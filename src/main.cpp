#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <csignal>
#include <sched.h>
#include <sys/statvfs.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <iomanip>

#include "rp.h"
#include "Common/SystemUtils.hpp"
#include "Common/ADC.hpp"
#include "Common/DAC.hpp"
#include "Common/TCP.hpp"

#include "CH1/Common_CH1.hpp"
#include "CH1/DataAcquisition_CH1.hpp"
#include "CH1/DataWriterCSV_CH1.hpp"
#include "CH1/DataWriterDAC_CH1.hpp"
#include "CH1/ModelProcessing_CH1.hpp"
#include "CH1/ModelWriterCSV_CH1.hpp"
#include "CH1/ModelWriterDAC_CH1.hpp"
#include "CH1/ModelWriterTCP_CH1.hpp"

#include "CH2/Common_CH2.hpp"
#include "CH2/DataAcquisition_CH2.hpp"
#include "CH2/DataWriterCSV_CH2.hpp"
#include "CH2/DataWriterDAC_CH2.hpp"
#include "CH2/ModelProcessing_CH2.hpp"
#include "CH2/ModelWriterCSV_CH2.hpp"
#include "CH2/ModelWriterDAC_CH2.hpp"
#include "CH2/ModelWriterTCP_CH2.hpp"

int main()
{
    if (rp_Init() != RP_OK)
    {
        std::cerr << "Rp API init failed!" << std::endl;
        return -1;
    }

    for (auto *sem : {
             &channel1.data_sem_csv, &channel1.data_sem_dac, &channel1.model_sem,
             &channel1.result_sem_csv, &channel1.result_sem_dac, &channel1.result_sem_tcp,
             &channel2.data_sem_csv, &channel2.data_sem_dac, &channel2.model_sem,
             &channel2.result_sem_csv, &channel2.result_sem_dac, &channel2.result_sem_tcp})
        sem_init(sem, 0, 0);

    std::signal(SIGINT, signal_handler);
    folder_manager("DataOutput");
    folder_manager("ModelOutput");

    int shm_fd = shm_open(SHM_COUNTERS, O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1 || ftruncate(shm_fd, sizeof(shared_counters_CH1_t) + sizeof(shared_counters_CH2_t)) == -1)
        return -1;

    void *mapped = mmap(0, sizeof(shared_counters_CH1_t) + sizeof(shared_counters_CH2_t),
                        PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (mapped == MAP_FAILED)
        return -1;

    auto *c1 = reinterpret_cast<shared_counters_CH1_t *>(mapped);
    auto *c2 = reinterpret_cast<shared_counters_CH2_t *>((char *)mapped + sizeof(shared_counters_CH1_t));
    channel1.counters = c1;
    channel2.counters = c2;

    new (&c1->ready_barrier) std::atomic<int>(0);
    new (&c2->ready_barrier) std::atomic<int>(0);

    if (!ask_user_preferences(save_data_csv, save_data_dac, save_output_csv, save_output_dac, save_output_tcp))
        return -1;

    ::save_data_csv = save_data_csv;
    ::save_data_dac = save_data_dac;
    ::save_output_csv = save_output_csv;
    ::save_output_dac = save_output_dac;
    ::save_output_tcp = save_output_tcp;

    initialize_acq();
    if (save_data_dac || save_output_dac)
        initialize_DAC();

    if ((pid1 = fork()) == 0)
    {
        set_process_affinity(0);

        std::thread acq_thread(
            static_cast<void (*)(Channel_CH1 &, rp_channel_t)>(acquire_data),
            std::ref(channel1), RP_CH_1);

        std::thread model_thread(
            static_cast<void (*)(Channel_CH1 &)>(model_inference),
            std::ref(channel1));

        std::thread w_csv, w_dac, l_csv, l_dac, l_tcp;
        if (save_data_csv)
            w_csv = std::thread(
                static_cast<void (*)(Channel_CH1 &, const std::string &)>(write_data_csv),
                std::ref(channel1), "DataOutput/data_ch1.csv");

        if (save_data_dac)
            w_dac = std::thread(
                static_cast<void (*)(Channel_CH1 &, rp_channel_t)>(write_data_dac),
                std::ref(channel1), RP_CH_1);

        if (save_output_csv)
            l_csv = std::thread(
                static_cast<void (*)(Channel_CH1 &, const std::string &)>(log_results_csv),
                std::ref(channel1), "ModelOutput/output_ch1.csv");

        if (save_output_dac)
            l_dac = std::thread(
                static_cast<void (*)(Channel_CH1 &, rp_channel_t)>(log_results_dac),
                std::ref(channel1), RP_CH_1);

        if (save_output_tcp)
            l_tcp = std::thread(
                static_cast<void (*)(Channel_CH1 &, int)>(log_results_tcp),
                std::ref(channel1), 5000);

        set_thread_priority(model_thread, model_priority);
        acq_thread.join();
        model_thread.join();
        if (save_data_csv)
            w_csv.join();
        if (save_data_dac)
            w_dac.join();
        if (save_output_csv)
            l_csv.join();
        if (save_output_dac)
            l_dac.join();
        if (save_output_tcp)
            l_tcp.join();
        exit(0);
    }

    if ((pid2 = fork()) == 0)
    {
        set_process_affinity(1);

        std::thread acq_thread(
            static_cast<void (*)(Channel_CH2 &, rp_channel_t)>(acquire_data),
            std::ref(channel2), RP_CH_2);

        std::thread model_thread(
            static_cast<void (*)(Channel_CH2 &)>(model_inference),
            std::ref(channel2));

        std::thread w_csv, w_dac, l_csv, l_dac, l_tcp;
        if (save_data_csv)
            w_csv = std::thread(
                static_cast<void (*)(Channel_CH2 &, const std::string &)>(write_data_csv),
                std::ref(channel2), "DataOutput/data_ch2.csv");

        if (save_data_dac)
            w_dac = std::thread(
                static_cast<void (*)(Channel_CH2 &, rp_channel_t)>(write_data_dac),
                std::ref(channel2), RP_CH_2);

        if (save_output_csv)
            l_csv = std::thread(
                static_cast<void (*)(Channel_CH2 &, const std::string &)>(log_results_csv),
                std::ref(channel2), "ModelOutput/output_ch2.csv");

        if (save_output_dac)
            l_dac = std::thread(
                static_cast<void (*)(Channel_CH2 &, rp_channel_t)>(log_results_dac),
                std::ref(channel2), RP_CH_2);

        if (save_output_tcp)
            l_tcp = std::thread(
                static_cast<void (*)(Channel_CH2 &, int)>(log_results_tcp),
                std::ref(channel2), 5001);

        set_thread_priority(model_thread, model_priority);
        acq_thread.join();
        model_thread.join();
        if (save_data_csv)
            w_csv.join();
        if (save_data_dac)
            w_dac.join();
        if (save_output_csv)
            l_csv.join();
        if (save_output_dac)
            l_dac.join();
        if (save_output_tcp)
            l_tcp.join();
        exit(0);
    }

    waitpid(pid1, nullptr, 0);
    waitpid(pid2, nullptr, 0);

    cleanup();
    print_channel_stats(channel1.counters, "CH1");
    print_channel_stats(channel2.counters, "CH2");
    shm_unlink(SHM_COUNTERS);

    for (auto *sem : {
             &channel1.data_sem_csv, &channel1.data_sem_dac, &channel1.model_sem,
             &channel1.result_sem_csv, &channel1.result_sem_dac, &channel1.result_sem_tcp,
             &channel2.data_sem_csv, &channel2.data_sem_dac, &channel2.model_sem,
             &channel2.result_sem_csv, &channel2.result_sem_dac, &channel2.result_sem_tcp})
        sem_destroy(sem);

    return 0;
}
