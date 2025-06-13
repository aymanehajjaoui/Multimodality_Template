/*ModelWriterTCP_CH1.cpp*/

#include "CH1/ModelWriterTCP_CH1.hpp"
#include <iostream>
#include <cmath>

void log_results_tcp(Channel_CH1 &channel, int port)
{
    char ip_buffer[128] = {0};
    if (get_gateway_ip(ip_buffer, sizeof(ip_buffer)) != 0)
    {
        std::cerr << "[TCP] Failed to detect gateway IP\n";
        return;
    }

    int socket_fd = initialize_tcp(ip_buffer, port);
    if (socket_fd < 0)
    {
        std::cerr << "[TCP] Could not establish connection. Exiting TCP thread for CH" << (int)channel.channel_id + 1 << "\n";
        return;
    }

    while (true)
    {
        if (sem_wait(&channel.result_sem_tcp) != 0)
        {
            if (errno == EINTR && stop_program.load())
                break;
            else
                perror("[TCP] sem_wait failed");
            continue;
        }

        if (stop_program.load() && channel.result_buffer_tcp.empty())
            break;

        while (!channel.result_buffer_tcp.empty())
        {
            const model_result_CH1_t &result = channel.result_buffer_tcp.front();
            float value = static_cast<float>(result.output[0]);

            if (!send_tcp_data(socket_fd, value))
            {
                std::cerr << "[TCP] Send failed, aborting thread for CH" << (int)channel.channel_id + 1 << "\n";
                close(socket_fd);
                return;
            }

            channel.result_buffer_tcp.pop_front();
            channel.counters->log_count_tcp.fetch_add(1, std::memory_order_relaxed);
        }

        if (channel.processing_done && channel.result_buffer_tcp.empty())
            break;
    }

    close(socket_fd);
    std::cout << "[TCP] Logging thread exited for channel " << (int)channel.channel_id + 1 << std::endl;
}
