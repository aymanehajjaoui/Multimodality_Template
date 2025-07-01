/*DataWriterTCP.hpp*/

#pragma once

#include "TCP.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"
#include <iostream>
#include <type_traits>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

template <typename In, typename Out>
void write_data_tcp(Channel<In, Out> &channel, int port)
{
    try
    {
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0)
        {
            std::cerr << "Failed to create socket for " << channel.name << std::endl;
            return;
        }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in server_addr = {};
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(port);

        if (bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
        {
            std::cerr << "Failed to bind TCP socket on port " << port << std::endl;
            close(server_fd);
            return;
        }

        listen(server_fd, 1);

        std::cout << "Waiting for client on port " << port << "..." << std::endl;

        int client_fd = -1;
        while (!stop_program.load())
        {
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(server_fd, &fds);

            timeval timeout = {1, 0};
            int activity = select(server_fd + 1, &fds, nullptr, nullptr, &timeout);

            if (activity > 0 && FD_ISSET(server_fd, &fds))
            {
                client_fd = accept(server_fd, nullptr, nullptr);
                break;
            }
        }

        if (client_fd < 0)
        {
            std::cerr << "No client connected on port " << port << ". Exiting TCP thread." << std::endl;
            close(server_fd);
            return;
        }

        std::cout << "Client connected on port " << port << std::endl;

        while (true)
        {
            if (stop_program.load())
                sem_post(&channel.data_sem_tcp); // Force wake-up

            if (sem_wait(&channel.data_sem_tcp) != 0)
            {
                if (errno == EINTR && stop_program.load())
                    break;
                continue;
            }

            if (stop_program.load() && channel.data_queue_tcp.empty())
                break;

            while (!channel.data_queue_tcp.empty())
            {
                std::shared_ptr<typename Channel<In, Out>::DataPart> part = channel.data_queue_tcp.front();
                channel.data_queue_tcp.pop();

                constexpr size_t count = Channel<In, Out>::InputLength;
                std::vector<float> float_buffer(count);

                for (size_t i = 0; i < count; ++i)
                    float_buffer[i] = static_cast<float>(part->data[i][0]);

                ssize_t bytes_sent = send(client_fd, float_buffer.data(), count * sizeof(float), 0);
                if (bytes_sent != static_cast<ssize_t>(count * sizeof(float)))
                    std::cerr << "Failed to send TCP data on " << channel.name << std::endl;

                channel.write_count_tcp.fetch_add(1, std::memory_order_relaxed);
            }

            if (stop_program.load() && channel.acquisition_done && channel.data_queue_tcp.empty())
                break;
        }

        close(client_fd);
        close(server_fd);
        std::cout << "TCP writing thread on " << channel.name << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in write_data_tcp: " << e.what() << std::endl;
    }
}