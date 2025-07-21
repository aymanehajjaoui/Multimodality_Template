/*DataWriterTCP.hpp*/

#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include "Channel1.hpp"
#include "Channel2.hpp"
#include "TCP.hpp"

template <typename In, typename Out>
void write_data_tcp(Channel<In, Out> &channel, int port)
{
    try
    {
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0)
        {
            std::cerr << "Failed to create socket on port " << port << std::endl;
            return;
        }

        sockaddr_in serv_addr{};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr.s_addr = INADDR_ANY;
        serv_addr.sin_port = htons(port);

        int opt = 1;
        setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0)
        {
            std::cerr << "Failed to bind socket on port " << port << ": " << strerror(errno) << std::endl;
            close(sockfd);
            return;
        }

        if (listen(sockfd, 1) < 0)
        {
            std::cerr << "Failed to listen on port " << port << ": " << strerror(errno) << std::endl;
            close(sockfd);
            return;
        }

        int flags = fcntl(sockfd, F_GETFL, 0);
        fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

        std::cout << "Waiting for client on port " << port << "..." << std::endl;

        int client_sock = -1;
        while (!stop_program.load())
        {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(sockfd, &readfds);
            struct timeval timeout{1, 0};

            int ready = select(sockfd + 1, &readfds, nullptr, nullptr, &timeout);
            if (ready > 0 && FD_ISSET(sockfd, &readfds))
            {
                client_sock = accept(sockfd, nullptr, nullptr);
                if (client_sock >= 0)
                {
                    std::cout << "Client connected on port " << port << std::endl;
                    break;
                }
                else if (errno != EWOULDBLOCK && errno != EAGAIN)
                {
                    std::cerr << "Accept failed on port " << port << ": " << strerror(errno) << std::endl;
                    close(sockfd);
                    return;
                }
            }
        }

        if (client_sock < 0)
        {
            std::cout << "No client connected on port " << port << ". Exiting TCP thread." << std::endl;
            close(sockfd);
            return;
        }

        while (true)
        {
            if (stop_program.load() && channel.data_queue_tcp.empty())
                break;

            if (sem_wait(&channel.data_sem_tcp) != 0)
            {
                if (errno == EINTR)
                {
                    if (stop_program.load() && channel.processing_done && channel.acquisition_done && channel.data_queue_tcp.empty())
                        break;
                    continue;
                }
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
                    float_buffer[i] = static_cast<float>(part->converted_data[i][0]);

                ssize_t bytes_sent = send(client_sock, float_buffer.data(), count * sizeof(float), 0);
                if (bytes_sent != static_cast<ssize_t>(count * sizeof(float)))
                    std::cerr << "Failed to send TCP data on " << channel.name << std::endl;

                channel.write_count_tcp.fetch_add(1, std::memory_order_relaxed);
            }
        }

        close(client_sock);
        close(sockfd);
        std::cout << "TCP writing thread on " << channel.name << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in write_data_tcp: " << e.what() << std::endl;
    }
}