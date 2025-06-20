/* ModelWriterTCP.cpp */

#include "ModelWriterTCP.hpp"
#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>
#include <errno.h>
#include <chrono>
#include <thread>

template <typename In, typename Out>
void log_results_tcp(Channel<In, Out> &channel, int port)
{
    try {
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            std::cerr << "Failed to create socket on port " << port << std::endl;
            return;
        }

        sockaddr_in serv_addr{};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_addr.s_addr = INADDR_ANY;
        serv_addr.sin_port = htons(port);

        int opt = 1;
        setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        if (bind(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
            std::cerr << "Failed to bind socket to port " << port << ": " << strerror(errno) << std::endl;
            close(sockfd);
            return;
        }

        if (listen(sockfd, 1) < 0) {
            std::cerr << "Failed to listen on port " << port << ": " << strerror(errno) << std::endl;
            close(sockfd);
            return;
        }

        int flags = fcntl(sockfd, F_GETFL, 0);
        fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

        std::cout << "Waiting for client on port " << port << "..." << std::endl;

        int client_sock = -1;
        while (!stop_program.load()) {
            fd_set readfds;
            FD_ZERO(&readfds);
            FD_SET(sockfd, &readfds);
            struct timeval timeout{1, 0};

            int ready = select(sockfd + 1, &readfds, nullptr, nullptr, &timeout);
            if (ready > 0 && FD_ISSET(sockfd, &readfds)) {
                client_sock = accept(sockfd, nullptr, nullptr);
                if (client_sock >= 0) {
                    std::cout << "Client connected on port " << port << std::endl;
                    break;
                } else if (errno != EWOULDBLOCK && errno != EAGAIN) {
                    std::cerr << "Accept failed on port " << port << ": " << strerror(errno) << std::endl;
                    close(sockfd);
                    return;
                }
            }
        }

        if (client_sock < 0) {
            std::cout << "No client connected on port " << port << ". Exiting TCP thread." << std::endl;
            close(sockfd);
            return;
        }

        int output_index = 1;
        while (true) {
            if (sem_wait(&channel.result_sem_tcp) != 0) {
                if (errno == EINTR && stop_program.load())
                    break;
                continue;
            }

            if (stop_program.load() && channel.result_buffer_tcp.empty())
                break;

            while (!channel.result_buffer_tcp.empty()) {
                const auto &result = channel.result_buffer_tcp.front();
                float value = result.output[0];
                send(client_sock, &value, sizeof(float), 0);
                channel.result_buffer_tcp.pop_front();
                channel.log_count_tcp.fetch_add(1, std::memory_order_relaxed);
                ++output_index;
            }

            if (stop_program.load() && channel.processing_done && channel.acquisition_done && channel.result_buffer_tcp.empty())
                break;
        }

        close(client_sock);
        close(sockfd);
        std::cout << "TCP logging on port " << port << " finished." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Exception in log_results_tcp: " << e.what() << std::endl;
    }
}

template void log_results_tcp<input_type1, output_type1>(Channel<input_type1, output_type1> &, int);
template void log_results_tcp<input_type2, output_type2>(Channel<input_type2, output_type2> &, int);
