/*TCP.hpp*/

#pragma once

#include "Common.hpp"
#include <string>
#include <netinet/in.h>
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <fstream>
#include <sstream>

int initialize_tcp(const std::string &ip, int port);
int get_gateway_ip(char *ip_buffer, size_t size);

template <typename T>
bool send_tcp_array(int socket_fd, const T *data, size_t count)
{
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    if (socket_fd < 0 || data == nullptr || count == 0)
        return false;

    const char *ptr = reinterpret_cast<const char *>(data);
    size_t total_bytes = sizeof(T) * count;
    size_t bytes_sent = 0;

    while (bytes_sent < total_bytes)
    {
        ssize_t sent = send(socket_fd, ptr + bytes_sent, total_bytes - bytes_sent, 0);
        if (sent <= 0)
        {
            perror("TCP send array failed");
            return false;
        }
        bytes_sent += static_cast<size_t>(sent);
    }

    return true;
}
