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
bool send_tcp_data(int socket_fd, const T &value)
{
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    if (socket_fd < 0)
        return false;

    ssize_t sent = send(socket_fd, reinterpret_cast<const void *>(&value), sizeof(T), 0);
    if (sent < 0)
    {
        perror("TCP send failed");
        std::cerr << "Error sending data to socket " << socket_fd << std::endl;
        return false;
    }
    return sent == static_cast<ssize_t>(sizeof(T));
}
