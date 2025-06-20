/*DataWriterTCP.hpp*/

#pragma once

#include "TCP.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"

template <typename In, typename Out>
void write_data_tcp(Channel<In, Out>& channel, int socket_fd);