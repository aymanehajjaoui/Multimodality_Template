/* ModelWriterTCP.hpp */

#pragma once

#include "Common.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"
#include <string>

template <typename In, typename Out>
void log_results_tcp(Channel<In, Out> &channel, int port);