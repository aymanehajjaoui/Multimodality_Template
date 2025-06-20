/* ModelWriterDAC.hpp */

#pragma once

#include "Common.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"
#include "DAC.hpp"
#include <string>
#include <algorithm>

template <typename In, typename Out>
void log_results_dac(Channel<In, Out> &channel, rp_channel_t rp_ch);