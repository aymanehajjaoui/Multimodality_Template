/*DataWriterDAC.hpp*/

#pragma once

#include "DAC.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"

template <typename In, typename Out>
void write_data_dac(Channel<In, Out> &channel);