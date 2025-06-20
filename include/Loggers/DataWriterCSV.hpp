/*DataWriterCSV.hpp*/

#pragma once

#include "Common.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"

template <typename In, typename Out>
void write_data_csv(Channel<In, Out> &channel, const std::string &filename);