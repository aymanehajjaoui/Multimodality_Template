/*ModelWriterCSV.hpp*/

#pragma once

#include <string>
#include "Common.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"
#include "SystemUtils.hpp"

template <typename In, typename Out>
void log_results_csv(Channel<In, Out> &channel, const std::string &filename);