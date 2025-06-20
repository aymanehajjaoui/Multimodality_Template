/*ModelProcessing.hpp*/

#pragma once

#include "SystemUtils.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"

template <typename In, typename Out>
void model_inference(Channel<In, Out> &channel);

template <typename In, typename Out>
void model_inference_mod(Channel<In, Out> &channel);