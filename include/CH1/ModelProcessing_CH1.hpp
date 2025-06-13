/*ModelProcessing_CH1.hpp*/

#pragma once

#include "../Common/SystemUtils.hpp"
#include "../model1/include/model.h"
#include "Common_CH1.hpp"

#define WITH_CMSIS_NN 1
#define ARM_MATH_DSP 1
#define ARM_NN_TRUNCATE 

void model_inference(Channel_CH1 &channel);
void model_inference_mod(Channel_CH1 &channel);