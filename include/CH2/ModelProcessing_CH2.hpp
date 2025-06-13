/*ModelProcessing_CH2.hpp*/

#pragma once

#include "../Common/SystemUtils.hpp"
#include "../model1/include/model.h"
#include "Common_CH2.hpp"

#define WITH_CMSIS_NN 1
#define ARM_MATH_DSP 1
#define ARM_NN_TRUNCATE 

void model_inference(Channel_CH2 &channel);
void model_inference_mod(Channel_CH2 &channel);