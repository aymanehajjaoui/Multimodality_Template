/*Channel1.cpp*/

#include "Channel1.hpp"
#include "full_model1.h"
#include <iostream>

Channel1::Channel1()
{
    name = "Channel1";
    rp_channel= RP_CH_1;
}

void Channel1::run_model1(const input_type1 input, output_type1 output)
{
    cnn1(input, output);
}

void Channel1::runmodel(const input_type1 input, output_type1 output)
{
    run_model1(input, output);
}
