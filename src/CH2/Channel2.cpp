/*Channel2.cpp*/

#include "Channel2.hpp"
#include "full_model2.h"
#include <iostream>

Channel2::Channel2()
{
    name = "Channel2";
    rp_channel = RP_CH_2;
}

void Channel2::run_model2(const input_type2 input, output_type2 output)
{
    cnn2(input, output);
}

void Channel2::runmodel(const input_type2 input, output_type2 output)
{
    run_model2(input, output);
}
