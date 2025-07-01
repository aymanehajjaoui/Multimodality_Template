/*Channel2.hh*/

#pragma once
#include <string>
#include "Channel.hpp"
#define FC_UNITS2 1

typedef int8_t dense_1_output_type2[FC_UNITS2];
typedef int8_t input_type2[48][1];
typedef dense_1_output_type2 output_type2;

class Channel2 : public Channel<input_type2, output_type2>
{
public:
    Channel2();
    void run_model1(const input_type2 input, output_type2 output);
    void runmodel(const input_type2 input, output_type2 output) override;
};