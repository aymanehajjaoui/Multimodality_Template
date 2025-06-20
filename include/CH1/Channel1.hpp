/*Channel1.hpp*/

#pragma once
#include <string>
#include "Channel.hpp"
#define FC_UNITS1 1

typedef int16_t dense_1_output_type[FC_UNITS1];
typedef int16_t input_type1[48][1];
typedef dense_1_output_type output_type1;

class Channel1 : public Channel<input_type1, output_type1>
{
public:
    Channel1();
    void run_model(const input_type1 input, output_type1 output);
    void runmodel(const input_type1 input, output_type1 output) override;
};
