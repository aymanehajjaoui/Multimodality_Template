/*Channel1.hh*/

#pragma once
#include <string>
#include "Channel.hpp"
#define FC_UNITS1 1

typedef float dense_1_output_type1[FC_UNITS1];
typedef float input_type1[48][1];
typedef dense_1_output_type1 output_type1;

class Channel1 : public Channel<input_type1, output_type1>
{
public:
    Channel1();
    void run_model(const input_type1 input, output_type1 output);
    void runmodel(const input_type1 input, output_type1 output) override;
};