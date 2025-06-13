/*Common.cpp*/

#include "Common/Common.hpp"

std::atomic<bool> stop_acquisition(false);
std::atomic<bool> stop_program(false);

bool save_data_csv = false;
bool save_data_dac = false;
bool save_output_csv = false;
bool save_output_dac = false;
bool save_output_tcp = false;

pid_t pid1 = 0;
pid_t pid2 = 0;
