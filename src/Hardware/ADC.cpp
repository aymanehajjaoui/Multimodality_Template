/* ADC.cpp */

#include "Hardware/ADC.hpp"
#include "Channel1.hpp"
#include "Channel2.hpp"
#include <iostream>

void initialize_acq()
{
    constexpr uint32_t DECIMATION_CH1 = Channel1::get_decimation();
    constexpr uint32_t DECIMATION_CH2 = Channel2::get_decimation();

    rp_AcqResetFpga();
    rp_AcqReset();

    if (rp_AcqSetSplitTrigger(true) != RP_OK)
        std::cerr << "rp_AcqSetSplitTrigger failed!" << std::endl;

    if (rp_AcqSetSplitTriggerPass(true) != RP_OK)
        std::cerr << "rp_AcqSetSplitTriggerPass failed!" << std::endl;

    uint32_t g_adc_axi_start, g_adc_axi_size;
    if (rp_AcqAxiGetMemoryRegion(&g_adc_axi_start, &g_adc_axi_size) != RP_OK)
    {
        std::cerr << "rp_AcqAxiGetMemoryRegion failed!" << std::endl;
        exit(-1);
    }

    std::cout << "Reserved memory Start 0x" << std::hex << g_adc_axi_start
              << " Size 0x" << g_adc_axi_size << std::dec << std::endl;

    // Set decimation for each channel based on model input size
    if (rp_AcqAxiSetDecimationFactorCh(RP_CH_1, DECIMATION_CH1) != RP_OK)
    {
        std::cerr << "rp_AcqAxiSetDecimationFactor CH1 failed!" << std::endl;
        exit(-1);
    }

    if (rp_AcqAxiSetDecimationFactorCh(RP_CH_2, DECIMATION_CH2) != RP_OK)
    {
        std::cerr << "rp_AcqAxiSetDecimationFactor CH2 failed!" << std::endl;
        exit(-1);
    }

    float sampling_rate;
    if (rp_AcqGetSamplingRateHz(&sampling_rate) == RP_OK)
    {
        printf("Current Sampling Rate: %.2f Hz\n", sampling_rate);
    }
    else
    {
        fprintf(stderr, "Failed to get sampling rate\n");
    }

    // Set trigger delay to 0
    if (rp_AcqAxiSetTriggerDelay(RP_CH_1, 0) != RP_OK)
        std::cerr << "Trigger delay CH1 failed!" << std::endl;

    if (rp_AcqAxiSetTriggerDelay(RP_CH_2, 0) != RP_OK)
        std::cerr << "Trigger delay CH2 failed!" << std::endl;

    // Allocate buffer space
    if (rp_AcqAxiSetBufferSamples(RP_CH_1, g_adc_axi_start, DATA_SIZE) != RP_OK)
        std::cerr << "SetBuffer CH1 failed!" << std::endl;

    if (rp_AcqAxiSetBufferSamples(RP_CH_2, g_adc_axi_start + (g_adc_axi_size / 2), DATA_SIZE) != RP_OK)
        std::cerr << "SetBuffer CH2 failed!" << std::endl;

    if (rp_AcqAxiEnable(RP_CH_1, true) != RP_OK)
        std::cerr << "Enable CH1 failed!" << std::endl;

    if (rp_AcqAxiEnable(RP_CH_2, true) != RP_OK)
        std::cerr << "Enable CH2 failed!" << std::endl;

    // Trigger config
    if (rp_AcqSetTriggerLevel(RP_T_CH_1, 0) != RP_OK)
        std::cerr << "Trigger level CH1 failed!" << std::endl;

    if (rp_AcqSetTriggerLevel(RP_T_CH_2, 0) != RP_OK)
        std::cerr << "Trigger level CH2 failed!" << std::endl;

    if (rp_AcqSetTriggerSrcCh(RP_CH_1, RP_TRIG_SRC_CHA_PE) != RP_OK)
        std::cerr << "Trigger source CH1 failed!" << std::endl;

    if (rp_AcqSetTriggerSrcCh(RP_CH_2, RP_TRIG_SRC_CHB_PE) != RP_OK)
        std::cerr << "Trigger source CH2 failed!" << std::endl;

    // Start acquisition
    if (rp_AcqStartCh(RP_CH_1) != RP_OK)
        std::cerr << "Start CH1 failed!" << std::endl;

    if (rp_AcqStartCh(RP_CH_2) != RP_OK)
        std::cerr << "Start CH2 failed!" << std::endl;
}

void cleanup()
{
    std::cout << "\nReleasing resources\n";
    rp_AcqStopCh(RP_CH_1);
    rp_AcqStopCh(RP_CH_2);
    rp_AcqAxiEnable(RP_CH_1, false);
    rp_AcqAxiEnable(RP_CH_2, false);
    rp_Release();
    std::cout << "Cleanup done." << std::endl;
}
