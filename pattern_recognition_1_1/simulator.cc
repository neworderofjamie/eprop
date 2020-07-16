#include "pattern_recognition_1_1_CODE/definitions.h"

#include <iostream>
// GeNN userproject includes
#include "analogueRecorder.h"
#include "spikeRecorder.h"

// EProp includes
#include "transpose.h"


int main()
{
    allocateMem();
    initialize();
    
    // Use CUDA to calculate initial transpose of feedforward recurrent->output weights
    BatchLearning::transposeCUDA(d_gRecurrentOutput, d_gOutputRecurrent, 600, 3);

    initializeSparse();
    
    // **TEMP** test transpose
    /*pullgRecurrentOutputFromDevice();
    pullgOutputRecurrentFromDevice();
    for(unsigned int i = 0; i < 600; i++) {
        for(unsigned int j = 0; j < 3; j++) {
            std::cout << gRecurrentOutput[(i * 3) + j] << ", ";
            assert(gRecurrentOutput[(i * 3) + j] == gOutputRecurrent[(j * 600) + i]);
        }
        std::cout << std::endl;
    }*/
    
    SpikeRecorder<SpikeWriterTextCached> inputSpikeRecorder(&getInputCurrentSpikes, &getInputCurrentSpikeCount, "input_spikes.csv", ",", true);
    SpikeRecorder<SpikeWriterTextCached> recurrentSpikeRecorder(&getRecurrentCurrentSpikes, &getRecurrentCurrentSpikeCount, "recurrent_spikes.csv", ",", true);
    
    AnalogueRecorder<float> outputRecorder("output.csv", {YOutput, YStarOutput}, 3, ",");

    while(t < 5000.0) {
        stepTime();
        
        // Download state
        pullInputCurrentSpikesFromDevice();
        pullRecurrentCurrentSpikesFromDevice();
        pullYOutputFromDevice();
        pullYStarOutputFromDevice();
        
        // Record
        inputSpikeRecorder.record(t);
        recurrentSpikeRecorder.record(t);
        outputRecorder.record(t);
    }
}