#include "pattern_recognition_1_1_CODE/definitions.h"

// GeNN userproject includes
#include "analogueRecorder.h"
#include "spikeRecorder.h"

int main()
{
    allocateMem();
    initialize();
    initializeSparse();
    
    SpikeRecorder<SpikeWriterTextCached> inputSpikeRecorder(&getInputCurrentSpikes, &getInputCurrentSpikeCount, "input_spikes.csv", ",", true);
    
    AnalogueRecorder<float> outputRecorder("output.csv", {YOutput, YStarOutput}, 3, ",");

    while(t < 5000.0) {
        stepTime();
        
        // Download state
        pullInputCurrentSpikesFromDevice();
        pullYOutputFromDevice();
        pullYStarOutputFromDevice();
        
        // Record
        inputSpikeRecorder.record(t);
        outputRecorder.record(t);
    }
}