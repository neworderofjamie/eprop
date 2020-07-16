#include "pattern_recognition_1_1_CODE/definitions.h"

// GeNN userproject includes
#include "analogueRecorder.h"

int main()
{
    allocateMem();
    initialize();
    initializeSparse();
    
    AnalogueRecorder<float> outputRecorder("output.csv", {YOutput, YStarOutput}, 3, ",");

    while(t < 5000.0) {
        stepTime();
        
        // Download state
        pullYOutputFromDevice();
        pullYStarOutputFromDevice();

        // Record
        outputRecorder.record(t);
    }
}