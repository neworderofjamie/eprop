#include "pattern_recognition_1_1_CODE/definitions.h"

// GeNN userproject includes
#include "analogueRecorder.h"

int main()
{
    allocateMem();
    initialize();
    initializeSparse();
    
    AnalogueRecorder<float> outputRecorder("output.csv", {YStarOutput}, 3, ",");

    while(t < 5000.0) {
        stepTime();
        
        pullYStarOutputFromDevice();

        outputRecorder.record(t);
    }
}