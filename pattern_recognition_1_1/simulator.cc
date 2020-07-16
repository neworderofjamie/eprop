#include "pattern_recognition_1_1_CODE/definitions.h"

#include <iostream>

// GeNN userproject includes
#include "analogueRecorder.h"
#include "spikeRecorder.h"

// EProp includes
#include "batch_learning.h"
#include "parameters.h"

int main()
{
    try 
    {
        allocateMem();
        initialize();
        
        // Use CUDA to calculate initial transpose of feedforward recurrent->output weights
        BatchLearning::transposeCUDA(d_gRecurrentOutput, d_gOutputRecurrent, 
                                     Parameters::numRecurrentNeurons, Parameters::numOutputNeurons);

        initializeSparse();
        
        // **TEMP** test transpose
        /*pullgRecurrentOutputFromDevice();
        pullgOutputRecurrentFromDevice();
        for(unsigned int i = 0; i < Parameters::numRecurrentNeurons; i++) {
            for(unsigned int j = 0; j < Parameters::numOutputNeurons; j++) {
                std::cout << gRecurrentOutput[(i * Parameters::numOutputNeurons) + j] << ", ";
                assert(gRecurrentOutput[(i * Parameters::numOutputNeurons) + j] == gOutputRecurrent[(j * Parameters::numRecurrentNeurons) + i]);
            }
            std::cout << std::endl;
        }*/
        
        SpikeRecorder<SpikeWriterTextCached> inputSpikeRecorder(&getInputCurrentSpikes, &getInputCurrentSpikeCount, "input_spikes.csv", ",", true);
        SpikeRecorder<SpikeWriterTextCached> recurrentSpikeRecorder(&getRecurrentCurrentSpikes, &getRecurrentCurrentSpikeCount, "recurrent_spikes.csv", ",", true);
        
        AnalogueRecorder<float> outputRecorder("output.csv", {YOutput, YStarOutput}, Parameters::numOutputNeurons, ",");

        // Loop through trials
        float learningRate = -0.0000001f;
        for(unsigned int trial = 0; trial <= 1000; trial++) {
            if((trial % 100) == 0) {
                // if this isn't the first trial, reduce learning rate
                if(trial != 0) {
                    learningRate *= 0.7f;
                }

                std::cout << "Trial " << trial << " (learning rate " << learningRate << ")" << std::endl;
            }
            // Loop through timesteps within trial
            for(unsigned int i = 0; i < 1000; i++) {
                stepTime();
                
                if((trial % 100) == 0) {
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
            // Apply learning
            BatchLearning::fixedRateLearningCUDA(d_DeltaGInputRecurrent, d_gInputRecurrent, 
                                                 Parameters::numInputNeurons, Parameters::numRecurrentNeurons, 
                                                 learningRate);
            BatchLearning::fixedRateLearningCUDA(d_DeltaGRecurrentRecurrent, d_gRecurrentRecurrent, 
                                                 Parameters::numRecurrentNeurons, Parameters::numRecurrentNeurons, 
                                                 learningRate);
            BatchLearning::fixedRateLearningTransposeCUDA(d_DeltaGRecurrentOutput, d_gRecurrentOutput, d_gOutputRecurrent, 
                                                          Parameters::numRecurrentNeurons, Parameters::numOutputNeurons, 
                                                          learningRate);
        }
    }
    catch(std::exception &ex) {
        std::cerr << ex.what() << std::endl;
    }
}