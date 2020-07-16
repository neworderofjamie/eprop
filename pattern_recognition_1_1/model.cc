#include "modelSpec.h"

#include "parameters.h"

constexpr double PI = 3.14159265358979323846264338327950288419;

//----------------------------------------------------------------------------
// Recurrent
//----------------------------------------------------------------------------
class Recurrent : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Recurrent, 3, 5);

    SET_SIM_CODE(
        "$(E) = $(IsynFeedback);\n"
        "$(V) = ($(Alpha) * $(V)) + $(Isyn);\n"
        "$(ZFilter) *= $(Alpha);\n"
        "if ($(RefracTime) > 0.0) {\n"
        "  $(RefracTime) -= DT;\n"
        "  $(Psi) = 0.0;\n"
        "}\n"
        "else {\n"
        "  $(Psi) = (1.0 / $(Vthresh)) * fmax(0.0, 1.0 - fabs(($(V) - $(Vthresh)) / $(Vthresh)));\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(RefracTime) = $(TauRefrac);\n"
        "$(ZFilter) += 1.0f;\n"
        "$(V) -= $(Vthresh);\n");

    SET_PARAM_NAMES({
        "TauM",         // Membrane time constant [ms]
        "Vthresh",      // Spiking threshold [mV]
        "TauRefrac"});  // Refractory time constant [ms]

    SET_VARS({{"V", "scalar"}, {"Psi", "scalar"}, {"RefracTime", "scalar"}, 
              {"ZFilter", "scalar"}, {"E", "scalar"}});

    SET_DERIVED_PARAMS({
        {"Alpha", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});

    SET_ADDITIONAL_INPUT_VARS({{"IsynFeedback", "scalar", 0.0}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(Recurrent);

//----------------------------------------------------------------------------
// Input
//----------------------------------------------------------------------------
class Input : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Input, 5, 2);

    SET_SIM_CODE(
        "const scalar tPattern = fmod($(t), $(PatternLength));\n"
        "const unsigned int neuronGroup = $(id) / (unsigned int)$(GroupSize);\n"
        "const scalar groupStartTime = neuronGroup * $(ActiveInterval);\n"
        "const scalar groupEndTime = groupStartTime + $(ActiveInterval);\n"
        "$(ZFilter) *= $(Alpha);\n"
        "if ($(RefracTime) > 0.0) {\n"
        "  $(RefracTime) -= DT;\n"
        "}\n");

    SET_THRESHOLD_CONDITION_CODE(
        "tPattern > groupStartTime && tPattern < groupEndTime && $(RefracTime) <= 0.0");

    SET_RESET_CODE(
        "$(RefracTime) = $(TauRefrac);\n"
        "$(ZFilter) += 1.0;\n");

    SET_PARAM_NAMES({
        "TauIn",            // Membrane time constant [ms]
        "GroupSize",        // Number of neurons in each group
        "ActiveInterval",   // How long each group is active for [ms]
        "ActiveRate",       // Rate active neurons fire at [Hz]
        "PatternLength"});  // Pattern length [ms]
        
    SET_VARS({{"ZFilter", "scalar"}, {"RefracTime", "scalar"}});
   
    SET_DERIVED_PARAMS({
        {"Alpha", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"TauRefrac", [](const std::vector<double> &pars, double){ return 1000.0 / pars[3]; }}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(Input);

//----------------------------------------------------------------------------
// Output
//----------------------------------------------------------------------------
class OutputRegression : public NeuronModels::Base
{
public:
    DECLARE_MODEL(OutputRegression, 6, 9);

    SET_SIM_CODE(
        "$(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(Bias);\n"
        "const scalar tPattern = fmod($(t), $(PatternLength));\n"
        "$(YStar) = $(Ampl1) * sin(($(Freq1Radians) * tPattern) + $(Phase1));\n"
        "$(YStar) += $(Ampl2) * sin(($(Freq2Radians) * tPattern) + $(Phase2));\n"
        "$(YStar) += $(Ampl3) * sin(($(Freq3Radians) * tPattern) + $(Phase3));\n"
        "$(E) = $(Y) - $(YStar);\n");

    SET_PARAM_NAMES({
        "TauOut",           // Membrane time constant [ms]
        "Bias",             // Bias [mV]
        "Freq1",            // Frequency of sine wave 1 [Hz]
        "Freq2",            // Frequency of sine wave 2 [Hz]
        "Freq3",            // Frequency of sine wave 3 [Hz]
        "PatternLength"});  // Pattern length [ms]

    SET_VARS({{"Y", "scalar"}, {"YStar", "scalar"}, {"E", "scalar"},
              {"Ampl1", "scalar", VarAccess::READ_ONLY}, {"Ampl2", "scalar", VarAccess::READ_ONLY}, {"Ampl3", "scalar", VarAccess::READ_ONLY},
              {"Phase1", "scalar", VarAccess::READ_ONLY}, {"Phase2", "scalar", VarAccess::READ_ONLY}, {"Phase3", "scalar", VarAccess::READ_ONLY}});

    SET_DERIVED_PARAMS({
        {"Kappa", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }},
        {"Freq1Radians", [](const std::vector<double> &pars, double){ return pars[2] * 2.0 * PI / 1000.0; }},
        {"Freq2Radians", [](const std::vector<double> &pars, double){ return pars[3] * 2.0 * PI / 1000.0; }},
        {"Freq3Radians", [](const std::vector<double> &pars, double){ return pars[4] * 2.0 * PI / 1000.0; }}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(OutputRegression);

//---------------------------------------------------------------------------
// Feedback
//---------------------------------------------------------------------------
//! Simple postsynaptic model which transfer input directly to neuron without any dynamics
class Feedback : public PostsynapticModels::Base
{
public:
    DECLARE_MODEL(Feedback, 0, 0);

    SET_APPLY_INPUT_CODE(
        "$(IsynFeedback) += $(inSyn);\n"
        "$(inSyn) = 0;\n");
};
IMPLEMENT_MODEL(Feedback);

//---------------------------------------------------------------------------
// Continuous
//---------------------------------------------------------------------------
//! Simple continous synapse for error feedback
class Continuous : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(Continuous, 0, 1);

    SET_VARS({{"g", "scalar"}});

    SET_SYNAPSE_DYNAMICS_CODE("$(addToInSyn, $(g) * $(E_pre));\n");
};
IMPLEMENT_MODEL(Continuous);

//---------------------------------------------------------------------------
// EProp
//---------------------------------------------------------------------------
//! Basic implementation of EProp learning rule
class EProp : public WeightUpdateModels::Base
{
public:
    DECLARE_MODEL(EProp, 1, 3);
    
    SET_SIM_CODE("$(addToInSyn, $(g));\n");
    
    SET_SYNAPSE_DYNAMICS_CODE(
        "const scalar e = $(ZFilter_pre) * $(Psi_post);\n"
        "scalar eFiltered = $(eFiltered);\n"
        "eFiltered = (eFiltered * $(Alpha)) + e;\n"
        "$(DeltaG) += eFiltered * $(E_post);\n"
        "$(eFiltered) = eFiltered;\n");
    
    SET_PARAM_NAMES({"TauE"});  // Eligibility trace time constant [ms]
        
    SET_VARS({{"g", "scalar"}, {"eFiltered", "scalar"}, {"DeltaG", "scalar"}});
    
    SET_DERIVED_PARAMS({
        {"Alpha", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});
    
};
IMPLEMENT_MODEL(EProp);

void modelDefinition(ModelSpec &model)
{
    // Calculate weight scaling factor
    // **NOTE** "Long short-term memory and 
    // learning-to-learn in networks of spiking neurons"
    // suggests that this should be (1 Volt * DT)/Rm but
    // that results in 1E-9 or something which is never 
    // going to make these neurons spike - the paper then 
    // SORT OF suggests that they use 1.0
    const double weight0 = 1.0;
 
    model.setDT(1.0);
    model.setName("pattern_recognition_1_1");
    model.setMergePostsynapticModels(true);

    //---------------------------------------------------------------------------
    // Parameters and state variables
    //---------------------------------------------------------------------------
    // Input population
    Input::ParamValues inputParamVals(
        20.0,       // Membrane time constant [ms]
        4,          // Number of neurons in each group
        200.0,      // How long each group is active for [ms]
        100.0,      // Rate active neurons fire at [Hz]
        1000.0);    // Pattern length [ms]

    Input::VarValues inputInitVals(
        0.0,    // Trace
        0.0);   // Refrac time

    // Recurrent population
    Recurrent::ParamValues recurrentParamVals(
        20.0,   // Membrane time constant [ms]
        0.61,   // Spiking threshold [mV]
        5.0);   // Refractory time constant [ms]

    Recurrent::VarValues recurrentInitVals(
        0.0,    // V
        0.0,    // Psi
        0.0,    // RefracTime
        0.0,    // ZFilter
        0.0);   // E

    // Output population
    OutputRegression::ParamValues outputParamVals(
        20.0,       // Membrane time constant [ms]
        0.0,        // Bias [mV]
        2.0,        // Frequency of sine wave 1 [Hz]
        3.0,        // Frequency of sine wave 2 [Hz]
        5.0,        // Frequency of sine wave 3 [Hz]
        1000.0);    // Pattern length [ms]

    InitVarSnippet::Uniform::ParamValues outputAmplDist(0.5, 2.0);
    InitVarSnippet::Uniform::ParamValues outputPhaseDist(0.0, 2.0 * PI);
    OutputRegression::VarValues outputInitVals(
        0.0,                                                // Y
        0.0,                                                // Y*
        0.0,                                                // E
        initVar<InitVarSnippet::Uniform>(outputAmplDist),   // Ampl1
        initVar<InitVarSnippet::Uniform>(outputAmplDist),   // Ampl2
        initVar<InitVarSnippet::Uniform>(outputAmplDist),   // Ampl3
        initVar<InitVarSnippet::Uniform>(outputPhaseDist),  // Phase1
        initVar<InitVarSnippet::Uniform>(outputPhaseDist),  // Phase2
        initVar<InitVarSnippet::Uniform>(outputPhaseDist)); // Phase3

    EProp::ParamValues epropParamVals(20.0);    // Eligibility trace time constant [ms]
    
    // Feedforward input->recurrent connections
    InitVarSnippet::Normal::ParamValues inputRecurrentWeightDist(0.0, weight0 / sqrt(20.0));
    EProp::VarValues inputRecurrentInitVals(
        initVar<InitVarSnippet::Normal>(inputRecurrentWeightDist),  // g
        0.0,                                                        // eFiltered
        0.0);                                                       // DeltaG

    // Recurrent connections
    InitVarSnippet::Normal::ParamValues recurrentRecurrentWeightDist(0.0, weight0 / sqrt(600.0));
    EProp::VarValues recurrentRecurrentInitVals(
        initVar<InitVarSnippet::Normal>(recurrentRecurrentWeightDist),  // g
        0.0,                                                            // eFiltered
        0.0);                                                           // DeltaG

    // Feedforward recurrent->output connections
    InitVarSnippet::Normal::ParamValues recurrentOutputWeightDist(0.0, weight0 / sqrt(600.0));
    WeightUpdateModels::StaticPulse::VarValues recurrentOutputInitVals(
        initVar<InitVarSnippet::Normal>(recurrentOutputWeightDist));    // g

    // Feedback connections
    // **HACK** this is actually a nasty corner case for the initialisation rules
    // We really want this uninitialised as we are going to copy over transpose 
    // But then initialiseSparse would copy over host values
    Continuous::VarValues outputRecurrentInitVals(0.0);

    //---------------------------------------------------------------------------
    // Neuron populations
    //---------------------------------------------------------------------------
    model.addNeuronPopulation<Input>("Input", 20, inputParamVals, inputInitVals);
    model.addNeuronPopulation<Recurrent>("Recurrent", 600, recurrentParamVals, recurrentInitVals);
    model.addNeuronPopulation<OutputRegression>("Output", 3, outputParamVals, outputInitVals);

    //---------------------------------------------------------------------------
    // Synapse populations
    //---------------------------------------------------------------------------
    model.addSynapsePopulation<EProp, PostsynapticModels::DeltaCurr>(
        "InputRecurrent", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Input", "Recurrent",
        epropParamVals, inputRecurrentInitVals,
        {}, {});

    model.addSynapsePopulation<Continuous, Feedback>(
        "OutputRecurrent", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Output", "Recurrent",
        {}, outputRecurrentInitVals,
        {}, {});

    model.addSynapsePopulation<EProp, PostsynapticModels::DeltaCurr>(
        "RecurrentRecurrent", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Recurrent", "Recurrent",
        epropParamVals, recurrentRecurrentInitVals,
        {}, {});

    model.addSynapsePopulation<WeightUpdateModels::StaticPulse, PostsynapticModels::DeltaCurr>(
        "RecurrentOutput", SynapseMatrixType::DENSE_INDIVIDUALG, NO_DELAY,
        "Recurrent", "Output",
        {}, recurrentOutputInitVals,
        {}, {});
}