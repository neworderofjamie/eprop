#include "modelSpec.h"

#include "parameters.h"

constexpr double PI = 3.14159265358979323846264338327950288419;
//----------------------------------------------------------------------------
// Recurrent
//----------------------------------------------------------------------------
class Recurrent : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Recurrent, 3, 3);

    SET_SIM_CODE(
        "$(V) = ($(Alpha) * $(V)) + $(Isyn);\n"
        "$(trace) *= $(Alpha);\n"
        "if ($(RefracTime) > 0.0) {\n"
        "  $(RefracTime) -= DT;\n"
        "  $(Psi) = 0.0;\n"
        "}\n"
        "else {\n"
        "  $(Psi) = (1.0 / $(Vthresh)) * fmax(0.0, 1.0 - fabs(($(V) - $(Vthresh)) / $(Vthresh)));\n"
        "}\n"
    );

    SET_THRESHOLD_CONDITION_CODE("$(RefracTime) <= 0.0 && $(V) >= $(Vthresh)");

    SET_RESET_CODE(
        "$(RefracTime) = $(TauRefrac);\n"
        "$(trace) += 1.0f;\n"
        "$(V) -= $(Vthresh);\n");

    SET_PARAM_NAMES({
        "TauM",         // Membrane time constant [ms]
        "Vthresh",      // Spiking threshold [mV]
        "TauRefrac"});  // Refractory time constant [ms]

    SET_DERIVED_PARAMS({
        {"Alpha", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});

    SET_VARS({{"V", "scalar"}, {"Psi", "scalar"}, {"RefracTime", "scalar"}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(Recurrent);

//----------------------------------------------------------------------------
// Input
//----------------------------------------------------------------------------
class Input : public NeuronModels::Base
{
public:
    DECLARE_MODEL(Input, 1, 3);

    SET_SIM_CODE("$(trace) *= $(Alpha);");

    SET_THRESHOLD_CONDITION_CODE(
        "$(startSpike) != $(endSpike) && "
        "$(t) >= $(spikeTimes)[$(startSpike)]");

    SET_RESET_CODE(
        "$(startSpike)++;\n"
        "$(trace) += 1.0;\n");

    SET_PARAM_NAMES({"TauIn"});
    SET_VARS({{"trace", "scalar"}, {"startSpike", "unsigned int"}, {"endSpike", "unsigned int", VarAccess::READ_ONLY}});
    SET_EXTRA_GLOBAL_PARAMS({{"spikeTimes", "scalar*"}});

    SET_DERIVED_PARAMS({
        {"Alpha", [](const std::vector<double> &pars, double dt){ return std::exp(-dt / pars[0]); }}});

    SET_NEEDS_AUTO_REFRACTORY(false);
};
IMPLEMENT_MODEL(Input);

//----------------------------------------------------------------------------
// Output
//----------------------------------------------------------------------------
class OutputRegression : public NeuronModels::Base
{
public:
    DECLARE_MODEL(OutputRegression, 5, 9);

    SET_SIM_CODE(
        "$(Y) = ($(Kappa) * $(Y)) + $(Isyn) + $(Bias);\n"
        "$(Ystar) = $(Ampl1) * sin(($(Freq1Radians) * $(t)) + $(Phase1));\n"
        "$(Ystar) += $(Ampl2) * sin(($(Freq2Radians) * $(t)) + $(Phase2));\n"
        "$(Ystar) += $(Ampl3) * sin(($(Freq3Radians) * $(t)) + $(Phase3));\n"
        "$(E) = $(Y) - $(Ystar);\n");

    SET_PARAM_NAMES({
        "TauOut",       // Membrane time constant [ms]
        "Bias",         // Bias [mV]
        "Freq1",        // Frequency of sine wave 1 (Hz)
        "Freq2",        // Frequency of sine wave 2 (Hz)
        "Freq3"});      // Frequency of sine wave 3 (Hz)

    SET_VARS({{"Y", "scalar"}, {"Ystar", "scalar"}, {"E", "scalar"},
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


void modelDefinition(ModelSpec &model)
{
    model.setDT(1.0);
    model.setName("pattern_recognition_1_1");

    //---------------------------------------------------------------------------
    // Parameters and state variables
    //---------------------------------------------------------------------------
    OutputRegression::ParamValues outputParamVals(
        20.0,   // Membrane time constant [ms]
        0.0,    // Bias [mV]
        2.0,    // Frequency of sine wave 1 (Hz)
        3.0,    // Frequency of sine wave 2 (Hz)
        5.0);  // Frequency of sine wave 3 (Hz)
    
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
    
    //---------------------------------------------------------------------------
    // Neuron populations
    //---------------------------------------------------------------------------
    model.addNeuronPopulation<OutputRegression>("Output", 3, outputParamVals, outputInitVals);
}