#ifndef MTJ_PARAMS_H
#define MTJ_PARAMS_H

#include <stdexcept>

// Electrical parameters of one MTJ cell and of the read operation.
//
// PLACEHOLDER VALUES: the defaults below are typical orders of magnitude for
// perpendicular CoFeB/MgO STT-MRAM cells, not measured data. TMR = 100% is a
// conservative choice next to the >120% reported by Ikeda et al., Nature Materials
// 9, 721 (2010). Replace all three with Spintronics Lab measurements when available.
// With noise disabled, the predicted classes do not depend on these values.
struct MTJParams {
    double vRead = 0.1;  // PLACEHOLDER: read voltage magnitude [V]
    double rP = 5.0e3;   // PLACEHOLDER: Parallel-state resistance R_P [ohm]
    double tmr = 1.0;    // PLACEHOLDER: TMR = (R_AP - R_P) / R_P (1.0 = 100%)

    double rAP() const { return rP * (1.0 + tmr); }  // Anti-Parallel resistance [ohm]
    double gP() const { return 1.0 / rP; }           // Parallel conductance [S]
    double gAP() const { return 1.0 / rAP(); }       // Anti-Parallel conductance [S]

    // Differential current of one P/AP pair driven at +V_read, i.e. the bit-line
    // current difference that corresponds to an integer column sum of 1 [A].
    double unitCurrent() const { return vRead * (gP() - gAP()); }

    void validate() const {
        if (!(vRead > 0.0) || !(rP > 0.0) || !(tmr > 0.0)) {
            throw std::invalid_argument("MTJParams: vRead, rP and tmr must all be > 0");
        }
    }
};

#endif  // MTJ_PARAMS_H
