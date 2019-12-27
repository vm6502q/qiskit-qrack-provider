/**
 * (C) Copyright Daniel Strano and the Qrack Contributors 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _qrack_controller_hpp_
#define _qrack_controller_hpp_

#include <cstdint>

#include "qrack/qfactory.hpp"
#include "qrack/common/config.h"

namespace AER {
namespace Simulator {

//=========================================================================
// QrackController class
//=========================================================================

class QrackController {
protected:
    Qrack::QInterfacePtr qReg;

    static std::complex<double> cast_cfloat(std::complex<float> c) {
        return std::complex<double>(c);
    }

    static double cast_float(float c) {
        return (double)c;
    }

public:
    QrackController() = default;
    virtual ~QrackController() = default;

    virtual void initialize_qreg(unsigned char num_qubits) {
        qReg = Qrack::CreateQuantumInterface(Qrack::QINTERFACE_QUNIT, Qrack::QINTERFACE_QFUSION, Qrack::QINTERFACE_OPTIMAL, num_qubits, 0);
    }

//-------------------------------------------------------------------------
// Operations
//-------------------------------------------------------------------------

    virtual void u(unsigned char target, double* params) {
        qReg->U((bitLenInt)target, params[0], params[1], params[2]);
    }

    virtual void u2(unsigned char target, double* params) {
        qReg->U2((bitLenInt)target, params[0], params[1]);
    }

    virtual void u1(unsigned char target, double* params) {
        qReg->RT(params[0] * 2, (bitLenInt)target);
    }

    virtual void cx(unsigned char* bits, unsigned char ctrlCount) {
        const Qrack::complex pauliX[4] = {
            Qrack::ZERO_CMPLX, Qrack::ONE_CMPLX,
            Qrack::ONE_CMPLX, Qrack::ZERO_CMPLX
        };
        qReg->ApplyControlledSingleBit((bitLenInt*)bits, ctrlCount, bits[ctrlCount], pauliX);
    }

    virtual void cz(unsigned char* bits, unsigned char ctrlCount) {
        const Qrack::complex pauliZ[4] = {
            Qrack::ONE_CMPLX, Qrack::ZERO_CMPLX,
            Qrack::ZERO_CMPLX, -Qrack::ONE_CMPLX
        };
        qReg->ApplyControlledSingleBit((bitLenInt*)bits, ctrlCount, bits[ctrlCount], pauliZ);
    }

    virtual void h(unsigned char target) {
        qReg->H((bitLenInt)target);
    }

    virtual void x(unsigned char target) {
        qReg->X((bitLenInt)target);
    }

    virtual void y(unsigned char target) {
        qReg->Y((bitLenInt)target);
    }

    virtual void z(unsigned char target) {
        qReg->Z((bitLenInt)target);
    }

    virtual void s(unsigned char target) {
        qReg->S((bitLenInt)target);
    }

    virtual void sdg(unsigned char target) {
        qReg->IS((bitLenInt)target);
    }

    virtual void t(unsigned char target) {
        qReg->T((bitLenInt)target);
    }

    virtual void tdg(unsigned char target) {
        qReg->IT((bitLenInt)target);
    }

    virtual void swap(unsigned char target1, unsigned char target2) {
        qReg->Swap((bitLenInt)target1, (bitLenInt)target2);
    }

    virtual void reset(unsigned char target) {
        qReg->SetBit((bitLenInt)target, false);
    }

    virtual std::vector<std::complex<double>> amplitudes() {
        std::vector<std::complex<Qrack::real1>> amps(qReg->GetMaxQPower());
        qReg->GetQuantumState(&(amps[0]));

#if ENABLE_COMPLEX8
        std::vector<std::complex<double>> ampsDouble(amps.size());
        std::transform(amps.begin(), amps.end(), ampsDouble.begin(), cast_cfloat);
        return ampsDouble;
#else
        return amps;
#endif
    }

    virtual std::vector<double> probabilities() {
        bitCapInt probCount = qReg->GetMaxQPower();
        std::vector<double> probs(probCount);
        double totProb = 0;
        for (bitCapInt i = 0; i < probCount; i++) {
            probs[i] = (double)qReg->ProbAll(i);
            totProb += probs[i];
        }

        for (bitCapInt i = 0; i < probCount; i++) {
            probs[i] /= totProb;
        }

        return probs;
    }

    virtual int measure_all() {
        return qReg->MReg(0, qReg->GetQubitCount());
    }
};

//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
