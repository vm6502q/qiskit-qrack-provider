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

    static std::complex<float> cast_cdouble(std::complex<double> c) {
        return std::complex<float>(c);
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

    virtual void _cu(unsigned char* bits, unsigned char ctrlCount, Qrack::real1 theta, Qrack::real1 phi, Qrack::real1 lambda) {
        Qrack::real1 cos0 = cos(theta / 2);
        Qrack::real1 sin0 = sin(theta / 2);
        Qrack::complex uGate[4] = { Qrack::complex(cos0, ZERO_R1), sin0 * Qrack::complex(-cos(lambda), -sin(lambda)),
                                    sin0 * Qrack::complex(cos(phi), sin(phi)), cos0 * Qrack::complex(cos(phi + lambda), sin(phi + lambda)) };

        qReg->ApplyControlledSingleBit((bitLenInt*)bits, ctrlCount, bits[ctrlCount], uGate);
    }

    virtual void cu(unsigned char* bits, unsigned char ctrlCount, double* params) {
        _cu(bits, ctrlCount, params[0], params[1], params[2]);
    }

    virtual void cu2(unsigned char* bits, unsigned char ctrlCount, double* params) {
        _cu(bits, ctrlCount, (Qrack::real1)(M_PI / 2), params[0], params[1]);
    }

    virtual void cu1(unsigned char* bits, unsigned char ctrlCount, double* params) {
        _cu(bits, ctrlCount, ZERO_R1, ZERO_R1, params[0]);
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

    virtual void ch(unsigned char* bits, unsigned char ctrlCount) {
        const Qrack::complex hadamard[4] = {
            Qrack::complex(M_SQRT1_2, ZERO_R1), Qrack::complex(M_SQRT1_2, ZERO_R1),
            Qrack::complex(M_SQRT1_2, ZERO_R1), Qrack::complex(-M_SQRT1_2, ZERO_R1)
        };
        qReg->ApplyControlledSingleBit((bitLenInt*)bits, ctrlCount, bits[ctrlCount], hadamard);
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

    virtual void rx(unsigned char target, double* params) {
        qReg->RX(params[0], (bitLenInt)target);
    }

    virtual void ry(unsigned char target, double* params) {
        qReg->RY(params[0], (bitLenInt)target);
    }

    virtual void rz(unsigned char target, double* params) {
        qReg->RZ(params[0], (bitLenInt)target);
    }

    virtual void swap(unsigned char target1, unsigned char target2) {
        qReg->Swap((bitLenInt)target1, (bitLenInt)target2);
    }

    virtual void cswap(unsigned char* bits, unsigned char ctrlCount) {
        qReg->CSwap((bitLenInt*)bits, ctrlCount, bits[ctrlCount], bits[ctrlCount + 1U]);
    }

    virtual void initialize(unsigned char* bits, unsigned char bitCount, double* params) {
        bitLenInt origBitCount = qReg->GetQubitCount();
        bitCapInt partPower = Qrack::pow2(bitCount);
        Qrack::complex* amps = new Qrack::complex[partPower];
        for (bitCapInt j = 0; j < partPower; j++) {
            amps[j] = Qrack::complex(Qrack::real1(params[2 * j]), Qrack::real1(params[2 * j + 1]));
        }

        bitLenInt i;

        bool isNatural = true;
        for (i = 0; i < bitCount; i++) {
            if (i != bits[i]) {
                isNatural = false;
                break;
            }
        }

        if (isNatural && (bitCount == origBitCount)) {
            qReg->SetQuantumState(amps);
        } else {
            for (i = 0; i < bitCount; i++) {
                qReg->M(bits[i]);
            }

            Qrack::QInterfacePtr qRegTemp = Qrack::CreateQuantumInterface(Qrack::QINTERFACE_QUNIT, Qrack::QINTERFACE_QFUSION, Qrack::QINTERFACE_OPTIMAL, bitCount, 0);
            qRegTemp->SetQuantumState(amps);
            qReg->Compose(qRegTemp);

            for (i = 0; i < bitCount; i++) {
                qReg->Swap(origBitCount + i, bits[i]);
            }

            qReg->Dispose(origBitCount, bitCount);
        }

        for (i = 0; i < bitCount; i++) {
            qReg->TrySeparate(bits[i]);
        }

        delete[] amps;
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

    virtual unsigned long long measure_all() {
        return qReg->MReg(0, qReg->GetQubitCount());
    }
};

//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
