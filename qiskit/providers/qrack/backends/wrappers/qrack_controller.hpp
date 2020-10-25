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

#define MAKE_ENGINE(num_qubits, perm) Qrack::CreateQuantumInterface(qIType1, qIType2, num_qubits, perm, nullptr, Qrack::complex(-999.0, -999.0), doNorm, false, false, deviceId, true, amplitudeFloor)

class QrackController {
protected:
    Qrack::QInterfacePtr qReg;
    Qrack::QInterfaceEngine qIType1;
    Qrack::QInterfaceEngine qIType2;
    int deviceId;
    bool doNorm;
    Qrack::real1 amplitudeFloor;

    static inline std::complex<double> cast_cfloat(std::complex<float> c) {
        return std::complex<double>(c);
    }

    static inline std::complex<float> cast_cdouble(std::complex<double> c) {
        return std::complex<float>(c);
    }

    static inline double cast_float(float c) {
        return (double)c;
    }

    static inline double normHelper(Qrack::complex c) { return norm(c); }

public:
    QrackController() = default;
    virtual ~QrackController() = default;

    virtual void initialize_qreg(bool use_opencl, bool use_qunit, unsigned char num_qubits, int device_id, bool opencl_multi, bool doNormalize, Qrack::real1 zero_threshold) {
        deviceId = device_id;
        doNorm = doNormalize;
        amplitudeFloor = zero_threshold;

        qIType2 = use_opencl ? Qrack::QINTERFACE_OPTIMAL : Qrack::QINTERFACE_CPU;
        qIType1 = use_qunit ? (opencl_multi ? Qrack::QINTERFACE_QUNIT_MULTI : Qrack::QINTERFACE_QUNIT) : qIType2;

        qReg = MAKE_ENGINE(num_qubits, 0);
    }

//-------------------------------------------------------------------------
// Operations
//-------------------------------------------------------------------------

    virtual void u(unsigned char* bits, unsigned char bitCount, double* params) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->U((bitLenInt)bits[i], params[0], params[1], params[2]);
        }
    }

    virtual void u2(unsigned char* bits, unsigned char bitCount, double* params) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->U2((bitLenInt)bits[i], params[0], params[1]);
        }
    }

    virtual void u1(unsigned char* bits, unsigned char bitCount, double* params) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->RT(params[0] * 2, (bitLenInt)bits[i]);
        }
    }

    virtual void unitary1qb(unsigned char* bits, unsigned char bitCount, double* params) {
        Qrack::complex mtrx[4];
        _darray_to_creal1_array(params, 4, mtrx);

        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->ApplySingleBit(mtrx, (bitLenInt)bits[i]);
        }
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
        qReg->ApplyControlledSingleInvert((bitLenInt*)bits, ctrlCount, bits[ctrlCount], Qrack::ONE_CMPLX, Qrack::ONE_CMPLX);
    }

    virtual void cz(unsigned char* bits, unsigned char ctrlCount) {
        qReg->ApplyControlledSinglePhase((bitLenInt*)bits, ctrlCount, bits[ctrlCount], Qrack::ONE_CMPLX, -Qrack::ONE_CMPLX);
    }

    virtual void ch(unsigned char* bits, unsigned char ctrlCount) {
        if (ctrlCount == 0) {
            qReg->H(bits[0]);
            return;
        }
        if (ctrlCount == 1) {
            qReg->CH(bits[0], bits[1]);
            return;
        }

        const Qrack::complex hadamard[4] = {
            Qrack::complex(M_SQRT1_2, ZERO_R1), Qrack::complex(M_SQRT1_2, ZERO_R1),
            Qrack::complex(M_SQRT1_2, ZERO_R1), Qrack::complex(-M_SQRT1_2, ZERO_R1)
        };
        qReg->ApplyControlledSingleBit((bitLenInt*)bits, ctrlCount, bits[ctrlCount], hadamard);
    }

    virtual void h(unsigned char* bits, unsigned char bitCount) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->H((bitLenInt)bits[i]);
        }
    }

    virtual void x(unsigned char* bits, unsigned char bitCount) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->X((bitLenInt)bits[i]);
        }
    }
    
    virtual void sx(unsigned char* bits, unsigned char bitCount) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->SqrtX((bitLenInt)bits[i]);
        }
    }

    virtual void y(unsigned char* bits, unsigned char bitCount) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->Y((bitLenInt)bits[i]);
        }
    }

    virtual void z(unsigned char* bits, unsigned char bitCount) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->Z((bitLenInt)bits[i]);
        }
    }

    virtual void s(unsigned char* bits, unsigned char bitCount) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->IS((bitLenInt)bits[i]);
        }
    }

    virtual void sdg(unsigned char* bits, unsigned char bitCount) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->S((bitLenInt)bits[i]);
        }
    }

    virtual void t(unsigned char* bits, unsigned char bitCount) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->IT((bitLenInt)bits[i]);
        }
    }

    virtual void tdg(unsigned char* bits, unsigned char bitCount) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->T((bitLenInt)bits[i]);
        }
    }

    virtual void rx(unsigned char* bits, unsigned char bitCount, double* params) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->RX(params[0], (bitLenInt)bits[i]);
        }
    }

    virtual void ry(unsigned char* bits, unsigned char bitCount, double* params) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->RY(params[0], (bitLenInt)bits[i]);
        }
    }

    virtual void rz(unsigned char* bits, unsigned char bitCount, double* params) {
        for (bitLenInt i = 0; i < bitCount; i++) {
            qReg->RZ(params[0], (bitLenInt)bits[i]);
        }
    }

    virtual void swap(unsigned char target1, unsigned char target2) {
        qReg->Swap((bitLenInt)target1, (bitLenInt)target2);
    }

    virtual void cswap(unsigned char* bits, unsigned char ctrlCount) {
        qReg->CSwap((bitLenInt*)bits, ctrlCount, bits[ctrlCount], bits[ctrlCount + 1U]);
    }

    virtual void _darray_to_creal1_array(double *params, bitCapInt componentCount, Qrack::complex* amps) {
        for (bitCapInt j = 0; j < componentCount; j++) {
            amps[j] = Qrack::complex(Qrack::real1(params[2 * j]), Qrack::real1(params[2 * j + 1]));
        }
    }

    virtual void initialize(unsigned char* bits, unsigned char bitCount, double* params) {
        bitLenInt origBitCount = qReg->GetQubitCount();
        bitCapInt partPower = Qrack::pow2(bitCount);
        Qrack::complex* amps = new Qrack::complex[partPower];
        _darray_to_creal1_array(params, partPower, amps);

        bitLenInt i;

        bool isNatural = (bitCount == origBitCount);
        if (isNatural) {
            for (i = 0; i < bitCount; i++) {
                if (i != bits[i]) {
                    isNatural = false;
                    break;
                }
            }
        }

        if (isNatural) {
            qReg->SetQuantumState(amps);
        } else {
            for (i = 0; i < bitCount; i++) {
                qReg->M(bits[i]);
            }

            Qrack::QInterfacePtr qRegTemp = MAKE_ENGINE(bitCount, 0);
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

    virtual void multiplexer(unsigned char* bits, unsigned char ctrlCount, double* params) {
        bitCapInt componentCount = 4U * Qrack::pow2(ctrlCount);
        Qrack::complex* mtrxs = new Qrack::complex[componentCount];
        _darray_to_creal1_array(params, componentCount, mtrxs);

        qReg->UniformlyControlledSingleBit((bitLenInt*)bits, ctrlCount, (bitLenInt)bits[ctrlCount], mtrxs);

        delete[] mtrxs;
    }

    virtual void reset(unsigned char target) {
        qReg->SetBit((bitLenInt)target, false);
    }

    virtual std::vector<std::complex<double>> amplitudes() {
        std::vector<Qrack::complex> amps(qReg->GetMaxQPower());
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
        std::vector<Qrack::complex> amps(qReg->GetMaxQPower());
        qReg->GetQuantumState(&(amps[0]));

        std::vector<double> probsDouble(amps.size());
        std::transform(amps.begin(), amps.end(), probsDouble.begin(), normHelper);
        return probsDouble;
    }

    virtual unsigned long int measure(unsigned char* bits, unsigned char bitCount) {
        return (unsigned long int)qReg->M(bits, bitCount);
    }

    virtual unsigned long long measure_all() {
        return (unsigned long int)qReg->MReg(0, qReg->GetQubitCount());
    }

    virtual std::map<unsigned long int, int> measure_shots(unsigned char* bits, unsigned char bitCount, unsigned int shots) {
        bitCapInt* qPowers = new bitCapInt[bitCount];
        for (bitLenInt i = 0; i < bitCount; i++) {
            qPowers[i] = Qrack::pow2(bits[i]);
        }

        std::map<bitCapInt, int> result = qReg->MultiShotMeasureMask(qPowers, bitCount, shots);

        delete[] qPowers;

#if ENABLE_PURE32 || ENABLE_UINT128
        std::map<unsigned long int, int> resultULL;

        std::map<bitCapInt, int>::iterator it = result.begin();
        while (it != result.end())
	{
            resultULL[(unsigned long int)it->first] = it->second;
            it++;
        }

        return resultULL;
#else
        return result;
#endif
    }
};

//-------------------------------------------------------------------------
} // end namespace Simulator
//-------------------------------------------------------------------------
} // end namespace AER
//-------------------------------------------------------------------------
#endif
