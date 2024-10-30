/*
 * @BEGIN LICENSE
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2024 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

/*
Title
    A file I actually wrote myself (probably littered with problems and bad practices and bugs) 
    T_ijkl_ is taken from ccsd-f12b.cc (okay Erika C. Mitchell wrote that one!) and consolidated with the plethora of amplitude stuff
Functions
    initialise_amplitudes (I'm British, US friends imagine there's a z in there)
        Set <i|t|a> to zero and <ij|t|ab> to V_ijab (<ij|V|ab>) / (e_ii + e_jj - e_aa - e_bb) and V_ikjl (<ij|V|kl>) = T_ijkl_
    update_t1
        Updates t_ia_ to be -v_ia / D_ia
    update_t2
        Updates T_ijab to be -V_ijab / D_ijab
    form_tau
        Forms tau_ijab = T_ijab + t_ia * t_jb
    form_taut
        Forms taut_ijab = 0.5 * T_ijab + t_ia * t_jb
*/

#include "ccsd-f12b.h"

#include "einsums.hpp"

namespace psi { namespace ccsd_f12b {

void CCSDF12B::initialise_amplitudes(einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 4> *T_ijab, 
                           einsums::Tensor<double, 4> *V_ijab, einsums::Tensor<double, 4> *D_ijab) {

    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    
    // Check the size of occupied basis
    const bool use_frzn = (nfrzn_ > 0);
    const int start = (use_frzn) ? nfrzn_ : 0;

    {
        // Set <i|t|a> to zero
        for (int i = start; i < nocc_; i++) {
            for (int a = nocc_; a < nobs_; a++) {
                (*t_ia)(i, a) = 0.0;
            }
        }
    }

    // Set <ij|t|ab> to V_ijab (<ij|V|ab>) / (e_ii + e_jj - e_aa - e_bb)
    {
        einsum(1.0, Indices{i, j, a, b}, &(*T_ijab), 1.0, Indices{i, j, a, b}, *V_ijab, Indices{i, j, a, b}, *D_ijab);
    }

}

void CCSDF12B::save_amplitudes(einsums::Tensor<double, 2> *t_ia_old, einsums::Tensor<double, 4> *T_ijab_old, 
                               einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 4> *T_ijab) {
    // Save the amplitudes between iterations
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    sort(0.0, Indices{i, a}, &(*t_ia_old), 1.0, Indices{i, a}, *t_ia);
    sort(0.0, Indices{i, j, a, b}, &(*T_ijab_old), 1.0, Indices{i, j, a, b}, *T_ijab);
}

void CCSDF12B::update_t1(einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 2> *v_ia, einsums::Tensor<double, 2> *D_ia) {
    // Update t_ia to be v_ia / D_ia
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    einsum(1.0, Indices{i, a}, &(*t_ia), -1.0, Indices{i, a}, *v_ia, Indices{i, a}, *D_ia);
}

void CCSDF12B::update_t2(einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 4> *V_ijab, einsums::Tensor<double, 4> *D_ijab) {
    // Update T_ijab to be v_ijab / D_ijab
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    einsum(1.0, Indices{i, j, k, l}, &(*T_ijab), -1.0, Indices{i, j, k, l}, *V_ijab, Indices{i, j, k, l}, *D_ijab);
}

void CCSDF12B::form_tau(einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *t_ia) {

    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    // Form tau_ijab_ = T_ijab + t_ia * t_jb
    sort(1.0, Indices{i, j, a, b}, &(*tau), 1.0, Indices{i, j, a, b}, *T_ijab);
    einsum(1.0, Indices{i, j, a, b}, &(*tau), 1.0, Indices{i, a}, *t_ia, Indices{j, b}, *t_ia);
}

void CCSDF12B::form_taut(einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *t_ia) {

    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    // Form taut_ijab = 0.5 * T_ijab + t_ia * t_jb
    sort(1.0, Indices{i, j, a, b}, &(*taut), 0.5, Indices{i, j, a, b}, *T_ijab);
    einsum(1.0, Indices{i, j, a, b}, &(*taut), 1.0, Indices{i, a}, *t_ia, Indices{j, b}, *t_ia);
}

double CCSDF12B::get_root_mean_square_amplitude_change(einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 4> *T_ijab, 
                                                       einsums::Tensor<double, 2> *t_ia_old, einsums::Tensor<double, 4> *T_ijab_old) {
    // Get the root mean square amplitude change
    double rms = 0.0;
#pragma omp parallel for reduction(+:rms)
    for (int i = 0; i < nocc_; i++) {
        for (int a = nocc_; a < nobs_; a++) {
            rms += pow((*t_ia)(i, a) - (*t_ia_old)(i, a), 2);
        }
    }
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = nocc_; a < nobs_; a++) {
                for (int b = nocc_; b < nobs_; b++) {
                    rms += pow((*T_ijab)(i, j, a, b) - (*T_ijab_old)(i, j, a, b), 2);
                }
            }
        }
    }
    return sqrt(rms);

}
}} // End namespaces