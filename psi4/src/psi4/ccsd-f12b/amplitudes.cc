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

    {
        // Set <i|t|a> to zero
        for (int i = nfrzn_; i < nocc_; i++) {
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

////////////////////////////////
//* Disk Algorithm (CONV/DF) *//
////////////////////////////////

void DiskCCSDF12B::initialise_amplitudes(einsums::DiskTensor<double, 2> *t_ia, einsums::DiskTensor<double, 4> *T_ijab, 
                                         einsums::DiskTensor<double, 4> *V_ijab, einsums::DiskTensor<double, 4> *D_ijab) {

    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;


    DiskView<double, 2, 2> t_view{(*t_ia), Dim<2>{nact_, nvir_}, Count<2> {nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
    t_view.zero();

    // Set <ij|t|ab> to V_ijab (<ij|V|ab>) / (e_ii + e_jj - e_aa - e_bb)
    for (int i = 0; i < nact_; i++) {
        for (int j = i; j < nact_; j++) {
            auto V_IJ = (*V_ijab)(i, j, Range{nocc_, nobs_}, Range{nocc_, nobs_}); V_IJ.set_read_only(true);
            auto D_IJ = (*D_ijab)(i, j, All, All); D_IJ.set_read_only(true);

            //IJAB
            {
                auto T_IJ = (*T_ijab)(i, j, All, All);
                einsum(1.0, Indices{a, b}, &T_IJ.get(), 1.0, Indices{a, b}, V_IJ.get(), Indices{a, b}, D_IJ.get());
            }

            //JIAB
            if (i != j) {
                auto T_JI = (*T_ijab)(j, i, All, All);
                
                einsum(1.0, Indices{a, b}, &T_JI.get(), 1.0, Indices{a, b}, V_IJ.get(), Indices{a, b}, D_IJ.get());
            }
        }
    }
}

void DiskCCSDF12B::save_amplitudes(einsums::DiskTensor<double, 2> *t_ia_old, einsums::DiskTensor<double, 4> *T_ijab_old, 
                                   einsums::DiskTensor<double, 2> *t_ia, einsums::DiskTensor<double, 4> *T_ijab) {
    // Save the amplitudes between iterations
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    DiskView<double, 2, 2> t_ia_old_view{(*t_ia_old), Dim<2>{nact_, nvir_}, Count<2> {nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
    DiskView<double, 4, 4> T_ijab_old_view{(*T_ijab_old), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4> {nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
    DiskView<double, 2, 2> t_ia_view{(*t_ia), Dim<2>{nact_, nvir_}, Count<2> {nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
    DiskView<double, 4, 4> T_ijab_view{(*T_ijab), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4> {nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
    sort(0.0, Indices{i, a}, &t_ia_old_view.get(), 1.0, Indices{i, a}, t_ia_view.get());
    sort(0.0, Indices{i, j, a, b}, &T_ijab_old_view.get(), 1.0, Indices{i, j, a, b}, T_ijab_view.get());
}

void DiskCCSDF12B::update_t1(einsums::DiskTensor<double, 2> *t_ia, einsums::DiskTensor<double, 2> *v_ia, einsums::DiskTensor<double, 2> *D_ia) {
    // Update t_ia to be v_ia / D_ia
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    DiskView<double, 2, 2> t_ia_view{(*t_ia), Dim<2>{nact_, nvir_}, Count<2> {nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
    DiskView<double, 2, 2> v_ia_view{(*v_ia), Dim<2>{nact_, nvir_}, Count<2> {nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
    DiskView<double, 2, 2> D_ia_view{(*D_ia), Dim<2>{nact_, nvir_}, Count<2> {nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
    einsum(1.0, Indices{i, a}, &t_ia_view.get(), -1.0, Indices{i, a}, v_ia_view.get(), Indices{i, a}, D_ia_view.get());
    /*if (iteration_ < 2) {
        for (int i = 0; i < nact_; i++) {
            for (int a = 0; a < nvir_; a++) {
                outfile->Printf("t(%d, %d): %e\n", i, a, t_ia_view(i, a));
            }
            outfile->Printf("\n");
        }
    }*/
}

void DiskCCSDF12B::update_t2(einsums::DiskTensor<double, 4> *T_ijab, einsums::DiskTensor<double, 4> *V_ijab, einsums::DiskTensor<double, 4> *D_ijab) {
    // Update T_ijab to be v_ijab / D_ijab
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    for (int i = 0; i < nact_; i++) {
        for (int j = i; j < nact_; j++) {
            auto V_IJ = (*V_ijab)(i, j, All, All); V_IJ.set_read_only(true);
            auto D_IJ = (*D_ijab)(i, j, All, All); D_IJ.set_read_only(true);

            //IJAB
            {
                auto T_IJ = (*T_ijab)(i, j, All, All);
                einsum(1.0, Indices{a, b}, &T_IJ.get(), 1.0, Indices{a, b}, V_IJ.get(), Indices{a, b}, D_IJ.get());
                /*if (iteration_ < 2) {
                    for (int b = 0; b < nvir_; b++) {
                        outfile->Printf("T(%d, %d, %d, %d): %e, ", i, j, 5, b, T_IJ(5, b));
                    }
                    outfile->Printf("\n");
                }*/
            }
            //JIAB
            if (i != j) {
                auto T_JI = (*T_ijab)(j, i, All, All);
                einsum(1.0, Indices{a, b}, &T_JI.get(), 1.0, Indices{a, b}, V_IJ.get(), Indices{a, b}, D_IJ.get());
            }
        }
    }
}

void DiskCCSDF12B::form_tau(einsums::DiskTensor<double, 4> *tau, einsums::DiskTensor<double, 4> *T_ijab, einsums::DiskTensor<double, 2> *t_ia) {

    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    // Form tau_ijab_ = T_ijab + t_ia * t_jb
    for (int I = 0; I < nact_; I++) {
        for (int J = I; J < nact_; J++) {
            auto T_IJ = (*T_ijab)(I, J, All, All); T_IJ.set_read_only(true);
            auto t_I = (*t_ia)(I, All); t_I.set_read_only(true);
            auto t_J = (*t_ia)(J, All); t_J.set_read_only(true);
            //outfile->Printf("I: %d, J: %d\n", I, J);
            //outfile->Printf("T_IJ: %f\n", T_IJ(0, 0));
            //outfile->Printf("t_I: %f\n", t_I(0));
            //IJAB
            {
                auto tau_IJ = (*tau)(I, J, All, All);
                //outfile->Printf("tau_IJ: %f\n", tau_IJ(0, 0));
                sort(0.0, Indices{a, b}, &tau_IJ.get(), 1.0, Indices{a, b}, T_IJ.get());
                //outfile->Printf("tau_IJ: %f\n", tau_IJ(0, 0));
                einsum(1.0, Indices{a, b}, &tau_IJ.get(), 1.0, Indices{a}, t_I.get(), Indices{b}, t_J.get());
                //outfile->Printf("tau_IJ: %f\n", tau_IJ(0, 0));
            }

            //JIAB
            if (I != J) {
                auto tau_JI = (*tau)(J, I, All, All);
                sort(0.0, Indices{a, b}, &tau_JI.get(), 1.0, Indices{a, b}, T_IJ.get());
                einsum(1.0, Indices{a, b}, &tau_JI.get(), 1.0, Indices{a}, t_J.get(), Indices{b}, t_I.get());
            }
        }
    }
}

void DiskCCSDF12B::form_taut(einsums::DiskTensor<double, 4> *taut, einsums::DiskTensor<double, 4> *T_ijab, einsums::DiskTensor<double, 2> *t_ia) {

    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    
    // Form taut_ijab = 0.5 * T_ijab + t_ia * t_jb
    for (int I = 0; I < nact_; I++) {
        for (int J = 0; J < nact_; J++) {
            auto T_IJ = (*T_ijab)(I, J, All, All);
            auto t_I = (*t_ia)(I, All);
            auto t_J = (*t_ia)(J, All);
            //IJAB
            {
                auto taut_IJ = (*taut)(I, J, All, All);
                sort(0.0, Indices{a, b}, &taut_IJ.get(), 0.5, Indices{a, b}, T_IJ.get());
                einsum(1.0, Indices{a, b}, &taut_IJ.get(), 1.0, Indices{a}, t_I.get(), Indices{b}, t_J.get());
            }

            //JIAB
            if (I != J) {
                auto taut_JI = (*taut)(J, I, All, All);
                sort(0.0, Indices{a, b}, &taut_JI.get(), 0.5, Indices{a, b}, T_IJ.get());
                einsum(1.0, Indices{a, b}, &taut_JI.get(), 1.0, Indices{a}, t_J.get(), Indices{b}, t_I.get());
            }
        }
    }
}

double DiskCCSDF12B::get_root_mean_square_amplitude_change(einsums::DiskTensor<double, 2> *t_ia, einsums::DiskTensor<double, 4> *T_ijab, 
                                                           einsums::DiskTensor<double, 2> *t_ia_old, einsums::DiskTensor<double, 4> *T_ijab_old) {
    
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    
    // Get the root mean square amplitude change
    double rms = 0.0;
    DiskView<double, 2, 2> t_ia_view{(*t_ia), Dim<2>{nact_, nvir_}, Count<2> {nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_ia_view.set_read_only(true);
    DiskView<double, 4, 4> T_ijab_view{(*T_ijab), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4> {nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; T_ijab_view.set_read_only(true);
    DiskView<double, 2, 2> t_ia_old_view{(*t_ia_old), Dim<2>{nact_, nvir_}, Count<2> {nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_ia_old_view.set_read_only(true);
    DiskView<double, 4, 4> T_ijab_old_view{(*T_ijab_old), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4> {nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; T_ijab_old_view.set_read_only(true);
#pragma omp parallel for reduction(+:rms)
    for (int i = 0; i < nocc_; i++) {
        for (int a = nocc_; a < nobs_; a++) {
            rms += pow(t_ia_view(i, a) - t_ia_old_view(i, a), 2);
        }
    }
    for (int i = 0; i < nocc_; i++) {
        for (int j = 0; j < nocc_; j++) {
            for (int a = nocc_; a < nobs_; a++) {
                for (int b = nocc_; b < nobs_; b++) {
                    rms += pow(T_ijab_view(i, j, a, b) - T_ijab_old_view(i, j, a, b), 2);
                }
            }
        }
    }
    return sqrt(rms);

}
}} // End namespaces