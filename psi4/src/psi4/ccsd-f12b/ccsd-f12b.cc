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
Pulled from Erica C Mitchell's MP2-F12 code on 2024-09-24; last commit 6caf980, 2024-08-20

Description
Functions
    common_init
        Initializes the CCSDF12B object and common variables from the reference wavefunction and options
    print_header
    form_basissets
        Orbital (OBS) and Complimentary Auxiliary Basis Set (CABS) are formed from the reference wavefunction
        RI (resolution of the identity) space is built from the CABS of the reference wavefunction and used to build the CABS
        The new CABS is orthogonal to the OBS and covers the full RI space.
    form_D
        Forms the energy denominator D from the Fock matrix f (if p == q == r == s then D = 1/(<p|f|p>)) noting this is a orbital HF energy
    form_f12_energy
        Forms the F12 correlation energy returning the total CCSD-F12b energy summing the same and opposite spin contributions
        Calculates B - <p|f|q> <pq|X|rs> - <pq|X|rs> <p|f|q> and adds contributions of a multiple of B Tilde and V Tilde 
    form_cabs_singles
        Forms the CABS Singles correction to the energy from the C and f matrices as
        E_singles = sum_{a} sum_{i} |f^{a'}_{i}|^2 / (e_{a} - e_{i})
    V_Tilde
        Calculates the einsum of G_ij and fock energy (D_ij) and the result with C to generate V_Tilde by deducting from V_ij
        Returns the same and opposite spin contributions to the energy from the Tilde V matrix by multiplying with t (amplitude)
    B_Tilde
        C is einsummed with D into the occupied basis and the result einsummed again with C is taken away from B to obtain B_Tilde
        This is then multiplied with t_ to obtain the same and opposite spin contributions
    t_
        Is a constant - returning 1/2, 3/8 or 1/8  depending on indice symmetry
    compute_energy 
        The control centre of the operation, calling the other functions to form the energy
        form_basissets is first called, then fock and exchange matrices are formed (form_fock or form_df_fock)
        In the DF case, J_inv_AB is formed by calling form_metric_ints (is actually J^-1_AB Apq)
        Then in each case V, X, C, B, D, and G are formed and the energy is formed and if a CABS singles correction is requested it is added                                                                                                                
*/

#include "ccsd-f12b.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "psi4/libmints/basisset.h"
#include "psi4/libmints/dimension.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"

#include "einsums.hpp"

namespace psi { namespace ccsd_f12b {

CCSDF12B::CCSDF12B(SharedWavefunction ref_wfn, Options& options):
    Wavefunction(options) {
    shallow_copy(ref_wfn);
    reference_wavefunction_ = ref_wfn;

    common_init();
}

CCSDF12B::~CCSDF12B() {}

void CCSDF12B::common_init() 
{
    if (options_.get_str("REFERENCE") != "RHF") {
        throw PsiException("Only a restricted reference may be used",__FILE__,__LINE__);
    }

    options_.set_global_str("SCREENING", "NONE");
    
    print_ = options_.get_int("PRINT");
    singles_ = options_.get_bool("CABS_SINGLES");
    memory_ = Process::environment.get_memory();
    double_memory_ = sizeof(double);

    f12_type_ = options_.get_str("F12_TYPE");
    f12_read_ints_ = options_.get_bool("F12_READ_INTS");

    rms_tol_ = options_.get_double("R_CONVERGENCE");
    e_tol_ = options_.get_double("E_CONVERGENCE");
    maxiter_ = options_.get_int("MAXITER");

    std::vector<OrbitalSpace> bs_ = {};
    nobs_ = reference_wavefunction_->basisset()->nbf();
    nocc_ = reference_wavefunction_->doccpi()[0];
    nvir_ = nobs_ - nocc_;
    nfrzn_ = reference_wavefunction_->frzcpi()[0];
    nact_ = nocc_ - nfrzn_;
    naux_ = 0;

    if (f12_type_.find("DF") != std::string::npos) {
        use_df_ = true;
        DFBS_ = reference_wavefunction_->get_basisset("DF_BASIS_MP2");
        naux_ = DFBS_->nbf();
    }

    beta_ = options_.get_double("F12_BETA");
    cgtg_ = reference_wavefunction_->mintshelper()->f12_cgtg(beta_);

    nthreads_ = 1;
#ifdef _OPENMP
    nthreads_ = Process::environment.get_n_threads();
#endif
}

void CCSDF12B::print_header()
{
    outfile->Printf("\n -----------------------------------------------------------\n");
    if (use_df_) {
        outfile->Printf("                        DF-CCSD-F12b                        \n");
        outfile->Printf("             Density-Fitted Explicitly Correlated           \n");
        outfile->Printf("            Coupled Cluster Singles-Doubles Theory          \n");
        outfile->Printf("                CCSD Wavefunction, %2d Threads              \n\n", nthreads_);
        outfile->Printf("                        Nathan Harmer                       \n");
        outfile->Printf("                 adapted from Erica Mitchell                \n"); 
    } else {
        outfile->Printf("                          CCSD-F12b                         \n");
        outfile->Printf("                    Explicitly Correlated                   \n");
        outfile->Printf("            Coupled Cluster Singles-Doubles Theory          \n");
        outfile->Printf("                CCSD Wavefunction, %2d Threads              \n\n", nthreads_);
        outfile->Printf("                        Nathan Harmer                       \n");
        outfile->Printf("                  adapted from Erica Mitchell               \n");
    }
    outfile->Printf(" -----------------------------------------------------------\n\n");
    outfile->Printf(" Using %s algorithm \n\n", f12_type_.c_str());
}

void CCSDF12B::form_basissets()
{
    outfile->Printf(" ===> Forming the OBS and CABS <===\n\n");

    outfile->Printf("  Orbital Basis Set (OBS)\n");
    OrbitalSpace OBS = reference_wavefunction_->alpha_orbital_space("p", "SO", "ALL");
    OBS.basisset()->print();

    /* Known error, running MP2-F12 then CCSD-F12B causes some CABS confusion? */
    outfile->Printf("  Complementary Auxiliary Basis Set (CABS)\n");
    OrbitalSpace RI = OrbitalSpace::build_ri_space(reference_wavefunction_->get_basisset("CABS"), 1.0e-8);
    OrbitalSpace CABS = OrbitalSpace::build_cabs_space(OBS, RI, 1.0e-6);
    CABS.basisset()->print();
    
    if (use_df_) {
        outfile->Printf("  Auxiliary Basis Set\n");
        DFBS_->print();
    }

    nri_ = CABS.dim().max() + nobs_;
    ncabs_ = nri_ - nobs_;

    if (nfrzn_ != 0) {
        outfile->Printf("  Frozen Core Orbitals: %3d \n\n", nfrzn_);
    }

    outfile->Printf("  ----------------------------------------\n");
    outfile->Printf("     %5s  %5s   %5s  %5s  %5s   \n", "NOCC", "NOBS", "NCABS", "NRI", "NAUX");
    outfile->Printf("  ----------------------------------------\n");
    outfile->Printf("     %5d  %5d   %5d  %5d  %5d   \n", nocc_, nobs_, ncabs_, nri_, naux_);
    outfile->Printf("  ----------------------------------------\n");

    bs_ = {OBS, CABS};
}

void CCSDF12B::form_D_ijab(einsums::Tensor<double, 4> *D, einsums::Tensor<double, 2> *f)
{
    using namespace einsums;

#pragma omp parallel for collapse(4) num_threads(nthreads_)
    for (size_t i = nfrzn_; i < nocc_; i++) {
        for (size_t j = nfrzn_; j < nocc_; j++) {
            for (size_t a = nocc_; a < nobs_; a++) {
                for (size_t b = nocc_; b < nobs_; b++) {
                    auto denom = (*f)(a, a) + (*f)(b, b) - (*f)(i, i) - (*f)(j, j);
                    (*D)(i - nfrzn_, j - nfrzn_, a - nocc_, b - nocc_) = (1 / denom);
                }
            }
        }
    }
}

void CCSDF12B::form_D_ia(einsums::Tensor<double, 2> *D, einsums::Tensor<double, 2> *f)
{
    using namespace einsums;

#pragma omp parallel for collapse(2) num_threads(nthreads_)
    for (size_t i = nfrzn_; i < nocc_; i++) {
        for (size_t a = nocc_; a < nobs_; a++) {
            auto denom = (*f)(a, a) - (*f)(i, i);
            (*D)(i - nfrzn_, a - nocc_) = (1 / denom);
        }
    }
}

void CCSDF12B::form_f12_energy(einsums::Tensor<double,4> *V, einsums::Tensor<double,4> *X,
                             einsums::Tensor<double,4> *C, einsums::Tensor<double,4> *B,
                             einsums::Tensor<double,2> *f, einsums::Tensor<double,4> *G,
                             einsums::Tensor<double,4> *D)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto E_f12_s = 0.0;
    auto E_f12_t = 0.0;
    int kd;

    outfile->Printf("  \n");
    outfile->Printf("  %1s   %1s  |     %14s     %14s     %12s \n",
                    "i", "j", "E_F12(Singlet)", "E_F12(Triplet)", "E_F12");
    outfile->Printf(" ----------------------------------------------------------------------\n");
    for (size_t i = 0; i < nact_; i++) {
        for (size_t j = i; j < nact_; j++) {
            // Allocations
            Tensor B_ = (*B)(All, All, All, All);
            {
                Tensor X_ = (*X)(All, All, All, All);
                auto f_scale = (*f)(i + nfrzn_, i + nfrzn_) + (*f)(j + nfrzn_, j + nfrzn_);
                linear_algebra::scale(f_scale, &X_);
                sort(1.0, Indices{k, l, m, n}, &B_, -1.0, Indices{k, l, m, n}, X_);
            }

            // Getting V_Tilde and B_Tilde
            Tensor V_ = TensorView<double, 2>{(*V), Dim<2>{nact_, nact_}, Offset<4>{i, j, 0, 0},
                                            Stride<2>{(*V).stride(2), (*V).stride(3)}};
            auto G_ = TensorView<double, 2>{(*G), Dim<2>{nvir_, nvir_}, Offset<4>{i, j, 0, 0},
                                            Stride<2>{(*G).stride(2), (*G).stride(3)}};
            auto D_ = TensorView<double, 2>{(*D), Dim<2>{nvir_, nvir_}, Offset<4>{i, j, 0, 0},
                                            Stride<2>{(*D).stride(2), (*D).stride(3)}};
            auto VT = V_Tilde(V_, C, G_, D_, i, j);
            auto BT = B_Tilde(B_, C, D_, i, j);

            // Computing the energy
            ( i == j ) ? ( kd = 1 ) : ( kd = 2 );
            auto E_s = kd * (2 * VT.first + BT.first);
            E_f12_s += E_s;
            auto E_t = 0.0;
            if ( i != j ) {
                E_t = 3.0 * kd * (2 * VT.second + BT.second);
                E_f12_t += E_t;
            }
            auto E_f = E_s + E_t;
            outfile->Printf("%3d %3d  |   %16.12f   %16.12f     %16.12f \n", i+nfrzn_+1, j+nfrzn_+1, E_s, E_t, E_f);
        }
    }

    set_scalar_variable("F12 OPPOSITE-SPIN CORRELATION ENERGY", E_f12_s);
    set_scalar_variable("F12 SAME-SPIN CORRELATION ENERGY", E_f12_t);

    E_f12_ = E_f12_s + E_f12_t;
}

void CCSDF12B::form_cabs_singles(einsums::Tensor<double,2> *f)
{
    using namespace einsums;
    using namespace linear_algebra;

    int all_vir = nvir_ + ncabs_;

    // Diagonalize f_ij and f_AB
    Tensor<double, 2> C_ij{"occupied e-vecs", nocc_, nocc_};
    Tensor<double, 2> C_AB{"vir and CABS e-vecs", all_vir, all_vir};

    Tensor<double, 1> e_ij{"occupied e-vals", nocc_};
    Tensor<double, 1> e_AB{"vir and CABS e-vals", all_vir};
    {
        C_ij = (*f)(Range{0, nocc_}, Range{0, nocc_});
        C_AB = (*f)(Range{nocc_, nri_}, Range{nocc_, nri_});

        syev(&C_ij, &e_ij);
        syev(&C_AB, &e_AB);
    }

    // Form f_iA
    Tensor<double, 2> f_iA{"Fock Occ-All_vir", nocc_, all_vir};
    {
        Tensor f_view = (*f)(Range{0, nocc_}, Range{nocc_, nri_});

        gemm<false, false>(1.0, C_ij, gemm<false, true>(1.0, f_view, C_AB), 0.0, &f_iA);
    } // f_view 0-occ, nocc-nri ; C_AB nocc-nri, nocc-nri ; C_ij 0-occ, 0-occ

    double E_s = 0.0;
#pragma omp parallel for collapse(2) num_threads(nthreads_) reduction(+:E_s)
    for (size_t A = 0; A < all_vir; A++) {
        for (size_t i = 0; i < nocc_; i++) {
            E_s += 2 * pow(f_iA(i, A), 2) / (e_ij(i) - e_AB(A));
        }
    }

    E_singles_ = E_s;
}

double CCSDF12B::compute_energy()
{
    timer_on("CCSD-F12b Compute Energy");
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    timer::initialize();

    print_header();

    /* Form the orbital spaces */
    timer_on("OBS and CABS");
    form_basissets();
    timer_off("OBS and CABS");

    outfile->Printf("\n ===> Forming the Integrals <===");
    outfile->Printf("\n No screening will be used to compute integrals\n");

    /* Form the Fock Matrix */
    outfile->Printf("   Fock Matrix\n");
    auto f = std::make_unique<Tensor<double, 2>>("Fock Matrix", nri_, nri_);
    auto k = std::make_unique<Tensor<double, 2>>("Exchange MO Integral", nri_, nri_);
    timer_on("Fock Matrix");
    if (use_df_) {
        form_df_fock(f.get(), k.get());
    } else {
        form_fock(f.get(), k.get());
    }
    timer_off("Fock Matrix");

    /* Form the F12 Intermediates */
    outfile->Printf("\n ===> Forming the F12 Intermediate Tensors <===\n");
    auto D_Werner = std::make_unique<Tensor<double, 4>>("D (from Werner 1992) Intermediate Tensor", nact_, nact_, nvir_, nvir_);
    auto beta = std::make_unique<Tensor<double, 2>>("Constant matrix beta", nact_, nact_);
    auto A = std::make_unique<Tensor<double, 2>>("A Intermediate Tensor", nvir_, nvir_);
    auto s = std::make_unique<Tensor<double, 2>>("s Intermediate Tensor", nact_, nvir_);
    auto r = std::make_unique<Tensor<double, 2>>("r Intermediate Tensor", nact_, nvir_);
    auto v_ia = std::make_unique<Tensor<double, 2>>("T1 Residual (v) Tensor", nact_, nvir_);
    auto X_Werner = std::make_unique<Tensor<double, 2>>("X (from Werner 1992) Intermediate Tensor", nvir_, nvir_);
    auto Y_Werner = std::make_unique<Tensor<double, 4>>("Y (from Werner 1992) Intermediate Tensor", nact_, nact_, nvir_, nvir_);
    auto Z_Werner = std::make_unique<Tensor<double, 4>>("Z (from Werner 1992) Intermediate Tensor", nact_, nact_, nvir_, nvir_);
    auto alpha = std::make_unique<Tensor<double, 4>>("Constant matrix alpha", nact_, nact_, nact_, nact_);
    auto G_ij = std::make_unique<Tensor<double, 4>>("G Intermediate Tensor", nact_, nact_, nvir_, nvir_);
    auto V_ijab = std::make_unique<Tensor<double, 4>>("T2 Residual (V) Tensor", nact_, nact_, nvir_, nvir_);
    auto t_ia = std::make_unique<Tensor<double, 2>>("T1 Amplitude Tensor", nact_, nvir_);
    auto T_ijab = std::make_unique<Tensor<double, 4>>("T2 Amplitude Tensor", nact_, nact_, nvir_, nvir_);
    auto tau = std::make_unique<Tensor<double, 4>>("(T_ijab + t_ia * t_jb) Tensor", nact_, nact_, nvir_, nvir_);
    auto taut = std::make_unique<Tensor<double, 4>>("(0.5 T_ijab + t_ia * t_jb) Tensor", nact_, nact_, nvir_, nvir_);
    auto V = std::make_unique<Tensor<double, 4>>("V Intermediate Tensor", nact_, nact_, nact_, nact_);
    auto X = std::make_unique<Tensor<double, 4>>("X Intermediate Tensor", nact_, nact_, nact_, nact_);
    auto C = std::make_unique<Tensor<double, 4>>("C Intermediate Tensor", nact_, nact_, nvir_, nvir_);
    auto B = std::make_unique<Tensor<double, 4>>("B Intermediate Tensor", nact_, nact_, nact_, nact_);
    auto D_ijab = std::make_unique<Tensor<double, 4>>("D Tensor", nact_, nact_, nvir_, nvir_);
    auto D_ia = std::make_unique<Tensor<double, 2>>("D Tensor", nact_, nvir_);
    auto G_ijab = std::make_unique<Tensor<double, 4>>("ERI <ij|ab>", nact_, nact_, nvir_, nvir_);
    
    /* Initialise amplitudes */
    timer_on("Initialise Amplitudes");
    timer_on("Energy Denom");
    form_D_ijab(D_ijab.get(), f.get());
    form_D_ia(D_ia.get(), f.get());
    timer_off("Energy Denom");

    auto G = Tensor<double, 4>{"MO G Tensor", nact_, nact_, nobs_, nobs_};
    timer_on("ERI <ij|ab>");
    if (use_df_) {
        outfile->Printf("   [J_AB]^(-1)\n");
        auto J_inv_AB = std::make_unique<Tensor<double, 3>>("Metric MO ([J_AB]^{-1})", naux_, nobs_, nri_);
        timer_on("Metric Integrals");
        form_metric_ints(J_inv_AB.get(), false);
        timer_off("Metric Integrals");
        form_df_teints("G", &G, J_inv_AB.get(), {'o', 'O', 'o', 'O'});
        //J_inv_AB.reset();
    } else {
        form_teints("G", &G, {'o', 'O', 'o', 'O'});
    }
    (*G_ijab) = G(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
    timer_off("ERI <ij|ab>");

    initialise_amplitudes(t_ia.get(), T_ijab.get(), G_ijab.get(), D_ijab.get());
    G_ijab.reset();
    form_tau(tau.get(), T_ijab.get(), t_ia.get());
    form_taut(taut.get(), T_ijab.get(), t_ia.get());
    iteration_ = 0;
    double rms = 1.0;
    double e_diff = 1.0;
    timer_off("Initialise Amplitudes");

    /* Iterate to get a CCSD energy */
    while (iteration_ < maxiter_ && (rms > rms_tol_ || e_diff > e_tol_)) {
        timer_on("CCSD Iteration");
        /* Temp Store old amplitudes and energy */
        timer_on("Save Amplitudes");
        auto t_ia_old = std::make_unique<Tensor<double, 2>>("Old T1 Amplitude Tensor", nact_, nvir_);
        auto T_ijab_old = std::make_unique<Tensor<double, 4>>("Old T2 Amplitude Tensor", nact_, nact_, nvir_, nvir_);
        save_amplitudes(t_ia_old.get(), T_ijab_old.get(), t_ia.get(), T_ijab.get());
        auto E_old_ = E_ccsd_ + E_f12b_;
        timer_off("Save Amplitudes");

        outfile->Printf("\n ===> CCSD Iteration %d <===\n", iteration_);
        if (use_df_) {
            outfile->Printf("   [J_AB]^(-1)\n");
            auto J_inv_AB = std::make_unique<Tensor<double, 3>>("Metric MO ([J_AB]^{-1})", naux_, nobs_, nri_);
            timer_on("Metric Integrals");
            form_metric_ints(J_inv_AB.get(), false);
            timer_off("Metric Integrals");

            outfile->Printf("   V Intermediate\n");
            outfile->Printf("   X Intermediate\n");
            timer_on("V and X Intermediate");
            form_df_V_X(V.get(), X.get(), J_inv_AB.get());
            timer_off("V and X Intermediate");
            X.reset();

            /*
            outfile->Printf("   B Intermediate\n");
            timer_on("B Intermediate");
            form_df_B(B.get(), f.get(), k.get(), J_inv_AB.get());
            k.reset();
            timer_off("B Intermediate");
            */

            outfile->Printf("   D (from Werner 1992) Intermediate\n");
            timer_on("D (from Werner 1992) Intermediate");
            form_D_Werner(D_Werner.get(), tau.get(), t_ia.get());
            timer_off("D (from Werner 1992) Intermediate");

            outfile->Printf("   Constant matrix beta\n");
            timer_on("Constant matrix beta");
            form_df_beta(beta.get(), f.get(), t_ia.get(), tau.get(), J_inv_AB.get());
            timer_off("Constant matrix beta");

            outfile->Printf("   A Intermediate\n");
            timer_on("A Intermediate");
            form_df_A(A.get(), J_inv_AB.get(), T_ijab.get());
            timer_off("A Intermediate");

            outfile->Printf("   s Intermediate\n");
            timer_on("s Intermediate");
            form_df_s(s.get(), T_ijab.get(), f.get(), t_ia.get(), A.get(), D_Werner.get(), J_inv_AB.get());
            timer_off("s Intermediate");

            outfile->Printf("   r Intermediate\n");
            timer_on("r Intermediate");
            form_df_r(r.get(), f.get(), t_ia.get(), J_inv_AB.get());
            timer_off("r Intermediate");

            outfile->Printf("   T1 Residual (v) Tensor\n");
            timer_on("T1 Residual (v) Tensor");
            form_v_ia(v_ia.get(), T_ijab.get(), t_ia.get(), beta.get(), r.get(), s.get());
            timer_off("T1 Residual (v) Tensor");

            outfile->Printf("   Update T1\n");
            timer_on("Update T1");
            update_t1(t_ia.get(), v_ia.get(), D_ia.get());
            D_Werner.reset();
            beta.reset();
            r.reset();
            s.reset();
            A.reset();
            timer_off("Update T1");

            outfile->Printf("   D (from Werner 1992) Intermediate\n");
            timer_on("D (from Werner 1992) Intermediate");
            form_D_Werner(D_Werner.get(), tau.get(), t_ia.get());
            timer_off("D (from Werner 1992) Intermediate");

            outfile->Printf("   Constant matrix beta\n");
            timer_on("Constant matrix beta");
            form_df_beta(beta.get(), f.get(), t_ia.get(), tau.get(), J_inv_AB.get());
            timer_off("Constant matrix beta");

            outfile->Printf("   A Intermediate\n");
            timer_on("A Intermediate");
            form_df_A(A.get(), J_inv_AB.get(), T_ijab.get());
            timer_off("A Intermediate");

            outfile->Printf("   s Intermediate\n");
            timer_on("s Intermediate");
            form_df_s(s.get(), T_ijab.get(), f.get(), t_ia.get(), A.get(), D_Werner.get(), J_inv_AB.get());
            timer_off("s Intermediate");

            outfile->Printf("   r Intermediate\n");
            timer_on("r Intermediate");
            form_df_r(r.get(), f.get(), t_ia.get(), J_inv_AB.get());
            timer_off("r Intermediate");

            outfile->Printf("   X (from Werner 1992) Intermediate\n");
            timer_on("X (from Werner 1992) Intermediate");
            form_df_X_Werner(X_Werner.get(), f.get(), t_ia.get(), A.get(), r.get(), J_inv_AB.get());
            timer_off("X (from Werner 1992) Intermediate");

            outfile->Printf("   Y (from Werner 1992) Intermediate\n");
            timer_on("Y (from Werner 1992) Intermediate");
            form_df_Y_Werner(Y_Werner.get(), taut.get(), f.get(), t_ia.get(), J_inv_AB.get());
            timer_off("Y (from Werner 1992) Intermediate");

            outfile->Printf("   Z (from Werner 1992) Intermediate\n");
            timer_on("Z (from Werner 1992) Intermediate");
            form_df_Z_Werner(Z_Werner.get(), taut.get(), t_ia.get(), J_inv_AB.get());
            timer_off("Z (from Werner 1992) Intermediate");

            outfile->Printf("   Constant matrix alpha\n");
            timer_on("Constant matrix alpha");
            form_df_alpha(alpha.get(), tau.get(), t_ia.get(), J_inv_AB.get());
            timer_off("Constant matrix alpha");

            outfile->Printf("   C Intermediate\n");
            timer_on("C Intermediate");
            form_df_C(C.get(), f.get(), J_inv_AB.get());
            timer_off("C Intermediate");

            outfile->Printf("   G Intermediate\n");
            timer_on("G Intermediate");
            form_df_G(G_ij.get(), s.get(), t_ia.get(), D_Werner.get(), alpha.get(), beta.get(), T_ijab.get(), X_Werner.get(), Y_Werner.get(), Z_Werner.get(), tau.get(), J_inv_AB.get());
            timer_off("G Intermediate");

            outfile->Printf("   T2 Residual (V) Tensor\n");
            timer_on("T2 Residual (V) Tensor");
            form_df_V_ijab(V_ijab.get(), G_ij.get(), tau.get(), D_Werner.get(), alpha.get(), C.get(), J_inv_AB.get());
            timer_off("T2 Residual (V) Tensor");

            outfile->Printf("   Update T2\n");
            timer_on("Update T2");
            update_t2(T_ijab.get(), V_ijab.get(), D_ijab.get());
            timer_off("Update T2");

            auto G = Tensor<double, 4>{"MO G Tensor", nact_, nact_, nobs_, nobs_};
            timer_on("ERI <ij|ab>");
            form_df_teints("G", &G, J_inv_AB.get(), {'o', 'O', 'o', 'O'});
            (*G_ijab) = G(Range{0, nact_}, Range{0, nact_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
            timer_off("ERI <ij|ab>");

            outfile->Printf("\n ===> Computing CCSD-F12b Extra Energy Correction <===\n");
            timer_on("F12b Energy Correction");
            form_CCSDF12B_Energy_df(tau.get(), t_ia.get(), J_inv_AB.get());
            timer_off("F12 Energy Correction");
        } else {
            outfile->Printf("   V Intermediate\n");
            outfile->Printf("   X Intermediate\n");
            timer_on("V and X Intermediate");
            form_V_X(V.get(), X.get());
            timer_off("V and X Intermediate");

            /*
            outfile->Printf("   B Intermediate\n");
            timer_on("B Intermediate");
            form_B(B.get(), f.get(), k.get());
            k.reset();
            timer_off("B Intermediate");
            */

            outfile->Printf("   D (from Werner 1992) Intermediate\n");
            timer_on("D (from Werner 1992) Intermediate");
            form_D_Werner(D_Werner.get(), tau.get(), t_ia.get());
            timer_off("D (from Werner 1992) Intermediate");

            outfile->Printf("   Constant matrix beta\n");
            timer_on("Constant matrix beta");
            form_beta(beta.get(), f.get(), t_ia.get(), tau.get());
            timer_off("Constant matrix beta");

            outfile->Printf("   A Intermediate\n");
            timer_on("A Intermediate");
            form_A(A.get(), T_ijab.get());
            timer_off("A Intermediate");

            outfile->Printf("   s Intermediate\n");
            timer_on("s Intermediate");
            form_s(s.get(), T_ijab.get(), f.get(), t_ia.get(), A.get(), D_Werner.get());
            timer_off("s Intermediate");

            outfile->Printf("   r Intermediate\n");
            timer_on("r Intermediate");
            form_r(r.get(), f.get(), t_ia.get());
            timer_off("r Intermediate");

            outfile->Printf("   T1 Residual (v) Tensor\n");
            timer_on("T1 Residual (v) Tensor");
            form_v_ia(v_ia.get(), T_ijab.get(), t_ia.get(), beta.get(), r.get(), s.get());
            timer_off("T1 Residual (v) Tensor");

            outfile->Printf("   Update T1\n");
            timer_on("Update T1");
            update_t1(t_ia.get(), v_ia.get(), D_ia.get());
            D_Werner.reset();
            beta.reset();
            r.reset();
            s.reset();
            A.reset();
            timer_off("Update T1");

            outfile->Printf("   D (from Werner 1992) Intermediate\n");
            timer_on("D (from Werner 1992) Intermediate");
            form_D_Werner(D_Werner.get(), tau.get(), t_ia.get());
            timer_off("D (from Werner 1992) Intermediate");

            outfile->Printf("   Constant matrix beta\n");
            timer_on("Constant matrix beta");
            form_beta(beta.get(), f.get(), t_ia.get(), tau.get());
            timer_off("Constant matrix beta");

            outfile->Printf("   A Intermediate\n");
            timer_on("A Intermediate");
            form_A(A.get(), T_ijab.get());
            timer_off("A Intermediate");

            outfile->Printf("   s Intermediate\n");
            timer_on("s Intermediate");
            form_s(s.get(), T_ijab.get(), f.get(), t_ia.get(), A.get(), D_Werner.get());
            timer_off("s Intermediate");

            outfile->Printf("   r Intermediate\n");
            timer_on("r Intermediate");
            form_r(r.get(), f.get(), t_ia.get());
            timer_off("r Intermediate");

            outfile->Printf("   X (from Werner 1992) Intermediate\n");
            timer_on("X (from Werner 1992) Intermediate");
            form_X_Werner(X_Werner.get(), f.get(), t_ia.get(), A.get(), r.get());
            timer_off("X (from Werner 1992) Intermediate");

            outfile->Printf("   Y (from Werner 1992) Intermediate\n");
            timer_on("Y (from Werner 1992) Intermediate");
            form_Y_Werner(Y_Werner.get(), taut.get(), f.get(), t_ia.get());
            timer_off("Y (from Werner 1992) Intermediate");

            outfile->Printf("   Z (from Werner 1992) Intermediate\n");
            timer_on("Z (from Werner 1992) Intermediate");
            form_Z_Werner(Z_Werner.get(), taut.get(), t_ia.get());
            timer_off("Z (from Werner 1992) Intermediate");

            outfile->Printf("   Constant matrix alpha\n");
            timer_on("Constant matrix alpha");
            form_alpha(alpha.get(), tau.get(), t_ia.get());
            timer_off("Constant matrix alpha");

            outfile->Printf("   G Intermediate\n");
            timer_on("G Intermediate");
            form_G(G_ij.get(), s.get(), t_ia.get(), D_Werner.get(), alpha.get(), beta.get(), T_ijab.get(), X_Werner.get(), Y_Werner.get(), Z_Werner.get(), tau.get());
            timer_off("G Intermediate");

            outfile->Printf("   C Intermediate\n");
            timer_on("C Intermediate");
            form_C(C.get(), f.get());
            timer_off("C Intermediate");

            outfile->Printf("   T2 Residual (V) Tensor\n");
            timer_on("T2 Residual (V) Tensor");
            form_V_ijab(V_ijab.get(), G_ij.get(), tau.get(), D_Werner.get(), alpha.get(), C.get());
            timer_off("T2 Residual (V) Tensor");

            outfile->Printf("   Update T2\n");
            timer_on("Update T2");
            update_t2(T_ijab.get(), V_ijab.get(), D_ijab.get());
            timer_off("Update T2");

            outfile->Printf("   Update Tau and Tau Tilde\n");
            timer_on("Update Tau and Tau Tilde");
            form_tau(tau.get(), T_ijab.get(), t_ia.get());
            form_taut(taut.get(), T_ijab.get(), t_ia.get());
            timer_off("Update Tau and Tau Tilde");

            auto G = Tensor<double, 4>{"MO G Tensor", nact_, nact_, nobs_, nobs_};
            timer_on("ERI <ij|ab>");
            form_teints("G", &G, {'o', 'O', 'o', 'O'});
            (*G_ijab) = G(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
            timer_off("ERI <ij|ab>");

            outfile->Printf("\n ===> Computing CCSD-F12b Extra Energy Correction <===\n");
            timer_on("F12b Energy Correction");
            form_CCSDF12B_Energy(tau.get(), t_ia.get());
            timer_off("F12 Energy Correction");
        }

        /* Compute the CCSD-F12b Energy */
        outfile->Printf("\n ===> Computing CCSD-F12b Energy Correction <===\n");
        timer_on("F12b Energy Correction");
        form_CCSD_Energy(G_ijab.get(), V.get(), tau.get());
        timer_off("F12 Energy Correction");
        G_ijab.reset();
        V.reset();
        tau.reset();

        /* Compare the CCSD energies */
        timer_on("CCSD Energy Comparison");
        e_diff = std::abs(E_ccsd_ + E_f12b_ - E_old_);
        rms = get_root_mean_square_amplitude_change(t_ia.get(), T_ijab.get(), t_ia_old.get(), T_ijab_old.get());

        iteration_++;
        timer_off("CCSD Iteration");
    }

    if (singles_ == true) {
        timer_on("CABS Singles Correction");
        form_cabs_singles(f.get());
        timer_off("CABS Singles Correction");
    }

    print_results();

    if (print_ > 1) {
        timer::report();
    }
    timer::finalize();
    timer_off("CCSD-F12b Compute Energy");

    // Typically you would build a new wavefunction and populate it with data
    return E_ccsdf12b_;
}

void CCSDF12B::print_results()
{
    if (use_df_) {
        outfile->Printf("\n ===> DF-CCSD-F12b Energies <===\n\n");
    } else {
        outfile->Printf("\n ===> CCSD-F12b Energies <===\n\n");
    }

    auto E_rhf = Process::environment.globals["CURRENT REFERENCE ENERGY"];
    //auto E_mp2 = Process::environment.globals["MP2 CORRELATION ENERGY"];

    E_ccsdf12b_ = E_rhf + E_ccsd_ + E_f12b_; // + E_mp2 + E_singles_ (not needed in ccsdf12b)

    if (use_df_) {
        outfile->Printf("  Total DF-CCSD-F12b Energy:      %16.12f \n", E_ccsdf12b_);
    } else {
        outfile->Printf("  Total CCSD-F12b Energy:         %16.12f \n", E_ccsdf12b_);
    }
    outfile->Printf("     RHF Reference Energy:              %16.12f \n", E_rhf);
    outfile->Printf("     CCSD Correlation Energy:           %16.12f \n", E_ccsd_);
    outfile->Printf("     F12b Correlation Energy:           %16.12f \n", E_f12b_);

    if (singles_ == true) {
        outfile->Printf("     CABS Singles Correction:           %16.12f \n", E_singles_);
    }

    outfile->Printf("\n  Number of Iterations:               %d\n", iteration_);

    set_scalar_variable("F12B CORRELATION ENERGY", E_f12b_ + E_singles_);
    set_scalar_variable("CCSD-F12B CORRELATION ENERGY", E_ccsd_ + E_f12b_ + E_singles_);
    set_scalar_variable("CCSD-F12B TOTAL ENERGY", E_ccsdf12b_);

    set_scalar_variable("F12B SINGLES ENERGY", E_singles_);
    set_scalar_variable("F12B DOUBLES ENERGY", E_f12b_);
}

double CCSDF12B::T_ijkl(const int& p, const int& q, const int& r, const int& s)
{
    if (p == q && p == r && p == s) {
        return 0.5;
    } else if (p == r && q == s) {
        return 0.375;
    } else if (q == r && p == s) {
        return 0.125;
    } else {
        return 0.0;
    }
}

std::pair<double, double> CCSDF12B::V_Tilde(einsums::Tensor<double, 2>& V_ij, einsums::Tensor<double, 4> *C,
                                            einsums::TensorView<double, 2>& G_ij, einsums::TensorView<double, 2>& D_ij,
                                            const int& i, const int& j)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    double V_s, V_t;
    int kd;

    {
        Tensor<double, 2> GD{"G_ijab . D_ijab", nvir_, nvir_};
        einsum(Indices{a, b}, &GD, Indices{a, b}, G_ij, Indices{a, b}, D_ij);
        einsum(1.0, Indices{k, l}, &V_ij, -1.0, Indices{k, l, a, b}, *C, Indices{a, b}, GD);
    }

    ( i == j ) ? ( kd = 1 ) : ( kd = 2 );

    V_s += 0.25 * (T_ijkl(i, j, i, j) + T_ijkl(i, j, j, i)) * kd * (V_ij(i, j) + V_ij(j, i));

    if ( i != j ) {
        V_t += 0.25 * (T_ijkl(i, j, i, j) - T_ijkl(i, j, j, i)) * kd * (V_ij(i, j) - V_ij(j, i));
    }
    return {V_s, V_t};
}

std::pair<double, double> CCSDF12B::B_Tilde(einsums::Tensor<double, 4>& B_ij, einsums::Tensor<double, 4> *C,
                                            einsums::TensorView<double, 2>& D_ij, 
                                            const int& i, const int& j)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    double B_s, B_t;
    int kd;

    {
        Tensor<double, 4> CD{"C_klab . D_ijab", nact_, nact_, nvir_, nvir_};
        einsum(Indices{k, l, a, b}, &CD, Indices{k, l, a, b}, *C, Indices{a, b}, D_ij);
        einsum(1.0, Indices{k, l, m, n}, &B_ij, -1.0, Indices{m, n, a, b}, *C,
                                                      Indices{k, l, a, b}, CD);
    }

    ( i == j ) ? ( kd = 1 ) : ( kd = 2 );

    B_s += 0.125 * (T_ijkl(i, j, i, j) + T_ijkl(i, j, j, i)) * kd 
                 * (B_ij(i, j, i, j) + B_ij(i, j, j, i))
                 * (T_ijkl(i, j, i, j) + T_ijkl(i, j, j, i)) * kd;

    if ( i != j ) {
        B_t += 0.125 * (T_ijkl(i, j, i, j) - T_ijkl(i, j, j, i)) * kd
                     * (B_ij(i, j, i, j) - B_ij(i, j, j, i))
                     * (T_ijkl(i, j, i, j) - T_ijkl(i, j, j, i)) * kd;
    }
    return {B_s, B_t};
}

void CCSDF12B::form_CCSDF12B_Energy(einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 2> *t_ia)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    E_f12b_ = 0.0;

    auto D_ijab = std::make_unique<Tensor<double, 4>>("D_ijab", nact_, nact_, nvir_, nvir_); // D_ijab assumed as ab used in eq 9 and 10 of Alder 2007
    Tensor<double, 4> W_pqkl{"W_pqkl", nvir_, nvir_, nact_, nact_};
    Tensor<double, 4> K_pqrs{"K_pqrs", nvir_, nvir_, nobs_, nobs_};
    Tensor<double, 4> F_rskl{"F_rskl", nobs_, nobs_, nact_, nact_};

    form_teints("FG", &W_pqkl, {'O', 'o', 'O', 'o'});
    form_teints("K", &K_pqrs, {'O', 'O', 'O', 'O'});
    form_teints("F", &F_rskl, {'O', 'o', 'O', 'o'});
    Tensor W_vvoo = W_pqkl(Range{nocc_, nobs_}, Range{nocc_, nobs_}, All, All);
    Tensor K_vvpq = K_pqrs(Range{nocc_, nobs_}, Range{nocc_, nobs_}, All, All);
    form_D_Werner(D_ijab.get(), tau, t_ia);

    Tensor<double, 4> E_F12b_temp{"E_F12b_temp", nact_, nact_, nact_, nact_};
    einsum(1.0, Indices{p, q, k, l}, &W_vvoo, -1.0, Indices{p, q, r, s}, K_vvpq, Indices{r, s, k, l}, F_rskl);
    einsum(1.0, Indices{i, j, k, l}, &E_F12b_temp, 1.0, Indices{p, q, k, l}, W_vvoo, Indices{i, j, p, q}, D_ijab);
    for (int i = 0; i < nact_; i++) {
        E_f12b_ += 0.5 * E_F12b_temp(i, i, i, i);
    }
}

void CCSDF12B::form_CCSDF12B_Energy_df(einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    E_f12b_ = 0.0;

    auto D_ijab = std::make_unique<Tensor<double, 4>>("D_ijab", nact_, nact_, nobs_, nobs_);
    Tensor<double, 4> W_pqkl{"W_pqkl", nobs_, nobs_, nact_, nact_};
    Tensor<double, 4> K_pqrs{"K_pqrs", nobs_, nobs_, nobs_, nobs_};
    Tensor<double, 4> F_rskl{"F_rskl", nobs_, nobs_, nact_, nact_};

    form_df_teints("FG", &W_pqkl, J_inv_AB, {'O', 'o', 'O', 'o'});
    form_df_teints("K", &K_pqrs, J_inv_AB, {'O', 'O', 'O', 'O'});
    form_df_teints("F", &F_rskl, J_inv_AB, {'O', 'o', 'O', 'o'});
    Tensor W_vvoo = W_pqkl(Range{nocc_, nobs_}, Range{nocc_, nobs_}, All, All);
    Tensor K_vvpq = K_pqrs(Range{nocc_, nobs_}, Range{nocc_, nobs_}, All, All);
    form_D_Werner(D_ijab.get(), tau, t_ia);

    Tensor<double, 4> E_F12b_temp{"E_F12b_temp", nact_, nact_, nact_, nact_};
    einsum(1.0, Indices{p, q, k, l}, &W_vvoo, -1.0, Indices{p, q, r, s}, K_vvpq, Indices{r, s, k, l}, F_rskl);
    einsum(1.0, Indices{i, j, k, l}, &E_F12b_temp, 1.0, Indices{p, q, k, l}, W_vvoo, Indices{i, j, p, q}, D_ijab);
    for (int i = 0; i < nact_; i++) {
        E_f12b_ += 0.5 * E_F12b_temp(i, i, i, i);
    }
}

void CCSDF12B::form_CCSD_Energy(einsums::Tensor<double, 4> *G, einsums::Tensor<double, 4> *V, einsums::Tensor<double, 4> *tau)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    /* N.B. G_ijab == K_ijab in Alder papers */

    E_ccsd_ = 0.0;

    Tensor<double, 0> E_ccsd_tmp{"CCSD Energy"};

    Tensor<double, 4> tmp{"tmp", nocc_, nocc_, nvir_, nvir_};
    
    sort(1.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, *tau);
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, a, b}, *tau);
    einsum(0.0, Indices{}, &E_ccsd_tmp, 1.0, Indices{i, j, a, b}, *G, Indices{i, j, a, b}, tmp);

    for (int i = nfrzn_; i < nact_; i++) {
        for (int j = i; j < nact_; j++) {
            E_ccsd_tmp += T_ijkl(i, j, i, j) * (*V)(i, j, i, j);
            E_ccsd_tmp += T_ijkl(i, j, j, i) * (*V)(i, j, j, i);
        }
    }

    E_ccsd_ = E_ccsd_tmp;

}

////////////////////////////////
//* Disk Algorithm (CONV/DF) *//
////////////////////////////////

DiskCCSDF12B::DiskCCSDF12B(SharedWavefunction reference_wavefunction, Options& options):
    CCSDF12B(reference_wavefunction, options) {
    common_init();
}

DiskCCSDF12B::~DiskCCSDF12B() {}

void DiskCCSDF12B::form_D_ijab(einsums::DiskTensor<double, 4> *D, einsums::DiskTensor<double, 2> *f)
{
    using namespace einsums;

    auto f_view = (*f)(Range{nfrzn_, nobs_}, Range{nfrzn_, nobs_}); f_view.set_read_only(true);

#pragma omp parallel for collapse(2) num_threads(nthreads_)
    for (size_t i = nfrzn_; i < nocc_; i++) {
        for (size_t j = nfrzn_; j < nocc_; j++) {
            auto D_view = (*D)(i - nfrzn_, j - nfrzn_, All, All);
            D_view.zero();
            double e_ij = -1.0 * (f_view(i, i) + f_view(j, j));

            for (size_t a = nocc_; a < nobs_; a++) {
                for (size_t b = nocc_; b < nobs_; b++) {
                    D_view(a - nocc_, b - nocc_) = 1.0 / (e_ij + f_view(a, a) + f_view(b, b));
                }
            }
        }
    }
}

void DiskCCSDF12B::form_D_ia(einsums::DiskTensor<double, 2> *D, einsums::DiskTensor<double, 2> *f)
{
    using namespace einsums;

    auto f_view = (*f)(Range{0, nobs_}, Range{0, nobs_}); f_view.set_read_only(true);

#pragma omp parallel for collapse(2) num_threads(nthreads_)
    for (size_t i = nfrzn_; i < nocc_; i++) {
        auto D_view = (*D)(i - nfrzn_, All);
        D_view.zero();
        for (size_t a = nocc_; a < nobs_; a++) {
            auto denom = f_view(a, a) - f_view(i, i);
            D_view(a - nocc_) = (1 / denom);
        }
    }
}

void DiskCCSDF12B::form_f12_energy(einsums::DiskTensor<double,4> *V, einsums::DiskTensor<double,4> *X,
                                 einsums::DiskTensor<double,4> *C, einsums::DiskTensor<double,4> *B,
                                 einsums::DiskTensor<double,2> *f, einsums::DiskTensor<double,4> *G,
                                 einsums::DiskTensor<double,4> *D)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto E_f12_s = 0.0;
    auto E_f12_t = 0.0;
    int kd;

    auto f_act = (*f)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}); f_act.set_read_only(true);
    auto X_klmn = (*X)(All, All, All, All); X_klmn.set_read_only(true);
    auto B_klmn = (*B)(All, All, All, All); B_klmn.set_read_only(true);

    outfile->Printf("  \n");
    outfile->Printf("  %1s   %1s  |     %14s     %14s     %12s \n",
                    "i", "j", "E_F12(Singlet)", "E_F12(Triplet)", "E_F12");
    outfile->Printf(" ----------------------------------------------------------------------\n");
    for (size_t i = 0; i < nact_; i++) {
        for (size_t j = i; j < nact_; j++) {
            // Building B
            Tensor B_ = B_klmn.get();
            {
                Tensor X_ = X_klmn.get();
                auto f_scale = f_act(i, i) + f_act(j, j);
                linear_algebra::scale(f_scale, &X_);
                sort(1.0, Indices{k, l, m, n}, &B_, -1.0, Indices{k, l, m, n}, X_);
            }

            // Getting V_Tilde and B_Tilde
            Tensor V_ = ((*V)(i, j, All, All)).get();
            auto G_ = (*G)(i, j, Range{nocc_, nobs_}, Range{nocc_, nobs_});
            auto D_ = (*D)(i, j, All, All);
            auto VT = V_Tilde(V_, C, G_, D_, i, j);
            auto BT = B_Tilde(B_, C, D_, i, j);

            // Computing the energy
            ( i == j ) ? ( kd = 1 ) : ( kd = 2 );
            auto E_s = kd * (2 * VT.first + BT.first);
            E_f12_s += E_s;
            auto E_t = 0.0;
            if ( i != j ) {
                E_t = 3.0 * kd * (2 * VT.second + BT.second);
                E_f12_t += E_t;
            }
            auto E_f = E_s + E_t;
            outfile->Printf("%3d %3d  |   %16.12f   %16.12f     %16.12f \n", i+1, j+1, E_s, E_t, E_f);
        }
    }

    set_scalar_variable("F12 OPPOSITE-SPIN CORRELATION ENERGY", E_f12_s);
    set_scalar_variable("F12 SAME-SPIN CORRELATION ENERGY", E_f12_t);

    E_f12_ = E_f12_s + E_f12_t;
}

void DiskCCSDF12B::form_cabs_singles(einsums::DiskTensor<double,2> *f)
{
    using namespace einsums;
    using namespace linear_algebra;

    int all_vir = nvir_ + ncabs_;

    // Diagonalize f_ij and f_AB
    Tensor<double, 1> e_ij{"occupied e-vals", nocc_};
    Tensor<double, 1> e_AB{"vir and CABS e-vals", all_vir};

    DiskView<double, 2, 2> C_ij{(*f), Dim<2>{nocc_, nocc_}, Count<2>{nocc_, nocc_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; C_ij.set_read_only(true);
    auto C_AB = (*f)(Range{nocc_, nri_}, Range{nocc_, nri_});

    syev(&(C_ij.get()), &e_ij);
    syev(&(C_AB.get()), &e_AB);

    // Form f_iA
    Tensor<double, 2> f_iA{"Fock Occ-All_vir", nocc_, all_vir};
    {
        DiskView<double, 2, 2> f_view{(*f), Dim<2>{nocc_, all_vir}, Count<2>{nocc_, all_vir}, Offset<2>{0, nocc_}, Stride<2>{1, 1}}; f_view.set_read_only(true);

        gemm<false, false>(1.0, C_ij.get(), gemm<false, true>(1.0, f_view.get(), C_AB.get()), 0.0, &f_iA);
    }

    double E_s = 0.0;
#pragma omp parallel for collapse(2) num_threads(nthreads_) reduction(+:E_s)
    for (size_t A = 0; A < all_vir; A++) {
        for (size_t i = 0; i < nocc_; i++) {
            E_s += 2 * pow(f_iA(i, A), 2) / (e_ij(i) - e_AB(A));
        }
    }

    E_singles_ = E_s;
}

double DiskCCSDF12B::compute_energy() {

    using namespace einsums;
    timer::initialize();

    print_header();

    /* Form the orbital spaces */
    timer_on("OBS and CABS");
    form_basissets();
    timer_off("OBS and CABS");

    // Disable HDF5 diagnostic reporting.
    H5Eset_auto(0, nullptr, nullptr);
    std::string file_name = "Data_" + std::to_string(nocc_) + "_" + std::to_string(ncabs_);
    if (use_df_) {
        file_name += "_" + std::to_string(naux_);
    }
    file_name += ".h5";

    if (f12_read_ints_) {
        // Reads existing file
        einsums::state::data() = h5::open(file_name, H5F_ACC_RDWR);
    } else {
        // Creates new file
        einsums::state::data() = h5::create(file_name, H5F_ACC_TRUNC);
    }

    outfile->Printf("\n ===> Forming the Integrals <===");
    outfile->Printf("\n No screening will be used to compute integrals\n\n");

    /* Form the two-electron integrals */
    // Two-Electron Integrals
    auto G = std::make_unique<DiskTensor<double, 4>>(state::data(), "MO G Tensor", nact_, nact_, nobs_, nri_);
    auto F_KetBig = std::make_unique<DiskTensor<double, 4>>(state::data(), "MO F12 (ket) Tensor", nact_, nact_, nri_, nri_);
    auto F_BraBig = std::make_unique<DiskTensor<double, 4>>(state::data(), "MO F12 (bra) Tensor", nobs_, nobs_, nact_, nact_);
    auto F2 = std::make_unique<DiskTensor<double, 4>>(state::data(), "MO F12_Squared Tensor", nact_, nact_, nact_, nri_);
    auto FG = std::make_unique<DiskTensor<double, 4>>(state::data(), "MO F12G12 Tensor", nobs_, nobs_, nact_, nact_);
    auto Uf = std::make_unique<DiskTensor<double, 4>>(state::data(), "MO F12_DoubleCommutator Tensor", nact_, nact_, nact_, nact_);
    auto K = std::make_unique<DiskTensor<double, 4>>(state::data(), "MO K Tensor", nobs_, nobs_, nobs_, nobs_);
    auto J = std::make_unique<DiskTensor<double, 4>>(state::data(), "MO J Tensor", nobs_, nobs_, nobs_, nobs_);

    std::vector<std::string> teint = {};
    if (!(*FG).existed()) teint.push_back("FG");
    if (!(*J).existed()) teint.push_back("J");
    if (!(*K).existed()) teint.push_back("K");
    if (!(*Uf).existed()) teint.push_back("Uf");
    if (!(*G).existed()) teint.push_back("G");
    if (!(*F_KetBig).existed()) teint.push_back("F_Ket");
    if (!(*F_BraBig).existed()) teint.push_back("F_Bra");
    if (!(*F2).existed()) teint.push_back("F2");
    if (teint.size() == 0) outfile->Printf("   Two-Electron Integrals\n");

    // Fock Matrices
    auto f = std::make_unique<DiskTensor<double, 2>>(state::data(), "Fock Matrix", nri_, nri_);
    auto k = std::make_unique<DiskTensor<double, 2>>(state::data(), "Exchange Matrix", nri_, nri_);
    auto fk = std::make_unique<DiskTensor<double, 2>>(state::data(), "Fock-Exchange Matrix", nri_, nri_);

    if (use_df_) {
        outfile->Printf("   Fock Matrix\n");
        if (!(*f).existed() && !(*k).existed() && !(*fk).existed()) {
            timer_on("Fock Matrix");
            {
                DiskView<double, 2, 2> f_view{(*f), Dim<2>{nri_, nri_}, Count<2>{nri_, nri_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
                DiskView<double, 2, 2> k_view{(*k), Dim<2>{nri_, nri_}, Count<2>{nri_, nri_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
                DiskView<double, 2, 2> fk_view{(*fk), Dim<2>{nri_, nri_}, Count<2>{nri_, nri_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
                f_view.zero();
                k_view.zero();
                fk_view.zero();
            }
            form_df_fock(f.get(), k.get(), fk.get());
            timer_off("Fock Matrix");
        }

        timer_on("Metric Integrals");
        auto J_inv_AB = std::make_unique<Tensor<double, 3>>("Metric MO ([J_AB]^{-1})", naux_, nobs_, nri_);
        form_metric_ints(J_inv_AB.get(), false);
        {
            double mean = 0.0;
            double mean_ii = 0.0;
            double mean_ia = 0.0;
            double mean_ai = 0.0;
            double mean_aa = 0.0;
            double mean_ix = 0.0;
            double mean_ax = 0.0;
            double x_sq_sum = 0.0;
            double x_sq_ii = 0.0;
            double x_sq_ia = 0.0;
            double x_sq_ai = 0.0;
            double x_sq_aa = 0.0;
            double x_sq_ix = 0.0;
            double x_sq_ax = 0.0;
            for (int alp = 0; alp < naux_; alp++) {
                for (int p = 0; p < nobs_; p++) {
                    for (int q = 0; q < nri_; q++) {
                        mean += (*J_inv_AB)(alp, p, q);
                        x_sq_sum += pow((*J_inv_AB)(alp, p, q), 2);
                        if (p < nocc_ && q < nocc_) {
                            mean_ii += (*J_inv_AB)(alp, p, q);
                            x_sq_ii += pow((*J_inv_AB)(alp, p, q), 2);
                        } else if (p < nocc_ && q >= nobs_) {
                            mean_ix += (*J_inv_AB)(alp, p, q);
                            x_sq_ix += pow((*J_inv_AB)(alp, p, q), 2);
                        } else if (p < nocc_ && q >= nocc_) {
                            mean_ia += (*J_inv_AB)(alp, p, q);
                            x_sq_ia += pow((*J_inv_AB)(alp, p, q), 2);
                        } else if (p >= nocc_ && q < nocc_) {
                            mean_ai += (*J_inv_AB)(alp, p, q);
                            x_sq_ai += pow((*J_inv_AB)(alp, p, q), 2);
                        } else if (p >= nocc_ && q >= nobs_) {
                            mean_ax += (*J_inv_AB)(alp, p, q);
                            x_sq_ax += pow((*J_inv_AB)(alp, p, q), 2);
                        } else {
                            mean_aa += (*J_inv_AB)(alp, p, q);
                            x_sq_aa += pow((*J_inv_AB)(alp, p, q), 2);
                        } 
                    }
                }
            }
            mean /= (naux_ * nobs_ * nri_);
            mean_ii /= (naux_ * nocc_ * nocc_);
            mean_ia /= (naux_ * nocc_ * nvir_);
            mean_ai /= (naux_ * nvir_ * nocc_);
            mean_aa /= (naux_ * nvir_ * nvir_);
            mean_ix /= (naux_ * nocc_ * ncabs_);
            mean_ax /= (naux_ * nvir_ * ncabs_);
            x_sq_sum /= (naux_ * nobs_ * nri_);
            x_sq_ii /= (naux_ * nocc_ * nocc_);
            x_sq_ia /= (naux_ * nocc_ * nvir_);
            x_sq_ai /= (naux_ * nvir_ * nocc_);
            x_sq_aa /= (naux_ * nvir_ * nvir_);
            x_sq_ix /= (naux_ * nocc_ * ncabs_);
            x_sq_ax /= (naux_ * nvir_ * ncabs_);
            outfile->Printf("   J_inv_AB Mean: %e, Std. Dev.: %e\n", mean, sqrt(x_sq_sum - pow(mean, 2)));
            outfile->Printf("   J_inv_AB (ii) Mean: %e, Std. Dev.: %e\n", mean_ii, sqrt(x_sq_ii - pow(mean_ii, 2)));
            outfile->Printf("   J_inv_AB (ia) Mean: %e, Std. Dev.: %e\n", mean_ia, sqrt(x_sq_ia - pow(mean_ia, 2)));
            outfile->Printf("   J_inv_AB (ai) Mean: %e, Std. Dev.: %e\n", mean_ai, sqrt(x_sq_ai - pow(mean_ai, 2)));
            outfile->Printf("   J_inv_AB (aa) Mean: %e, Std. Dev.: %e\n", mean_aa, sqrt(x_sq_aa - pow(mean_aa, 2)));
            outfile->Printf("   J_inv_AB (ix) Mean: %e, Std. Dev.: %e\n", mean_ix, sqrt(x_sq_ix - pow(mean_ix, 2)));
            outfile->Printf("   J_inv_AB (ax) Mean: %e, Std. Dev.: %e\n", mean_ax, sqrt(x_sq_ax - pow(mean_ax, 2)));
        }
        timer_off("Metric Integrals");

        for (int i = 0; i < teint.size(); i++){
            if ( teint[i] == "F_Bra" ){
                outfile->Printf("   F (Big Bra Basis) Integral\n");
                timer_on("F_12 (Bra) Integral");
                {
                    DiskView<double, 4, 4>F_BraView{(*F_BraBig), Dim<4>{nobs_, nobs_, nact_, nact_}, Count<4>{nobs_, nobs_, nact_, nact_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    F_BraView.zero();
                }
                form_df_teints(teint[i], F_BraBig.get(), J_inv_AB.get());
                timer_off("F_12 (Bra) Integral");
            } else if ( teint[i] == "F_Ket" ){
                outfile->Printf("   F (Big Ket Basis) Integral Start\n");
                timer_on("F_12 (Ket) Integral");
                {
                    DiskView<double, 4, 4>F_KetView{(*F_KetBig), Dim<4>{nact_, nact_, nri_, nri_}, Count<4>{nact_, nact_, nri_, nri_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    F_KetView.zero();
                }
                form_df_teints(teint[i], F_KetBig.get(), J_inv_AB.get());
                timer_off("F_12 (Ket) Integral");
            } else if ( teint[i] == "FG" ){
                outfile->Printf("   FG Integral\n");
                timer_on("FG_12 Integral");
                {
                    DiskView<double, 4, 4>FGView{(*FG), Dim<4>{nobs_, nobs_, nact_, nact_}, Count<4>{nobs_, nobs_, nact_, nact_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    FGView.zero();
                }
                form_df_teints(teint[i], FG.get(), J_inv_AB.get());
                timer_off("FG_12 Integral");
            } else if ( teint[i] == "F2" ){
                outfile->Printf("   F Squared Integral\n");
                timer_on("F^2_12 Integral");
                {
                    DiskView<double, 4, 4>F2View{(*F2), Dim<4>{nact_, nact_, nact_, nri_}, Count<4>{nact_, nact_, nact_, nri_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    F2View.zero();
                }
                form_df_teints(teint[i], F2.get(), J_inv_AB.get());
                timer_off("F^2_12 Integral");
            } else if ( teint[i] == "Uf" ){
                outfile->Printf("   F Double Commutator Integral\n");
                timer_on("U^F_12 Integral");
                {
                    DiskView<double, 4, 4>UfView{(*Uf), Dim<4>{nact_, nact_, nact_, nact_}, Count<4>{nact_, nact_, nact_, nact_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    UfView.zero();
                }
                form_df_teints(teint[i], Uf.get(), J_inv_AB.get());
                timer_off("U^F_12 Integral");
            } else if ( teint[i] == "K" ){
                outfile->Printf("   K Integral\n");
                timer_on("K Integral");
                {
                    DiskView<double, 4, 4>KView{(*K), Dim<4>{nobs_, nobs_, nobs_, nobs_}, Count<4>{nobs_, nobs_, nobs_, nobs_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    KView.zero();
                }
                form_df_teints(teint[i], K.get(), J_inv_AB.get());
                timer_off("K Integral");
            } else if ( teint[i] == "J" ){
                outfile->Printf("   J Integral\n");
                timer_on("J Integral");
                {
                    DiskView<double, 4, 4>JView{(*J), Dim<4>{nobs_, nobs_, nobs_, nobs_}, Count<4>{nobs_, nobs_, nobs_, nobs_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    JView.zero();
                }
                form_df_teints(teint[i], J.get(), J_inv_AB.get());
                timer_off("J Integral");
            } else {
                outfile->Printf("   G Integral\n");
                timer_on("G Integral");
                {
                    DiskView<double, 4, 4>GView{(*G), Dim<4>{nact_, nact_, nobs_, nri_}, Count<4>{nact_, nact_, nobs_, nri_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    GView.zero();
                }
                form_df_teints(teint[i], G.get(), J_inv_AB.get());
                timer_off("G Integral");
            }
        }
    } else {
        outfile->Printf("   Fock Matrix\n");
        if (!(*f).existed() && !(*k).existed() && !(*fk).existed()) {
            timer_on("Fock Matrix");
            {
                DiskView<double, 2, 2> f_view{(*f), Dim<2>{nri_, nri_}, Count<2>{nri_, nri_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
                DiskView<double, 2, 2> k_view{(*k), Dim<2>{nri_, nri_}, Count<2>{nri_, nri_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
                DiskView<double, 2, 2> fk_view{(*fk), Dim<2>{nri_, nri_}, Count<2>{nri_, nri_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
                f_view.zero();
                k_view.zero();
                fk_view.zero();
            }
            form_fock(f.get(), k.get(), fk.get());
            timer_off("Fock Matrix");
        }

        for (int i = 0; i < teint.size(); i++){
            if ( teint[i] == "F_Bra" ){
                outfile->Printf("   F (Big Ket Basis) Integral\n");
                timer_on("F_12 (Bra) Integral");
                {
                    DiskView<double, 4, 4>F_BraView{(*F_BraBig), Dim<4>{nobs_, nobs_, nact_, nact_}, Count<4>{nobs_, nobs_, nact_, nact_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    F_BraView.zero();
                }
                form_teints(teint[i], F_BraBig.get());
                timer_off("F_12 (Bra) Integral");
            } else if ( teint[i] == "F_Ket" ){
                outfile->Printf("   F_12 (Ket) Integral\n");
                timer_on("F_12 (Ket) Integral");
                {
                    DiskView<double, 4, 4>F_KetView{(*F_KetBig), Dim<4>{nact_, nact_, nri_, nri_}, Count<4>{nact_, nact_, nri_, nri_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    F_KetView.zero();
                }
                form_teints(teint[i], F_KetBig.get());
                timer_off("F_12 (Ket) Integral");
            } else if ( teint[i] == "FG" ){
                outfile->Printf("   FG Integral\n");
                timer_on("FG_12 Integral");
                {
                    DiskView<double, 4, 4>FGView{(*FG), Dim<4>{nobs_, nobs_, nact_, nact_}, Count<4>{nobs_, nobs_, nact_, nact_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    FGView.zero();
                }
                form_teints(teint[i], FG.get());
                timer_off("FG_12 Integral");
            } else if ( teint[i] == "F2" ){
                outfile->Printf("   F Squared Integral\n");
                timer_on("F^2_12 Integral");
                {
                    DiskView<double, 4, 4>F2View{(*F2), Dim<4>{nact_, nact_, nact_, nri_}, Count<4>{nact_, nact_, nact_, nri_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    F2View.zero();
                }
                form_teints(teint[i], F2.get());
                timer_off("F^2_12 Integral");
            } else if ( teint[i] == "Uf" ){
                outfile->Printf("   F Double Commutator Integral\n");
                timer_on("U^F_12 Integral");
                {
                    DiskView<double, 4, 4>UfView{(*Uf), Dim<4>{nact_, nact_, nact_, nact_}, Count<4>{nact_, nact_, nact_, nact_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    UfView.zero();
                }
                form_teints(teint[i], Uf.get());
                timer_off("U^F_12 Integral");
            } else if ( teint[i] == "K" ){
                outfile->Printf("   K Integral\n");
                timer_on("K Integral");
                {
                    DiskView<double, 4, 4>KView{(*K), Dim<4>{nobs_, nobs_, nobs_, nobs_}, Count<4>{nobs_, nobs_, nobs_, nobs_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    KView.zero();
                }
                form_teints(teint[i], K.get());
                timer_off("K Integral");
            } else if ( teint[i] == "J" ){
                outfile->Printf("   J Integral\n");
                timer_on("J Integral");
                {
                    DiskView<double, 4, 4>JView{(*J), Dim<4>{nobs_, nobs_, nobs_, nobs_}, Count<4>{nobs_, nobs_, nobs_, nobs_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    JView.zero();
                }
                form_teints(teint[i], J.get());
                timer_off("J Integral");
            } else {
                outfile->Printf("   G Integral\n");
                timer_on("G Integral");
                {
                    DiskView<double, 4, 4>GView{(*G), Dim<4>{nact_, nact_, nobs_, nri_}, Count<4>{nact_, nact_, nobs_, nri_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
                    GView.zero();
                }
                form_teints(teint[i], G.get());
                timer_off("G Integral");
            }
        }
    }

    /* Check K_prsq*/
    /*outfile->Printf("K_5pq5");
    auto K_pqrs = (*K)(All, All, All, All); K_pqrs.set_read_only(true);
    for (int i = 0; i < nobs_; i++) {
        outfile->Printf("K(5, %d, :", i);
        for (int j = 0; j < nobs_; j++) {
            outfile->Printf(" %d, 5) = %e, ", j, K_pqrs(5, i, j, 5));
        }
        outfile->Printf("\n");
    }*/

    /* Form the F12 Matrices */
    outfile->Printf("\n ===> Forming the F12 Intermediate Tensors <===\n");
    /*auto D_Werner = std::make_unique<DiskTensor<double, 4>>(state::data(), "D (from Werner 1992) Intermediate Tensor", nact_, nact_, nobs_ - nfrzn_, nobs_ - nfrzn_);
    auto L_oovv = std::make_unique<DiskTensor<double, 4>>(state::data(), "L Intermediate Tensor (extra vir)", nact_, nact_, nvir_, nvir_);
    auto L_ooov = std::make_unique<DiskTensor<double, 4>>(state::data(), "L Intermediate Tensor (extra act)", nact_, nact_, nact_, nvir_);
    auto beta = std::make_unique<DiskTensor<double, 2>>(state::data(), "Constant matrix beta", nact_, nact_);
    auto A = std::make_unique<DiskTensor<double, 2>>(state::data(), "A Intermediate Tensor", nvir_, nvir_);
    auto s = std::make_unique<DiskTensor<double, 2>>(state::data(), "s Intermediate Tensor", nact_, nvir_);
    auto r = std::make_unique<DiskTensor<double, 2>>(state::data(), "r Intermediate Tensor", nact_, nvir_);*/
    auto v_ia = std::make_unique<DiskTensor<double, 2>>(state::data(), "T1 Residual (v) Tensor", nact_, nvir_);
    /*auto X_Werner = std::make_unique<DiskTensor<double, 2>>(state::data(), "X (from Werner 1992) Intermediate Tensor", nvir_, nvir_);
    auto Y_Werner = std::make_unique<DiskTensor<double, 4>>(state::data(), "Y (from Werner 1992) Intermediate Tensor", nact_, nact_, nvir_, nvir_);
    auto Z_Werner = std::make_unique<DiskTensor<double, 4>>(state::data(), "Z (from Werner 1992) Intermediate Tensor", nact_, nact_, nvir_, nvir_);
    auto alpha = std::make_unique<DiskTensor<double, 4>>(state::data(), "Constant matrix alpha", nact_, nact_, nact_, nact_);
    auto G_ij = std::make_unique<DiskTensor<double, 4>>(state::data(), "G Intermediate Tensor", nact_, nact_, nvir_, nvir_);*/
    auto V_ijab = std::make_unique<DiskTensor<double, 4>>(state::data(), "T2 Residual (V) Tensor", nact_, nact_, nvir_, nvir_);
    auto t_ia = std::make_unique<DiskTensor<double, 2>>(state::data(), "T1 Amplitude Tensor", nact_, nvir_);
    auto T_ijab = std::make_unique<DiskTensor<double, 4>>(state::data(), "T2 Amplitude Tensor", nact_, nact_, nvir_, nvir_);
    auto tau = std::make_unique<DiskTensor<double, 4>>(state::data(), "(T_ijab + t_ia * t_jb) Tensor", nact_, nact_, nvir_, nvir_);
    auto taut = std::make_unique<DiskTensor<double, 4>>(state::data(), "(0.5 T_ijab + t_ia * t_jb) Tensor", nact_, nact_, nvir_, nvir_);
    auto V = std::make_unique<DiskTensor<double, 4>>(state::data(), "V Intermediate Tensor", nact_, nact_, nact_, nact_);
    //auto X = std::make_unique<DiskTensor<double, 4>>(state::data(), "X Intermediate Tensor", nact_, nact_, nact_, nact_);
    auto C = std::make_unique<DiskTensor<double, 4>>(state::data(), "C Intermediate Tensor", nact_, nact_, nvir_, nvir_);
    //auto B = std::make_unique<DiskTensor<double, 4>>(state::data(), "B Intermediate Tensor", nact_, nact_, nact_, nact_);
    auto D_ijab = std::make_unique<DiskTensor<double, 4>>(state::data(), "D_ijab Tensor", nact_, nact_, nvir_, nvir_);
    auto D_ia = std::make_unique<DiskTensor<double, 2>>(state::data(), "D_ia Tensor", nact_, nvir_);

    /* Form Energy Denominators (constant) */
    if (!(*D_ijab).existed() && !(*D_ia).existed()) {
        timer_on("Energy Denom");
        outfile->Printf("   Fock Matrix Data (xx):\n");
        auto f_read = (*f)(All, All); f_read.set_read_only(true);
        double meanValue = 0.0;
        double mean_ii = 0.0;
        double mean_aa = 0.0;
        double mean_ia = 0.0;
        double mean_ai = 0.0;
        double mean_ix = 0.0;
        double mean_xi = 0.0;
        double mean_xa = 0.0;
        double mean_ax = 0.0;
        double mean_xx = 0.0;
        double x_squared = 0.0;
        double x_sq_ii = 0.0;
        double x_sq_aa = 0.0;
        double x_sq_ia = 0.0;
        double x_sq_ai = 0.0;
        double x_sq_ix = 0.0;
        double x_sq_xi = 0.0;
        double x_sq_xa = 0.0;
        double x_sq_ax = 0.0;
        double x_sq_xx = 0.0;
        for (int i = 0; i < nri_; i++) {
            for (int j = 0; j < nri_; j++) {
                if (i < nocc_ && j < nocc_) {
                    mean_ii += abs(f_read(i, j));
                    x_sq_ii += f_read(i, j) * f_read(i, j);
                } else if (i < nobs_ && j < nobs_) {
                    mean_aa += abs(f_read(i, j));
                    x_sq_aa += f_read(i, j) * f_read(i, j);
                } else if (i < nocc_ && j < nobs_) {
                    mean_ia += abs(f_read(i, j));
                    x_sq_ia += f_read(i, j) * f_read(i, j);
                } else if (i < nobs_ && j < nocc_) {
                    mean_ai += abs(f_read(i, j));
                    x_sq_ai += f_read(i, j) * f_read(i, j);
                } else if (i >= nobs_ && j < nocc_) {
                    mean_xi += abs(f_read(i, j));
                    x_sq_xi += f_read(i, j) * f_read(i, j);
                } else if (i < nocc_ && j >= nobs_) {
                    mean_ix += abs(f_read(i, j));
                    x_sq_ix += f_read(i, j) * f_read(i, j);
                } else if (i < nobs_ && j >= nobs_) {
                    mean_ax += abs(f_read(i, j));
                    x_sq_ax += f_read(i, j) * f_read(i, j);
                } else if (i >= nobs_ && j < nobs_) {
                    mean_xa += abs(f_read(i, j));
                    x_sq_xa += f_read(i, j) * f_read(i, j);
                } else {
                    mean_xx += abs(f_read(i, j));
                    x_sq_xx += f_read(i, j) * f_read(i, j);
                }
                meanValue += abs(f_read(i, j));
                x_squared += f_read(i, j) * f_read(i, j);
                if (f_read(i, j) > 1e+2) {
                    outfile->Printf("      Unusual Value: %e at (%d, %d)\n", f_read(i, j), i, j);
                }
            }
        }
        meanValue /= nri_ * nri_;
        mean_ii /= nocc_ * nocc_;
        mean_aa /= nvir_ * nvir_;
        mean_ia /= nocc_ * nvir_;
        mean_ai /= nocc_ * nvir_;
        mean_ix /= nocc_ * ncabs_;
        mean_xi /= ncabs_ * nocc_;
        mean_xa /= ncabs_ * nvir_;
        mean_ax /= ncabs_ * nvir_;
        mean_xx /= ncabs_ * ncabs_;
        x_squared /= nri_ * nri_;
        x_sq_ii /= nocc_ * nocc_;
        x_sq_aa /= nvir_ * nvir_;
        x_sq_ia /= nocc_ * nvir_;
        x_sq_ai /= nocc_ * nvir_;
        x_sq_ix /= nocc_ * ncabs_;
        x_sq_xi /= ncabs_ * nocc_;
        x_sq_xa /= ncabs_ * nvir_;
        x_sq_ax /= ncabs_ * nvir_;
        x_sq_xx /= ncabs_ * ncabs_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - (meanValue * meanValue)));
        outfile->Printf("      Mean Value (ii): %e, Standard Deviation: %e\n", mean_ii, sqrt(x_sq_ii - (mean_ii * mean_ii)));
        outfile->Printf("      Mean Value (aa): %e, Standard Deviation: %e\n", mean_aa, sqrt(x_sq_aa - (mean_aa * mean_aa)));
        outfile->Printf("      Mean Value (ia): %e, Standard Deviation: %e\n", mean_ia, sqrt(x_sq_ia - (mean_ia * mean_ia)));
        outfile->Printf("      Mean Value (ai): %e, Standard Deviation: %e\n", mean_ai, sqrt(x_sq_ai - (mean_ai * mean_ai)));
        outfile->Printf("      Mean Value (ix): %e, Standard Deviation: %e\n", mean_ix, sqrt(x_sq_ix - (mean_ix * mean_ix)));
        outfile->Printf("      Mean Value (xi): %e, Standard Deviation: %e\n", mean_xi, sqrt(x_sq_xi - (mean_xi * mean_xi)));
        outfile->Printf("      Mean Value (xa): %e, Standard Deviation: %e\n", mean_xa, sqrt(x_sq_xa - (mean_xa * mean_xa)));
        outfile->Printf("      Mean Value (ax): %e, Standard Deviation: %e\n", mean_ax, sqrt(x_sq_ax - (mean_ax * mean_ax)));
        outfile->Printf("      Mean Value (xx): %e, Standard Deviation: %e\n", mean_xx, sqrt(x_sq_xx - (mean_xx * mean_xx)));

        outfile->Printf("   D_ijab Intermediate\n");
        form_D_ijab(D_ijab.get(), f.get());

        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> D_ijab_view{(*D_ijab), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; D_ijab_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(D_ijab_view(i, j, a, b));
                        x_squared += D_ijab_view(i, j, a, b) * D_ijab_view(i, j, a, b);
                        if (D_ijab_view(i, j, a, b) > 1e+2) {
                            outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", D_ijab_view(i, j, a, b), i, j, a, b);
                        }
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        outfile->Printf("   D_ia Intermediate\n");
        form_D_ia(D_ia.get(), f.get());

        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 2, 2> D_ia_view{(*D_ia), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; D_ia_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int a = 0; a < nvir_; a++) {
                meanValue += abs(D_ia_view(i, a));
                x_squared += D_ia_view(i, a) * D_ia_view(i, a);
                if (D_ia_view(i, a) > 1e+2 || abs(D_ia_view(i, a)) < 1e-40) {
                    outfile->Printf("      Unusual Value: %e at (%d, %d)\n", D_ia_view(i, a), i, a);
                }
            }
        }
        meanValue /= nact_ * nvir_;
        x_squared /= nact_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        timer_off("Energy Denom");
    }

    /* Initialise amplitudes */
    if (!(*t_ia).existed() && !(*T_ijab).existed()) {
        timer_on("Initialise Amplitudes");
        outfile->Printf("   T1 and T2 Amplitudes Initialise\n");

        outfile->Printf("   G MO Tensor Data (iipx):\n");
        DiskView<double, 4, 4> G_read{(*G), Dim<4>{nact_, nact_, nobs_, nri_}, Count<4>{nact_, nact_, nobs_, nri_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; G_read.set_read_only(true);
        double meanValue = 0.0;
        double mean_ii = 0.0;
        double mean_ia = 0.0;
        double mean_ai = 0.0;
        double mean_ix = 0.0;
        double mean_aa = 0.0;
        double mean_ax = 0.0;
        double x_squared = 0.0;
        double x_sq_ii = 0.0;
        double x_sq_ia = 0.0;
        double x_sq_ai = 0.0;
        double x_sq_ix = 0.0;
        double x_sq_aa = 0.0;
        double x_sq_ax = 0.0;
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int k = 0; k < nobs_; k++) {
                    for (int l = 0; l < nri_; l++) {
                        if (k < nocc_ && l < nocc_) {
                            mean_ii += abs(G_read(i, j, k, l));
                            x_sq_ii += G_read(i, j, k, l) * G_read(i, j, k, l);
                        } else if (k < nocc_ && l < nobs_) {
                            mean_ia += abs(G_read(i, j, k, l));
                            x_sq_ia += G_read(i, j, k, l) * G_read(i, j, k, l);
                        } else if (k < nobs_ && l < nocc_) {
                            mean_ai += abs(G_read(i, j, k, l));
                            x_sq_ai += G_read(i, j, k, l) * G_read(i, j, k, l);
                        } else if (k < nocc_ && l >= nobs_) {
                            mean_ix += abs(G_read(i, j, k, l));
                            x_sq_ix += G_read(i, j, k, l) * G_read(i, j, k, l);
                        } else if (k >= nocc_ && l < nobs_) {
                            mean_aa += abs(G_read(i, j, k, l));
                            x_sq_aa += G_read(i, j, k, l) * G_read(i, j, k, l);
                        } else {
                            mean_ax += abs(G_read(i, j, k, l));
                            x_sq_ax += G_read(i, j, k, l) * G_read(i, j, k, l);
                        }
                        meanValue += abs(G_read(i, j, k, l));
                        x_squared += G_read(i, j, k, l) * G_read(i, j, k, l);
                        if (G_read(i, j, k, l) > 1e+2) {
                            outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", G_read(i, j, k, l), i, j, k, l);
                        }
                    }
                }
            }
        }

        meanValue /= nocc_ * nocc_ * nobs_ * nri_;
        mean_ii /= nocc_ * nocc_ * nocc_ * nocc_;
        mean_ia /= nocc_ * nocc_ * nocc_ * nvir_;
        mean_ai /= nocc_ * nocc_ * nvir_ * nocc_;
        mean_ix /= nocc_ * nocc_ * nocc_ * ncabs_;
        mean_aa /= nocc_ * nocc_ * nvir_ * nvir_;
        mean_ax /= nocc_ * nocc_ * nvir_ * ncabs_;
        x_squared /= nocc_ * nocc_ * nobs_ * nri_;
        x_sq_ii /= nocc_ * nocc_ * nocc_ * nocc_;
        x_sq_ia /= nocc_ * nocc_ * nocc_ * nvir_;
        x_sq_ai /= nocc_ * nocc_ * nvir_ * nocc_;
        x_sq_ix /= nocc_ * nocc_ * nocc_ * ncabs_;
        x_sq_aa /= nocc_ * nocc_ * nvir_ * nvir_;
        x_sq_ax /= nocc_ * nocc_ * nvir_ * ncabs_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        outfile->Printf("      Mean Value (ii): %e, Standard Deviation: %e\n", mean_ii, sqrt(x_sq_ii - mean_ii * mean_ii));
        outfile->Printf("      Mean Value (ia): %e, Standard Deviation: %e\n", mean_ia, sqrt(x_sq_ia - mean_ia * mean_ia));
        outfile->Printf("      Mean Value (ai): %e, Standard Deviation: %e\n", mean_ai, sqrt(x_sq_ai - mean_ai * mean_ai));
        outfile->Printf("      Mean Value (ix): %e, Standard Deviation: %e\n", mean_ix, sqrt(x_sq_ix - mean_ix * mean_ix));
        outfile->Printf("      Mean Value (aa): %e, Standard Deviation: %e\n", mean_aa, sqrt(x_sq_aa - mean_aa * mean_aa));
        outfile->Printf("      Mean Value (ax): %e, Standard Deviation: %e\n", mean_ax, sqrt(x_sq_ax - mean_ax * mean_ax));

        initialise_amplitudes(t_ia.get(), T_ijab.get(), G.get(), D_ijab.get());

        outfile->Printf("   T1 Amplitude Data:\n");
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 2, 2> t_ia_view{(*t_ia), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_ia_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int a = 0; a < nvir_; a++) {
                meanValue += abs(t_ia_view(i, a));
                x_squared += t_ia_view(i, a) * t_ia_view(i, a);
                if (t_ia_view(i, a) > 1e+2) {
                    outfile->Printf("      Unusual Value: %e at (%d, %d)\n", t_ia_view(i, a), i, a);
                }
            }
        }
        meanValue /= nact_ * nvir_;
        x_squared /= nact_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        outfile->Printf("   T2 Amplitude Data:\n");
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> T_ijab_view{(*T_ijab), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; T_ijab_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(T_ijab_view(i, j, a, b));
                        x_squared += T_ijab_view(i, j, a, b) * T_ijab_view(i, j, a, b);
                        if (T_ijab_view(i, j, a, b) > 1e+2) {
                            outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", T_ijab_view(i, j, a, b), i, j, a, b);
                        }
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        timer_off("Initialise Amplitudes");
    }

    if (!(*tau).existed() && !(*taut).existed()) {
        timer_on("Form Tau");
        outfile->Printf("   Tau Intermediate\n");
        form_tau(tau.get(), T_ijab.get(), t_ia.get());

        double meanValue = 0.0;
        double x_squared = 0.0;
        DiskView<double, 4, 4> tau_view{(*tau), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; tau_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(tau_view(i, j, a, b));
                        x_squared += tau_view(i, j, a, b) * tau_view(i, j, a, b);
                        if (tau_view(i, j, a, b) > 1e+2) {
                            outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", tau_view(i, j, a, b), i, j, a, b);
                        }
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        outfile->Printf("   TauT Intermediate\n");
        form_taut(taut.get(), T_ijab.get(), t_ia.get());

        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> taut_view{(*taut), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; taut_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(taut_view(i, j, a, b));
                        x_squared += taut_view(i, j, a, b) * taut_view(i, j, a, b);
                        if (taut_view(i, j, a, b) > 1e+2) {
                            outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", taut_view(i, j, a, b), i, j, a, b);
                        }
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("Form Tau");
    }

    outfile->Printf("   Check K MO Tensor Data (pppp):\n");
    auto K_read = (*K)(All, All, All, All); K_read.set_read_only(true);
    double meanValue = 0.0;
    double meanTrace = 0.0;
    double mean_iiii = 0.0;
    double mean_aiii = 0.0;
    double mean_iaii = 0.0;
    double mean_iiai = 0.0;
    double mean_iiia = 0.0;
    double mean_aaii = 0.0;
    double mean_aiai = 0.0;
    double mean_iaai = 0.0;
    double mean_iaia = 0.0;
    double mean_iiaa = 0.0;
    double mean_aiia = 0.0;
    double mean_aaai = 0.0;
    double mean_aaia = 0.0;
    double mean_aiaa = 0.0;
    double mean_iaaa = 0.0;
    double mean_aaaa = 0.0;
    double x_squared = 0.0;
    double x_sqTrace = 0.0;
    double x_sq_iiii = 0.0;
    double x_sq_aiii = 0.0;
    double x_sq_iaii = 0.0;
    double x_sq_iiai = 0.0;
    double x_sq_iiia = 0.0;
    double x_sq_aaii = 0.0;
    double x_sq_aiai = 0.0;
    double x_sq_iaai = 0.0;
    double x_sq_iaia = 0.0;
    double x_sq_iiaa = 0.0;
    double x_sq_aiia = 0.0;
    double x_sq_aaai = 0.0;
    double x_sq_aaia = 0.0;
    double x_sq_aiaa = 0.0;
    double x_sq_iaaa = 0.0;
    double x_sq_aaaa = 0.0;
    for (int i = 0; i < nobs_; i++) {
        for (int j = 0; j < nobs_; j++) {
            for (int k = 0; k < nobs_; k++) {
                for (int l = 0; l < nobs_; l++) {
                    if (i == j && j == k && k == l) {
                        meanTrace += abs(K_read(i, j, k, l));
                        x_sqTrace += K_read(i, j, k, l) * K_read(i, j, k, l);
                    }
                    if (i < nocc_ && j < nocc_ && k < nocc_ && l < nocc_) {
                        mean_iiii += abs(K_read(i, j, k, l));
                        x_sq_iiii += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i >= nocc_ && j >= nocc_ && k >= nocc_ && l >= nocc_) {
                        mean_aaaa += abs(K_read(i, j, k, l));
                        x_sq_aaaa += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i >= nocc_ && j < nocc_ && k < nocc_ && l < nocc_) {
                        mean_aiii += abs(K_read(i, j, k, l));
                        x_sq_aiii += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i < nocc_ && j >= nocc_ && k < nocc_ && l < nocc_) {
                        mean_iaii += abs(K_read(i, j, k, l));
                        x_sq_iaii += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i < nocc_ && j < nocc_ && k >= nocc_ && l < nocc_) {
                        mean_iiai += abs(K_read(i, j, k, l));
                        x_sq_iiai += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i < nocc_ && j < nocc_ && k < nocc_ && l >= nocc_) {
                        mean_iiia += abs(K_read(i, j, k, l));
                        x_sq_iiia += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i < nocc_ && j >= nocc_ && k >= nocc_ && l >= nocc_) {
                        mean_iaaa += abs(K_read(i, j, k, l));
                        x_sq_iaaa += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i >= nocc_ && j < nocc_ && k >= nocc_ && l >= nocc_) {
                        mean_aiaa += abs(K_read(i, j, k, l));
                        x_sq_aiaa += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i >= nocc_ && j >= nocc_ && k < nocc_ && l >= nocc_) {
                        mean_aaia += abs(K_read(i, j, k, l));
                        x_sq_aaia += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i >= nocc_ && j >= nocc_ && k >= nocc_ && l < nocc_) {
                        mean_aaai += abs(K_read(i, j, k, l));
                        x_sq_aaai += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i >= nocc_ && j >= nocc_ && k < nocc_ && l < nocc_) {
                        mean_aaii += abs(K_read(i, j, k, l));
                        x_sq_aaii += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i >= nocc_ && j < nocc_ && k >= nocc_ && l < nocc_) {
                        mean_aiai += abs(K_read(i, j, k, l));
                        x_sq_aiai += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i < nocc_ && j >= nocc_ && k >= nocc_ && l < nocc_) {
                        mean_iaai += abs(K_read(i, j, k, l));
                        x_sq_iaai += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i < nocc_ && j < nocc_ && k >= nocc_ && l >= nocc_) {
                        mean_iiaa += abs(K_read(i, j, k, l));
                        x_sq_iiaa += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else if (i < nocc_ && j >= nocc_ && k < nocc_ && l >= nocc_) {
                        mean_iaia += abs(K_read(i, j, k, l));
                        x_sq_iaia += K_read(i, j, k, l) * K_read(i, j, k, l);
                    } else {
                        mean_aiia += abs(K_read(i, j, k, l));
                        x_sq_aiia += K_read(i, j, k, l) * K_read(i, j, k, l);
                    }
                    meanValue += abs(K_read(i, j, k, l));
                    x_squared += K_read(i, j, k, l) * K_read(i, j, k, l);
                    if (K_read(i, j, k, l) > 1e+2) {
                        outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", K_read(i, j, k, l), i, j, k, l);
                    }
                }
            }
        }
    }

    meanValue /= nobs_ * nobs_ * nobs_ * nobs_;
    meanTrace /= nobs_;
    mean_iiii /= nocc_ * nocc_ * nocc_ * nocc_;
    mean_aiii /= nocc_ * nocc_ * nocc_ * nvir_;
    mean_iaii /= nocc_ * nocc_ * nocc_ * nvir_;
    mean_iiai /= nocc_ * nocc_ * nocc_ * nvir_;
    mean_iiia /= nocc_ * nocc_ * nocc_ * nvir_;
    mean_aaii /= nvir_ * nocc_ * nvir_ * nocc_;
    mean_aiai /= nvir_ * nocc_ * nvir_ * nocc_;
    mean_iaai /= nvir_ * nocc_ * nvir_ * nocc_;
    mean_iaia /= nvir_ * nocc_ * nvir_ * nocc_;
    mean_iiaa /= nvir_ * nocc_ * nvir_ * nocc_;
    mean_aiia /= nvir_ * nocc_ * nvir_ * nocc_;
    mean_aaai /= nvir_ * nvir_ * nvir_ * nocc_;
    mean_aaia /= nvir_ * nvir_ * nvir_ * nocc_;
    mean_aiaa /= nvir_ * nvir_ * nvir_ * nocc_;
    mean_iaaa /= nvir_ * nvir_ * nvir_ * nocc_;
    mean_aaaa /= nvir_ * nvir_ * nvir_ * nvir_;
    x_squared /= nobs_ * nobs_ * nobs_ * nobs_;
    x_sqTrace /= nobs_;
    x_sq_iiii /= nocc_ * nocc_ * nocc_ * nocc_;
    x_sq_aiii /= nocc_ * nocc_ * nocc_ * nvir_;
    x_sq_iaii /= nocc_ * nocc_ * nocc_ * nvir_;
    x_sq_iiai /= nocc_ * nocc_ * nocc_ * nvir_;
    x_sq_iiia /= nocc_ * nocc_ * nocc_ * nvir_;
    x_sq_aaii /= nvir_ * nocc_ * nvir_ * nocc_;
    x_sq_aiai /= nvir_ * nocc_ * nvir_ * nocc_;
    x_sq_iaai /= nvir_ * nocc_ * nvir_ * nocc_;
    x_sq_iaia /= nvir_ * nocc_ * nvir_ * nocc_;
    x_sq_iiaa /= nvir_ * nocc_ * nvir_ * nocc_;
    x_sq_aiia /= nvir_ * nocc_ * nvir_ * nocc_;
    x_sq_aaai /= nvir_ * nvir_ * nvir_ * nocc_;
    x_sq_aaia /= nvir_ * nvir_ * nvir_ * nocc_;
    x_sq_aiaa /= nvir_ * nvir_ * nvir_ * nocc_;
    x_sq_iaaa /= nvir_ * nvir_ * nvir_ * nocc_;
    x_sq_aaaa /= nvir_ * nvir_ * nvir_ * nvir_;
    outfile->Printf("      Mean Value: %e, Standard Deviation: %e, Variance: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue), x_squared - meanValue * meanValue);
    outfile->Printf("      Mean Value (Trace): %e, Standard Deviation: %e, Variance: %e\n", meanTrace, sqrt(x_sqTrace - meanTrace * meanTrace), x_sqTrace - meanTrace * meanTrace);
    outfile->Printf("      Mean Value (iiii): %e, Standard Deviation: %e, Variance: %e\n", mean_iiii, sqrt(x_sq_iiii - mean_iiii * mean_iiii), x_sq_iiii - mean_iiii * mean_iiii);
    outfile->Printf("      Mean Value (aiii): %e, Standard Deviation: %e, Variance: %e\n", mean_aiii, sqrt(x_sq_aiii - mean_aiii * mean_aiii), x_sq_aiii - mean_aiii * mean_aiii);
    outfile->Printf("      Mean Value (iaii): %e, Standard Deviation: %e, Variance: %e\n", mean_iaii, sqrt(x_sq_iaii - mean_iaii * mean_iaii), x_sq_iaii - mean_iaii * mean_iaii);
    outfile->Printf("      Mean Value (iiai): %e, Standard Deviation: %e, Variance: %e\n", mean_iiai, sqrt(x_sq_iiai - mean_iiai * mean_iiai), x_sq_iiai - mean_iiai * mean_iiai);
    outfile->Printf("      Mean Value (iiia): %e, Standard Deviation: %e, Variance: %e\n", mean_iiia, sqrt(x_sq_iiia - mean_iiia * mean_iiia), x_sq_iiia - mean_iiia * mean_iiia);
    outfile->Printf("      Mean Value (aaii): %e, Standard Deviation: %e, Variance: %e\n", mean_aaii, sqrt(x_sq_aaii - mean_aaii * mean_aaii), x_sq_aaii - mean_aaii * mean_aaii);
    outfile->Printf("      Mean Value (aiai): %e, Standard Deviation: %e, Variance: %e\n", mean_aiai, sqrt(x_sq_aiai - mean_aiai * mean_aiai), x_sq_aiai - mean_aiai * mean_aiai);
    outfile->Printf("      Mean Value (iaai): %e, Standard Deviation: %e, Variance: %e\n", mean_iaai, sqrt(x_sq_iaai - mean_iaai * mean_iaai), x_sq_iaai - mean_iaai * mean_iaai);
    outfile->Printf("      Mean Value (iaia): %e, Standard Deviation: %e, Variance: %e\n", mean_iaia, sqrt(x_sq_iaia - mean_iaia * mean_iaia), x_sq_iaia - mean_iaia * mean_iaia);
    outfile->Printf("      Mean Value (iiaa): %e, Standard Deviation: %e, Variance: %e\n", mean_iiaa, sqrt(x_sq_iiaa - mean_iiaa * mean_iiaa), x_sq_iiaa - mean_iiaa * mean_iiaa);
    outfile->Printf("      Mean Value (aiia): %e, Standard Deviation: %e, Variance: %e\n", mean_aiia, sqrt(x_sq_aiia - mean_aiia * mean_aiia), x_sq_aiia - mean_aiia * mean_aiia);
    outfile->Printf("      Mean Value (aaai): %e, Standard Deviation: %e, Variance: %e\n", mean_aaai, sqrt(x_sq_aaai - mean_aaai * mean_aaai), x_sq_aaai - mean_aaai * mean_aaai);
    outfile->Printf("      Mean Value (aaia): %e, Standard Deviation: %e, Variance: %e\n", mean_aaia, sqrt(x_sq_aaia - mean_aaia * mean_aaia), x_sq_aaia - mean_aaia * mean_aaia);
    outfile->Printf("      Mean Value (aiaa): %e, Standard Deviation: %e, Variance: %e\n", mean_aiaa, sqrt(x_sq_aiaa - mean_aiaa * mean_aiaa), x_sq_aiaa - mean_aiaa * mean_aiaa);
    outfile->Printf("      Mean Value (iaaa): %e, Standard Deviation: %e, Variance: %e\n", mean_iaaa, sqrt(x_sq_iaaa - mean_iaaa * mean_iaaa), x_sq_iaaa - mean_iaaa * mean_iaaa);
    outfile->Printf("      Mean Value (aaaa): %e, Standard Deviation: %e, Variance: %e\n", mean_aaaa, sqrt(x_sq_aaaa - mean_aaaa * mean_aaaa), x_sq_aaaa - mean_aaaa * mean_aaaa);

    /* Not following this method temporarily
    if (!(*L_oovv).existed() && !(*L_ooov).existed()) {
        outfile->Printf("   L Intermediate\n");
        timer_on("L Intermediate");
        form_L(L_oovv.get(), L_ooov.get(), K.get());

        outfile->Printf("   L_oovv Data:\n");
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> L_oovv_view{(*L_oovv), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; L_oovv_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(L_oovv_view(i, j, a, b));
                        x_squared += L_oovv_view(i, j, a, b) * L_oovv_view(i, j, a, b);
                        if (L_oovv_view(i, j, a, b) > 1e+2) {
                            outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", L_oovv_view(i, j, a, b), i, j, a, b);
                        }
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        outfile->Printf("   L_ooov Data:\n");
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> L_ooov_view{(*L_ooov), Dim<4>{nact_, nact_, nact_, nvir_}, Count<4>{nact_, nact_, nact_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; L_ooov_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int k = 0; k < nact_; k++) {
                    for (int a = 0; a < nvir_; a++) {
                        meanValue += abs(L_ooov_view(i, j, k, a));
                        x_squared += L_ooov_view(i, j, k, a) * L_ooov_view(i, j, k, a);
                        if (L_ooov_view(i, j, k, a) > 1e+2) {
                            outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", L_ooov_view(i, j, k, a), i, j, k, a);
                        }
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nact_ * nvir_;
        x_squared /= nact_ * nact_ * nact_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        timer_off("L Intermediate");
    }//*/

    if (!(*C).existed()) {
        timer_on("C Intermediate");
        outfile->Printf("   Check F MO Tensor Data (iixx):\n");
        DiskView<double, 4, 4> F_read{(*F_KetBig), Dim<4>{nact_, nact_, nri_, nri_}, Count<4>{nact_, nact_, nri_, nri_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; F_read.set_read_only(true);
        double meanValue = 0.0;
        double mean_ii = 0.0;
        double mean_ia = 0.0;
        double mean_ai = 0.0;
        double mean_ix = 0.0;
        double mean_xi = 0.0;
        double mean_aa = 0.0;
        double mean_ax = 0.0;
        double mean_xa = 0.0;
        double mean_xx = 0.0;
        double x_squared = 0.0;
        double x_sq_ii = 0.0;
        double x_sq_ia = 0.0;
        double x_sq_ai = 0.0;
        double x_sq_ix = 0.0;
        double x_sq_xi = 0.0;
        double x_sq_aa = 0.0;
        double x_sq_ax = 0.0;
        double x_sq_xa = 0.0;
        double x_sq_xx = 0.0;
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int k = 0; k < nri_; k++) {
                    for (int l = 0; l < nri_; l++) {
                        if (k < nocc_ && l < nocc_) {
                            mean_ii += abs(F_read(i, j, k, l));
                            x_sq_ii += F_read(i, j, k, l) * F_read(i, j, k, l);
                        } else if (k >= nobs_ && l >= nobs_) {
                            mean_xx += abs(F_read(i, j, k, l));
                            x_sq_xx += F_read(i, j, k, l) * F_read(i, j, k, l);
                        } else if (k < nocc_ && l >= nobs_) {
                            mean_ix += abs(F_read(i, j, k, l));
                            x_sq_ix += F_read(i, j, k, l) * F_read(i, j, k, l);
                        } else if (k >= nobs_ && l < nocc_) {
                            mean_xi += abs(F_read(i, j, k, l));
                            x_sq_xi += F_read(i, j, k, l) * F_read(i, j, k, l);
                        } else if (k >= nocc_ && l >= nobs_) {
                            mean_ax += abs(F_read(i, j, k, l));
                            x_sq_ax += F_read(i, j, k, l) * F_read(i, j, k, l);
                        } else if (k >= nobs_ && l >= nocc_) {
                            mean_xa += abs(F_read(i, j, k, l));
                            x_sq_xa += F_read(i, j, k, l) * F_read(i, j, k, l);
                        }  else if (k >= nocc_ && l < nocc_) {
                            mean_ai += abs(F_read(i, j, k, l));
                            x_sq_ai += F_read(i, j, k, l) * F_read(i, j, k, l);
                        } else if (k < nocc_ && l >= nocc_) {
                            mean_ia += abs(F_read(i, j, k, l));
                            x_sq_ia += F_read(i, j, k, l) * F_read(i, j, k, l);
                        } else {
                            mean_aa += abs(F_read(i, j, k, l));
                            x_sq_aa += F_read(i, j, k, l) * F_read(i, j, k, l);
                        }
                        meanValue += abs(F_read(i, j, k, l));
                        x_squared += F_read(i, j, k, l) * F_read(i, j, k, l);
                        if (F_read(i, j, k, l) > 1e+2) {
                            outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", F_read(i, j, k, l), i, j, k, l);
                        }
                    }
                }
            }
        }

        meanValue /= nact_ * nact_ * nri_ * nri_;
        mean_ii /= nact_ * nact_ * nact_ * nact_;
        mean_ia /= nact_ * nact_ * nact_ * nvir_;
        mean_ai /= nact_ * nact_ * nact_ * nvir_;
        mean_ix /= nact_ * nact_ * nact_ * ncabs_;
        mean_xi /= nact_ * nact_ * nact_ * ncabs_;
        mean_aa /= nact_ * nact_ * nvir_ * nvir_;
        mean_ax /= nact_ * nact_ * nvir_ * ncabs_;
        mean_xa /= nact_ * nact_ * nvir_ * ncabs_;
        mean_xx /= nact_ * ncabs_ * nact_ * ncabs_;
        x_squared /= nact_ * nact_ * nri_ * nri_;
        x_sq_ii /= nact_ * nact_ * nact_ * nact_;
        x_sq_ia /= nact_ * nact_ * nact_ * nvir_;
        x_sq_ai /= nact_ * nact_ * nact_ * nvir_;
        x_sq_ix /= nact_ * nact_ * nact_ * ncabs_;
        x_sq_xi /= nact_ * nact_ * nact_ * ncabs_;
        x_sq_aa /= nact_ * nact_ * nvir_ * nvir_;
        x_sq_ax /= nact_ * nact_ * nvir_ * ncabs_;
        x_sq_xa /= nact_ * nact_ * nvir_ * ncabs_;
        x_sq_xx /= nact_ * ncabs_ * nact_ * ncabs_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        outfile->Printf("      Mean Value (ii): %e, Standard Deviation: %e\n", mean_ii, sqrt(x_sq_ii - mean_ii * mean_ii));
        outfile->Printf("      Mean Value (ia): %e, Standard Deviation: %e\n", mean_ia, sqrt(x_sq_ia - mean_ia * mean_ia));
        outfile->Printf("      Mean Value (ai): %e, Standard Deviation: %e\n", mean_ai, sqrt(x_sq_ai - mean_ai * mean_ai));
        outfile->Printf("      Mean Value (ix): %e, Standard Deviation: %e\n", mean_ix, sqrt(x_sq_ix - mean_ix * mean_ix));
        outfile->Printf("      Mean Value (xi): %e, Standard Deviation: %e\n", mean_xi, sqrt(x_sq_xi - mean_xi * mean_xi));
        outfile->Printf("      Mean Value (aa): %e, Standard Deviation: %e\n", mean_aa, sqrt(x_sq_aa - mean_aa * mean_aa));
        outfile->Printf("      Mean Value (ax): %e, Standard Deviation: %e\n", mean_ax, sqrt(x_sq_ax - mean_ax * mean_ax));
        outfile->Printf("      Mean Value (xa): %e, Standard Deviation: %e\n", mean_xa, sqrt(x_sq_xa - mean_xa * mean_xa));
        outfile->Printf("      Mean Value (xx): %e, Standard Deviation: %e\n", mean_xx, sqrt(x_sq_xx - mean_xx * mean_xx));

        outfile->Printf("   C Intermediate\n");
        form_C(C.get(), F_KetBig.get(), f.get());

        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> C_view{(*C), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; C_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int k = 0; k < nvir_; k++) {
                    for (int l = 0; l < nvir_; l++) {
                        meanValue += abs(C_view(i, j, k, l));
                        x_squared += C_view(i, j, k, l) * C_view(i, j, k, l);
                        if (C_view(i, j, k, l) > 1e+2) {
                            outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", C_view(i, j, k, l), i, j, k, l);
                        }
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        timer_off("C Intermediate");
    }
    
    if (!(*V).existed()) {
        timer_on("V Intermediate");
        outfile->Printf("   V Intermediate\n");
        outfile->Printf("   FG Data:\n");
        DiskView<double, 4, 4> FG_View{(*FG), Dim<4>{nact_, nact_, nact_, nact_}, Count<4>{nact_, nact_, nact_, nact_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; FG_View.set_read_only(false);
        double meanValue = 0.0;
        double x_squared = 0.0;
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int k = 0; k < nact_; k++) {
                    for (int l = 0; l < nact_; l++) {
                        meanValue += abs(FG_View(i, j, k, l));
                        x_squared += FG_View(i, j, k, l) * FG_View(i, j, k, l);
                        if (FG_View(i, j, k, l) > 1e+2) {
                            outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", FG_View(i, j, k, l), i, j, k, l);
                        }
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nact_ * nact_;
        x_squared /= nact_ * nact_ * nact_ * nact_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        form_V_X(V.get(), F_KetBig.get(), G.get(), FG.get());

        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> V_View{(*V), Dim<4>{nact_, nact_, nact_, nact_}, Count<4>{nact_, nact_, nact_, nact_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; V_View.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int k = 0; k < nact_; k++) {
                    for (int l = 0; l < nact_; l++) {
                        meanValue += abs(V_View(i, j, k, l));
                        x_squared += V_View(i, j, k, l) * V_View(i, j, k, l);
                        if (V_View(i, j, k, l) > 1e+2) {
                            outfile->Printf("      Unusual Value: %e at (%d, %d, %d, %d)\n", V_View(i, j, k, l), i, j, k, l);
                        }
                    }
                }
            }
        }

        meanValue /= nact_ * nact_ * nact_ * nact_;
        x_squared /= nact_ * nact_ * nact_ * nact_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        timer_off("V Intermediate");
    }

    iteration_ = 0;
    double rms = 1.0;
    double e_diff = 1.0;

    while (iteration_ < maxiter_ && (rms > rms_tol_ || e_diff > e_tol_)) {
        timer_on("CCSD Iteration");

        /* Temp Store old amplitudes and energy */
        timer_on("Save Amplitudes");
        outfile->Printf("   Set Old T1 and T2 Amplitudes\n");
        auto t_ia_old = std::make_unique<DiskTensor<double, 2>>(state::data(), "Old T1 Amplitude Tensor", nact_, nvir_);
        auto T_ijab_old = std::make_unique<DiskTensor<double, 4>>(state::data(), "Old T2 Amplitude Tensor", nact_, nact_, nvir_, nvir_);
        outfile->Printf("   Save Amplitudes\n");
        save_amplitudes(t_ia_old.get(), T_ijab_old.get(), t_ia.get(), T_ijab.get());
        auto E_old_ = E_ccsd_ + E_f12b_;
        timer_off("Save Amplitudes");

        outfile->Printf("\n ===> CCSD Iteration %d <===\n", iteration_);

        /*
        outfile->Printf("   X Intermediate\n");
        if (!(*X).existed()) {
            timer_on("X Intermediate");
            form_V_X(X.get(), F.get(), F.get(), F2.get());
            timer_off("X Intermediate");
        }
        */
    
        /* New Method
        outfile->Printf("   D (from Werner 1992) Intermediate\n");
        timer_on("D (from Werner 1992) Intermediate");
        form_D_Werner(D_Werner.get(), tau.get(), t_ia.get());//*/
        double meanValue = 0.0;
        double mean_ii = 0.0;
        double mean_ia = 0.0;
        double mean_aa = 0.0;
        double x_squared = 0.0;
        double x_sq_ii = 0.0;
        double x_sq_ia = 0.0;
        double x_sq_aa = 0.0;
        /*DiskView<double, 4, 4> D_Werner_view{(*D_Werner), Dim<4>{nact_, nact_, nobs_-nfrzn_, nobs_-nfrzn_}, Count<4>{nact_, nact_, nobs_-nfrzn_, nobs_-nfrzn_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; D_Werner_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nobs_-nfrzn_; a++) {
                    for (int b = 0; b < nobs_-nfrzn_; b++) {
                        meanValue += abs(D_Werner_view(i, j, a, b));
                        x_squared += D_Werner_view(i, j, a, b) * D_Werner_view(i, j, a, b);
                        if (a < nocc_ && b < nocc_) {
                            mean_ii += abs(D_Werner_view(i, j, a, b));
                            x_sq_ii += D_Werner_view(i, j, a, b) * D_Werner_view(i, j, a, b);
                        } else if (a >= nocc_ && b >= nocc_) {
                            mean_aa += abs(D_Werner_view(i, j, a, b));
                            x_sq_aa += D_Werner_view(i, j, a, b) * D_Werner_view(i, j, a, b);
                        } else {
                            mean_ia += abs(D_Werner_view(i, j, a, b));
                            x_sq_ia += D_Werner_view(i, j, a, b) * D_Werner_view(i, j, a, b);
                        }
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * (nobs_-nfrzn_) * (nobs_-nfrzn_);
        mean_ii /= nact_ * nact_ * nact_ * nact_;
        mean_ia /= nact_ * nact_ * nact_ * nvir_;
        mean_aa /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * (nobs_-nfrzn_) * (nobs_-nfrzn_);
        x_sq_ii /= nact_ * nact_ * nact_ * nact_;
        x_sq_ia /= nact_ * nact_ * nact_ * nvir_;
        x_sq_aa /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        outfile->Printf("      Mean Value (ii): %e, Standard Deviation: %e\n", mean_ii, sqrt(x_sq_ii - mean_ii * mean_ii));
        outfile->Printf("      Mean Value (ia): %e, Standard Deviation: %e\n", mean_ia, sqrt(x_sq_ia - mean_ia * mean_ia));
        outfile->Printf("      Mean Value (aa): %e, Standard Deviation: %e\n", mean_aa, sqrt(x_sq_aa - mean_aa * mean_aa));
        timer_off("D (from Werner 1992) Intermediate");//*/

        /*
        outfile->Printf("   B Intermediate\n");
        if (!(*B).existed()) {
            timer_on("B Intermediate");
            form_B(B.get(), Uf.get(), F2.get(), F.get(), f.get(), fk.get(), k.get());
            timer_off("B Intermediate");
        }
        */

        /*
        //Check if KD == K\tau + KE + KE*
        auto K_rsab = (*K)(Range{nfrzn_, nobs_}, Range{nfrzn_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_}); K_rsab.set_read_only(true);
        DiskView<double, 4, 4> K_cdab{(*K), Dim<4>{nvir_, nvir_, nvir_, nvir_}, Count<4>{nvir_, nvir_, nvir_, nvir_}, Offset<4>{nocc_, nocc_, nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; K_cdab.set_read_only(true);
        DiskView<double, 4, 4> K_icab{(*K), Dim<4>{nact_, nvir_, nvir_, nvir_}, Count<4>{nact_, nvir_, nvir_, nvir_}, Offset<4>{nfrzn_, nocc_, nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; K_icab.set_read_only(true);
        DiskView<double, 4, 4> K_cjab{(*K), Dim<4>{nvir_, nact_, nvir_, nvir_}, Count<4>{nvir_, nact_, nvir_, nvir_}, Offset<4>{nocc_, nfrzn_, nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; K_cjab.set_read_only(true);
        DiskView<double, 4, 4> tau_ijab_view{(*tau), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; tau_ijab_view.set_read_only(true);
        DiskView<double, 2, 2> t_ic_view{(*t_ia), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_ic_view.set_read_only(true);
        auto resA = std::make_unique<DiskTensor<double, 4>>(state::data(), "Result of KD", nact_, nact_, nvir_, nvir_);
        auto resB = std::make_unique<DiskTensor<double, 4>>(state::data(), "Result of Ktau KE KE", nact_, nact_, nvir_, nvir_);
        using namespace einsums;
        using namespace tensor_algebra;
        using namespace tensor_algebra::index;
        DiskView<double, 4, 4> resA_view{(*resA), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; resA_view.set_read_only(true);
        DiskView<double, 4, 4> resB_view{(*resB), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; resB_view.set_read_only(true);
        DiskView<double, 4, 4> resC_view{(*resB), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; resC_view.set_read_only(true);
        
        einsum(0.0, Indices{i, j, a, b}, &resA_view.get(), 1.0, Indices{p, q, a, b}, K_rsab.get(), Indices{i, j, p, q}, D_Werner_view.get());
        einsum(0.0, Indices{i, j, a, b}, &resB_view.get(), 1.0, Indices{c, d, a, b}, K_cdab.get(), Indices{i, j, c, d}, tau_ijab_view.get());
        sort(0.0, Indices{i, j, a, b}, &resC_view.get(), 1.0, Indices{i, j, a, b}, resB_view.get());
        einsum(1.0, Indices{i, j, a, b}, &resB_view.get(), 1.0, Indices{i, c, a, b}, K_icab.get(), Indices{j, c}, t_ic_view.get());
        einsum(1.0, Indices{i, j, a, b}, &resB_view.get(), 1.0, Indices{c, j, a, b}, K_cjab.get(), Indices{i, c}, t_ic_view.get());
        //Compare
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        if (abs(resA_view.get()(i, j, a, b) - resB_view.get()(i, j, a, b)) > 1.0e-10) {
                            outfile->Printf("   KD != Ktau + KE + KE* at (%d, %d, %d, %d)\n", i, j, a, b);
                            outfile->Printf("      KD: %e, Ktau + KE + KE*: %e\n", resA_view.get()(i, j, a, b), resB_view.get()(i, j, a, b));
                            outfile->Printf("      Delta: %e\n", abs(resA_view.get()(i, j, a, b) - resB_view.get()(i, j, a, b)));
                            outfile->Printf("      resC: %e, t_ia: %e, resB - resC: %e\n", resC_view.get()(i, j, a, b), t_ic_view.get()(i, a), resB_view.get()(i, j, a, b) - resC_view.get()(i, j, a, b));
                        }
                    }
                }
            }
        }*/

        /*
        outfile->Printf("   Constant matrix beta\n");
        timer_on("Constant matrix beta");
        form_beta(beta.get(), f.get(), t_ia.get(), tau.get(), L_oovv.get(), L_ooov.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 2, 2> beta_view{(*beta), Dim<2>{nact_, nact_}, Count<2>{nact_, nact_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; beta_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                meanValue += abs(beta_view(i, j));
                x_squared += beta_view(i, j) * beta_view(i, j);
            }
        }
        meanValue /= nact_ * nact_;
        x_squared /= nact_ * nact_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("Constant matrix beta");

        outfile->Printf("   A Intermediate\n");
        timer_on("A Intermediate");
        form_A(A.get(), T_ijab.get(), L_oovv.get());
        meanValue = 0.0;
        x_squared = 0.0;
        auto A_view = (*A)(All, All); A_view.set_read_only(true);
        for (int a = 0; a < nvir_; a++) {
            for (int b = 0; b < nvir_; b++) {
                meanValue += abs(A_view(a, b));
                x_squared += A_view(a, b) * A_view(a, b);
            }
        }
        meanValue /= nvir_ * nvir_;
        x_squared /= nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("A Intermediate");

        outfile->Printf("   s Intermediate\n");
        timer_on("s Intermediate");
        form_s(s.get(), T_ijab.get(), f.get(), t_ia.get(), A.get(), D_Werner.get(), K.get(), L_ooov.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 2, 2> s_view{(*s), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; s_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int a = 0; a < nvir_; a++) {
                meanValue += abs(s_view(i, a));
                x_squared += s_view(i, a) * s_view(i, a);
            }
        }
        meanValue /= nact_ * nvir_;
        x_squared /= nact_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("s Intermediate");

        outfile->Printf("   r Intermediate\n");
        timer_on("r Intermediate");
        form_r(r.get(), f.get(), t_ia.get(), L_oovv.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 2, 2> r_view{(*r), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; r_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int a = 0; a < nvir_; a++) {
                meanValue += abs(r_view(i, a));
                x_squared += r_view(i, a) * r_view(i, a);
            }
        }
        meanValue /= nact_ * nvir_;
        x_squared /= nact_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("r Intermediate");//*/

        outfile->Printf("   T1 Residual (v) Tensor\n");
        timer_on("T1 Residual (v) Tensor");
        //form_v_ia(v_ia.get(), T_ijab.get(), t_ia.get(), beta.get(), r.get(), s.get());
        form_Scuseria_via(v_ia.get(), T_ijab.get(), t_ia.get(), f.get(), J.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 2, 2> v_ia_view{(*v_ia), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; v_ia_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int a = 0; a < nvir_; a++) {
                meanValue += abs(v_ia_view(i, a));
                x_squared += v_ia_view(i, a) * v_ia_view(i, a);
            }
        }
        meanValue /= nact_ * nvir_;
        x_squared /= nact_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("T1 Residual (v) Tensor");

        outfile->Printf("   Update T1\n");
        timer_on("Update T1");
        update_t1(t_ia.get(), v_ia.get(), D_ia.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 2, 2> t_ia_view{(*t_ia), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_ia_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int a = 0; a < nvir_; a++) {
                meanValue += abs(t_ia_view(i, a));
                x_squared += t_ia_view(i, a) * t_ia_view(i, a);
                //Check if t_ia is real and appropriately sized
                if (abs(t_ia_view(i, a)) < 1.0e-40 || isnan((double)t_ia_view(i, a))) {
                    outfile->Printf("      Unusual Value for t: %e at (%d, %d)\n", t_ia_view(i, a), i, a);
                }
            }
        }
        meanValue /= nact_ * nvir_;
        x_squared /= nact_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("Update T1");

        outfile->Printf("   Update Tau and Tau Tilde (as t1 is updated)\n");
        timer_on("Update Tau and Tau Tilde");
        form_tau(tau.get(), T_ijab.get(), t_ia.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> tau_view{(*tau), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; tau_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(tau_view(i, j, a, b));
                        x_squared += tau_view(i, j, a, b) * tau_view(i, j, a, b);

                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        form_taut(taut.get(), T_ijab.get(), t_ia.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> taut_view{(*taut), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; taut_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(taut_view(i, j, a, b));
                        x_squared += taut_view(i, j, a, b) * taut_view(i, j, a, b);
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("Update Tau and Tau Tilde");
        /*

        outfile->Printf("   D (from Werner 1992) Intermediate\n");
        timer_on("D (from Werner 1992) Intermediate");
        form_D_Werner(D_Werner.get(), tau.get(), t_ia.get());
        meanValue = 0.0;
        mean_ii = 0.0;
        mean_ia = 0.0;
        mean_aa = 0.0;
        x_squared = 0.0;
        x_sq_ii = 0.0;
        x_sq_ia = 0.0;
        x_sq_aa = 0.0;
        DiskView<double, 4, 4> D_Werner_view_b{(*D_Werner), Dim<4>{nact_, nact_, nobs_-nfrzn_, nobs_-nfrzn_}, Count<4>{nact_, nact_, nobs_-nfrzn_, nobs_-nfrzn_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; D_Werner_view_b.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nobs_-nfrzn_; a++) {
                    for (int b = 0; b < nobs_-nfrzn_; b++) {
                        meanValue += abs(D_Werner_view_b(i, j, a, b));
                        x_squared += D_Werner_view_b(i, j, a, b) * D_Werner_view_b(i, j, a, b);
                        if (a < nocc_ && b < nocc_) {
                            mean_ii += abs(D_Werner_view_b(i, j, a, b));
                            x_sq_ii += D_Werner_view_b(i, j, a, b) * D_Werner_view_b(i, j, a, b);
                        } else if (a >= nocc_ && b >= nocc_) {
                            mean_aa += abs(D_Werner_view_b(i, j, a, b));
                            x_sq_aa += D_Werner_view_b(i, j, a, b) * D_Werner_view_b(i, j, a, b);
                        } else {
                            mean_ia += abs(D_Werner_view_b(i, j, a, b));
                            x_sq_ia += D_Werner_view_b(i, j, a, b) * D_Werner_view_b(i, j, a, b);
                        }
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * (nobs_-nfrzn_) * (nobs_-nfrzn_);
        mean_ii /= nact_ * nact_ * nact_ * nact_;
        mean_ia /= nact_ * nact_ * nact_ * nvir_;
        mean_aa /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * (nobs_-nfrzn_) * (nobs_-nfrzn_);
        x_sq_ii /= nact_ * nact_ * nact_ * nact_;
        x_sq_ia /= nact_ * nact_ * nact_ * nvir_;
        x_sq_aa /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        outfile->Printf("      Mean Value (ii): %e, Standard Deviation: %e\n", mean_ii, sqrt(x_sq_ii - mean_ii * mean_ii));
        outfile->Printf("      Mean Value (ia): %e, Standard Deviation: %e\n", mean_ia, sqrt(x_sq_ia - mean_ia * mean_ia));
        outfile->Printf("      Mean Value (aa): %e, Standard Deviation: %e\n", mean_aa, sqrt(x_sq_aa - mean_aa * mean_aa));
        timer_off("D (from Werner 1992) Intermediate");

        outfile->Printf("   Constant matrix beta\n");
        timer_on("Constant matrix beta");
        form_beta(beta.get(), f.get(), t_ia.get(), tau.get(), L_oovv.get(), L_ooov.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 2, 2> beta_view_b{(*beta), Dim<2>{nact_, nact_}, Count<2>{nact_, nact_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; beta_view_b.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                meanValue += abs(beta_view_b(i, j));
                x_squared += beta_view_b(i, j) * beta_view_b(i, j);
            }
        }
        meanValue /= nact_ * nact_;
        x_squared /= nact_ * nact_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("Constant matrix beta");

        outfile->Printf("   s Intermediate\n");
        timer_on("s Intermediate");
        form_s(s.get(), T_ijab.get(), f.get(), t_ia.get(), A.get(), D_Werner.get(), K.get(), L_ooov.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 2, 2> s_view_b{(*s), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; s_view_b.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int a = 0; a < nvir_; a++) {
                meanValue += abs(s_view_b(i, a));
                x_squared += s_view_b(i, a) * s_view_b(i, a);
            }
        }
        meanValue /= nact_ * nvir_;
        x_squared /= nact_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("s Intermediate");

        outfile->Printf("   r Intermediate\n");
        timer_on("r Intermediate");
        form_r(r.get(), f.get(), t_ia.get(), L_oovv.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 2, 2> r_view_b{(*r), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; r_view_b.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int a = 0; a < nvir_; a++) {
                meanValue += abs(r_view_b(i, a));
                x_squared += r_view_b(i, a) * r_view_b(i, a);
            }
        }
        meanValue /= nact_ * nvir_;
        x_squared /= nact_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("r Intermediate");//*/

        outfile->Printf("   Check J MO Tensor (pppp)\n");
        meanValue = 0.0;
        double mean_iiii = 0.0;
        double mean_iiia = 0.0;
        double mean_iiai = 0.0;
        double mean_iaii = 0.0;
        double mean_aiii = 0.0;
        double mean_iiaa = 0.0;
        double mean_aaii = 0.0;
        double mean_aiai = 0.0;
        double mean_iaia = 0.0;
        double mean_iaai = 0.0;
        double mean_aiia = 0.0;
        double mean_iaaa = 0.0;
        double mean_aiaa = 0.0;
        double mean_aaia = 0.0;
        double mean_aaai = 0.0;
        double mean_aaaa = 0.0;
        x_squared = 0.0;
        double x_sq_iiii = 0.0;
        double x_sq_iiia = 0.0;
        double x_sq_iiai = 0.0;
        double x_sq_iaii = 0.0;
        double x_sq_aiii = 0.0;
        double x_sq_iiaa = 0.0;
        double x_sq_aaii = 0.0;
        double x_sq_aiai = 0.0;
        double x_sq_iaia = 0.0;
        double x_sq_iaai = 0.0;
        double x_sq_aiia = 0.0;
        double x_sq_iaaa = 0.0;
        double x_sq_aiaa = 0.0;
        double x_sq_aaia = 0.0;
        double x_sq_aaai = 0.0;
        double x_sq_aaaa = 0.0;
        auto J_view = (*J)(All, All, All, All); J_view.set_read_only(true);
        bool J_pqrs_pqsr = true;
        bool J_pqrs_psrq = true;
        bool J_pqrs_prqs = true;
        bool J_pqrs_qprs = true;
        bool J_pqrs_qpsr = true;
        bool J_pqrs_rspq = true;
        bool J_pqrs_rqps = true;
        bool J_pqrs_sqrp = true;
        for (int i = 0; i < nobs_; i++) {
            for (int q = 0; q < nobs_; q++) {
                for (int r = 0; r < nobs_; r++) {
                    for (int l = 0; l < nobs_; l++) {
                        meanValue += abs(J_view(i, q, r, l));
                        x_squared += J_view(i, q, r, l) * J_view(i, q, r, l);
                        if (i < nocc_ && q < nocc_ && r < nocc_ && l < nocc_) {
                            mean_iiii += abs(J_view(i, q, r, l));
                            x_sq_iiii += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, q, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_pqsr = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(i, l, r, q)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_psrq = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(i, r, q, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_prqs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, r, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qprs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qpsr = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(r, q, i, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_rqps = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(l, q, r, i)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_sqrp = false;
                            }
                        } else if (i < nocc_ && q < nocc_ && r < nocc_ && l >= nocc_) {
                            mean_iiia += abs(J_view(i, q, r, l));
                            x_sq_iiia += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, r, q, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_prqs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, r, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qprs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(r, q, i, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_rqps = false;
                            }
                        } else if (i < nocc_ && q < nocc_ && r >= nocc_ && l < nocc_) {
                            mean_iiai += abs(J_view(i, q, r, l));
                            x_sq_iiai += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, l, r, q)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_psrq = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, r, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qprs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(l, q, r, i)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_sqrp = false;
                            }
                        } else if (i < nocc_ && q >= nocc_ && r < nocc_ && l < nocc_) {
                            mean_iaii += abs(J_view(i, q, r, l));
                            x_sq_iaii += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, q, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_pqsr = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(r, q, i, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_rqps = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(l, q, r, i)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_sqrp = false;
                            }
                        } else if (i >= nocc_ && q < nocc_ && r < nocc_ && l < nocc_) {
                            mean_aiii += abs(J_view(i, q, r, l));
                            x_sq_aiii += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, q, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_pqsr = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(i, l, r, q)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_psrq = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(i, r, q, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_prqs = false;
                            }
                        } else if (i < nocc_ && q >= nocc_ && r >= nocc_ && l >= nocc_) {
                            mean_iaaa += abs(J_view(i, q, r, l));
                            x_sq_iaaa += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, q, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_pqsr = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(i, l, r, q)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_psrq = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(i, r, q, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_prqs = false;
                            }
                        } else if (i >= nocc_ && q < nocc_ && r >= nocc_ && l >= nocc_) {
                            mean_aiaa += abs(J_view(i, q, r, l));
                            x_sq_aiaa += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, q, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_pqsr = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(r, q, i, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_rqps = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(l, q, r, i)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_sqrp = false;
                            }
                        } else if (i >= nocc_ && q >= nocc_ && r < nocc_ && l >= nocc_) {
                            mean_aaia += abs(J_view(i, q, r, l));
                            x_sq_aaia += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, l, r, q)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_psrq = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, r, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qprs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(l, q, r, i)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_sqrp = false;
                            }
                        } else if (i >= nocc_ && q >= nocc_ && r >= nocc_ && l < nocc_) {
                            mean_aaai += abs(J_view(i, q, r, l));
                            x_sq_aaai += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, r, q, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_prqs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, r, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qprs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(r, q, i, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_rqps = false;
                            }
                        } else if (i >= nocc_ && q >= nocc_ && r >= nocc_ && l >= nocc_) {
                            mean_aaaa += abs(J_view(i, q, r, l));
                            x_sq_aaaa += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, q, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_pqsr = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(i, l, r, q)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_psrq = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(i, r, q, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_prqs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, r, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qprs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qpsr = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(r, l, i, q)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_rspq = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(r, q, i, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_rqps = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(l, q, r, i)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_sqrp = false;
                            }
                        } else if (i >= nocc_ && q >= nocc_ && r < nocc_ && l < nocc_) {
                            mean_aaii += abs(J_view(i, q, r, l));
                            x_sq_aaii += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, q, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_pqsr = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, r, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qprs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qpsr = false;
                            }
                        } else if (i >= nocc_ && q < nocc_ && r < nocc_ && l >= nocc_) {
                            mean_aiia += abs(J_view(i, q, r, l));
                            x_sq_aiia += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, r, q, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_prqs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(l, q, r, i)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_sqrp = false;
                            }
                        } else if (i < nocc_ && q >= nocc_ && r >= nocc_ && l < nocc_) {
                            mean_iaai += abs(J_view(i, q, r, l));
                            x_sq_iaai += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, r, q, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_prqs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(l, q, r, i)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_sqrp = false;
                            }
                        } else if (i >= nocc_ && q < nocc_ && r >= nocc_ && l < nocc_) {
                            mean_aiai += abs(J_view(i, q, r, l));
                            x_sq_aiai += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, l, r, q)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_psrq = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(r, q, i, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_rqps = false;
                            }
                        } else if (i < nocc_ && q >= nocc_ && r < nocc_ && l >= nocc_) {
                            mean_iaia += abs(J_view(i, q, r, l));
                            x_sq_iaia += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, l, r, q)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_psrq = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(r, q, i, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_rqps = false;
                            }
                        } else {
                            mean_iiaa += abs(J_view(i, q, r, l));
                            x_sq_iiaa += J_view(i, q, r, l) * J_view(i, q, r, l);
                            if (abs(J_view(i, q, r, l) - J_view(i, q, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_pqsr = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, r, l)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qprs = false;
                            }
                            if (abs(J_view(i, q, r, l) - J_view(q, i, l, r)) > ((1.0e-5)*J_view(i, q, r, l))) {
                                J_pqrs_qpsr = false;
                            }
                        }
                    }
                }
            }
        }
        outfile->Printf("      J_pqrs_pqsr: %s\n", J_pqrs_pqsr ? "true" : "false");
        outfile->Printf("      J_pqrs_psqr: %s\n", J_pqrs_psqr ? "true" : "false");
        outfile->Printf("      J_pqrs_psrq: %s\n", J_pqrs_psrq ? "true" : "false");
        outfile->Printf("      J_pqrs_prqs: %s\n", J_pqrs_prqs ? "true" : "false");
        outfile->Printf("      J_pqrs_prsq: %s\n", J_pqrs_prsq ? "true" : "false");
        outfile->Printf("      J_pqrs_qprs: %s\n", J_pqrs_qprs ? "true" : "false");
        outfile->Printf("      J_pqrs_qpsr: %s\n", J_pqrs_qpsr ? "true" : "false");
        outfile->Printf("      J_pqrs_qrps: %s\n", J_pqrs_qrps ? "true" : "false");
        outfile->Printf("      J_pqrs_qrsp: %s\n", J_pqrs_qrsp ? "true" : "false");
        outfile->Printf("      J_pqrs_qspr: %s\n", J_pqrs_qspr ? "true" : "false");
        outfile->Printf("      J_pqrs_qsrp: %s\n", J_pqrs_qsrp ? "true" : "false");
        outfile->Printf("      J_pqrs_rspq: %s\n", J_pqrs_rspq ? "true" : "false");
        outfile->Printf("      J_pqrs_rsqp: %s\n", J_pqrs_rsqp ? "true" : "false");
        outfile->Printf("      J_pqrs_rpqs: %s\n", J_pqrs_rpqs ? "true" : "false");
        outfile->Printf("      J_pqrs_rqps: %s\n", J_pqrs_rqps ? "true" : "false");
        outfile->Printf("      J_pqrs_rpsq: %s\n", J_pqrs_rpsq ? "true" : "false");
        outfile->Printf("      J_pqrs_rqsp: %s\n", J_pqrs_rqsp ? "true" : "false");
        outfile->Printf("      J_pqrs_sqpr: %s\n", J_pqrs_sqpr ? "true" : "false");
        outfile->Printf("      J_pqrs_spqr: %s\n", J_pqrs_spqr ? "true" : "false");
        outfile->Printf("      J_pqrs_sqrp: %s\n", J_pqrs_sqrp ? "true" : "false");
        outfile->Printf("      J_pqrs_sprq: %s\n", J_pqrs_sprq ? "true" : "false");
        outfile->Printf("      J_pqrs_srqp: %s\n", J_pqrs_srqp ? "true" : "false");
        outfile->Printf("      J_pqrs_srpq: %s\n", J_pqrs_srpq ? "true" : "false");
        
        meanValue /= nobs_ * nobs_ * nobs_ * nobs_;
        mean_iiii /= nocc_ * nocc_ * nocc_ * nocc_;
        mean_iiia /= nocc_ * nocc_ * nocc_ * nvir_;
        mean_iiai /= nocc_ * nocc_ * nvir_ * nocc_;
        mean_iaii /= nocc_ * nvir_ * nocc_ * nocc_;
        mean_aiii /= nvir_ * nocc_ * nocc_ * nocc_;
        mean_iiaa /= nocc_ * nocc_ * nvir_ * nvir_;
        mean_aaii /= nvir_ * nocc_ * nocc_ * nvir_;
        mean_aiai /= nvir_ * nocc_ * nvir_ * nocc_;
        mean_iaia /= nocc_ * nvir_ * nocc_ * nvir_;
        mean_iaai /= nocc_ * nvir_ * nvir_ * nocc_;
        mean_aiia /= nvir_ * nocc_ * nocc_ * nvir_;
        mean_iaaa /= nocc_ * nvir_ * nvir_ * nvir_;
        mean_aiaa /= nvir_ * nocc_ * nvir_ * nvir_;
        mean_aaia /= nvir_ * nvir_ * nocc_ * nvir_;
        mean_aaai /= nvir_ * nvir_ * nvir_ * nocc_;
        mean_aaaa /= nvir_ * nvir_ * nvir_ * nvir_;
        x_squared /= nobs_ * nobs_ * nobs_ * nobs_;
        x_sq_iiii /= nocc_ * nocc_ * nocc_ * nocc_;
        x_sq_iiia /= nocc_ * nocc_ * nocc_ * nvir_;
        x_sq_iiai /= nocc_ * nocc_ * nvir_ * nocc_;
        x_sq_iaii /= nocc_ * nvir_ * nocc_ * nocc_;
        x_sq_aiii /= nvir_ * nocc_ * nocc_ * nocc_;
        x_sq_iiaa /= nocc_ * nocc_ * nvir_ * nvir_;
        x_sq_aaii /= nvir_ * nocc_ * nocc_ * nvir_;
        x_sq_aiai /= nvir_ * nocc_ * nvir_ * nocc_;
        x_sq_iaia /= nocc_ * nvir_ * nocc_ * nvir_;
        x_sq_iaai /= nocc_ * nvir_ * nvir_ * nocc_;
        x_sq_aiia /= nvir_ * nocc_ * nocc_ * nvir_;
        x_sq_iaaa /= nocc_ * nvir_ * nvir_ * nvir_;
        x_sq_aiaa /= nvir_ * nocc_ * nvir_ * nvir_;
        x_sq_aaia /= nvir_ * nvir_ * nocc_ * nvir_;
        x_sq_aaai /= nvir_ * nvir_ * nvir_ * nocc_;
        x_sq_aaaa /= nvir_ * nvir_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        outfile->Printf("      Mean Value (iiii): %e, Standard Deviation: %e\n", mean_iiii, sqrt(x_sq_iiii - mean_iiii * mean_iiii));
        outfile->Printf("      Mean Value (iiia): %e, Standard Deviation: %e\n", mean_iiia, sqrt(x_sq_iiia - mean_iiia * mean_iiia));
        outfile->Printf("      Mean Value (iiai): %e, Standard Deviation: %e\n", mean_iiai, sqrt(x_sq_iiai - mean_iiai * mean_iiai));
        outfile->Printf("      Mean Value (iaii): %e, Standard Deviation: %e\n", mean_iaii, sqrt(x_sq_iaii - mean_iaii * mean_iaii));
        outfile->Printf("      Mean Value (aiii): %e, Standard Deviation: %e\n", mean_aiii, sqrt(x_sq_aiii - mean_aiii * mean_aiii));
        outfile->Printf("      Mean Value (iiaa): %e, Standard Deviation: %e\n", mean_iiaa, sqrt(x_sq_iiaa - mean_iiaa * mean_iiaa));
        outfile->Printf("      Mean Value (aaii): %e, Standard Deviation: %e\n", mean_aaii, sqrt(x_sq_aaii - mean_aaii * mean_aaii));
        outfile->Printf("      Mean Value (aiai): %e, Standard Deviation: %e\n", mean_aiai, sqrt(x_sq_aiai - mean_aiai * mean_aiai));
        outfile->Printf("      Mean Value (iaia): %e, Standard Deviation: %e\n", mean_iaia, sqrt(x_sq_iaia - mean_iaia * mean_iaia));
        outfile->Printf("      Mean Value (iaai): %e, Standard Deviation: %e\n", mean_iaai, sqrt(x_sq_iaai - mean_iaai * mean_iaai));
        outfile->Printf("      Mean Value (aiia): %e, Standard Deviation: %e\n", mean_aiia, sqrt(x_sq_aiia - mean_aiia * mean_aiia));
        outfile->Printf("      Mean Value (iaaa): %e, Standard Deviation: %e\n", mean_iaaa, sqrt(x_sq_iaaa - mean_iaaa * mean_iaaa));
        outfile->Printf("      Mean Value (aiaa): %e, Standard Deviation: %e\n", mean_aiaa, sqrt(x_sq_aiaa - mean_aiaa * mean_aiaa));
        outfile->Printf("      Mean Value (aaia): %e, Standard Deviation: %e\n", mean_aaia, sqrt(x_sq_aaia - mean_aaia * mean_aaia));
        outfile->Printf("      Mean Value (aaai): %e, Standard Deviation: %e\n", mean_aaai, sqrt(x_sq_aaai - mean_aaai * mean_aaai));
        outfile->Printf("      Mean Value (aaaa): %e, Standard Deviation: %e\n", mean_aaaa, sqrt(x_sq_aaaa - mean_aaaa * mean_aaaa));

        /*outfile->Printf("   X (from Werner 1992) Intermediate\n");
        timer_on("X (from Werner 1992) Intermediate");
        form_X_Werner(X_Werner.get(), f.get(), t_ia.get(), A.get(), r.get(), J.get(), K.get());
        meanValue = 0.0;
        x_squared = 0.0;
        auto X_Werner_view = (*X_Werner)(All, All); X_Werner_view.set_read_only(true);
        for (int a = 0; a < nvir_; a++) {
            for (int b = 0; b < nvir_; b++) {
                meanValue += abs(X_Werner_view(a, b));
                x_squared += X_Werner_view(a, b) * X_Werner_view(a, b);
            }
        }
        meanValue /= nvir_ * nvir_;
        x_squared /= nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("X (from Werner 1992) Intermediate");

        outfile->Printf("   Y (from Werner 1992) Intermediate\n");
        timer_on("Y (from Werner 1992) Intermediate");
        form_Y_Werner(Y_Werner.get(), taut.get(), t_ia.get(), f.get(), J.get(), K.get(), L_oovv.get(), L_ooov.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> Y_Werner_view{(*Y_Werner), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; Y_Werner_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(Y_Werner_view(i, j, a, b));
                        x_squared += Y_Werner_view(i, j, a, b) * Y_Werner_view(i, j, a, b);
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        timer_off("Y (from Werner 1992) Intermediate");

        outfile->Printf("   Z (from Werner 1992) Intermediate\n");
        timer_on("Z (from Werner 1992) Intermediate");
        form_Z_Werner(Z_Werner.get(), taut.get(), t_ia.get(), J.get(), K.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> Z_Werner_view{(*Z_Werner), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; Z_Werner_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(Z_Werner_view(i, j, a, b));
                        x_squared += Z_Werner_view(i, j, a, b) * Z_Werner_view(i, j, a, b);
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("Z (from Werner 1992) Intermediate");

        outfile->Printf("   Constant matrix alpha\n");
        timer_on("Constant matrix alpha");
        form_alpha(alpha.get(), tau.get(), t_ia.get(), K.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> alpha_view{(*alpha), Dim<4>{nact_, nact_, nact_, nact_}, Count<4>{nact_, nact_, nact_, nact_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; alpha_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int k = 0; k < nact_; k++) {
                    for (int l = 0; l < nact_; l++) {
                        meanValue += abs(alpha_view(i, j, k, l));
                        x_squared += alpha_view(i, j, k, l) * alpha_view(i, j, k, l);
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nact_ * nact_;
        x_squared /= nact_ * nact_ * nact_ * nact_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("Constant matrix alpha");

        outfile->Printf("   G Intermediate\n");
        timer_on("G Intermediate");
        form_G(G_ij.get(), s.get(), t_ia.get(), D_Werner.get(), beta.get(), T_ijab.get(), X_Werner.get(), Y_Werner.get(), Z_Werner.get(), tau.get(), K.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> G_ij_view{(*G_ij), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; G_ij_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(G_ij_view(i, j, a, b));
                        x_squared += G_ij_view(i, j, a, b) * G_ij_view(i, j, a, b);
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("G Intermediate");

        outfile->Printf("   Check F_BraBig MO Tensor (ppii)\n");
        meanValue = 0.0;
        mean_ii = 0.0;
        mean_ia = 0.0;
        double mean_ai = 0.0;
        mean_aa = 0.0;
        x_squared = 0.0;
        x_sq_ii = 0.0;
        x_sq_ia = 0.0;
        double x_sq_ai = 0.0;
        x_sq_aa = 0.0;
        DiskView<double, 4, 4> F_BraBig_view{(*F_BraBig), Dim<4>{nobs_, nobs_, nact_, nact_}, Count<4>{nobs_, nobs_, nact_, nact_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; F_BraBig_view.set_read_only(true);
        for (int i = 0; i < nobs_; i++) {
            for (int j = 0; j < nobs_; j++) {
                for (int a = 0; a < nact_; a++) {
                    for (int b = 0; b < nact_; b++) {
                        meanValue += abs(F_BraBig_view(i, j, a, b));
                        x_squared += F_BraBig_view(i, j, a, b) * F_BraBig_view(i, j, a, b);
                        if (i < nocc_ && j < nocc_) {
                            mean_ii += abs(F_BraBig_view(i, j, a, b));
                            x_sq_ii += F_BraBig_view(i, j, a, b) * F_BraBig_view(i, j, a, b);
                        } else if (a >= nocc_ && b >= nocc_) {
                            mean_aa += abs(F_BraBig_view(i, j, a, b));
                            x_sq_aa += F_BraBig_view(i, j, a, b) * F_BraBig_view(i, j, a, b);
                        } else if (a >= nocc_ && b < nocc_) {
                            mean_ai += abs(F_BraBig_view(i, j, a, b));
                            x_sq_ai += F_BraBig_view(i, j, a, b) * F_BraBig_view(i, j, a, b);
                        } else {
                            mean_ia += abs(F_BraBig_view(i, j, a, b));
                            x_sq_ia += F_BraBig_view(i, j, a, b) * F_BraBig_view(i, j, a, b);
                        }
                    }
                }
            }
        }
        meanValue /= nobs_ * nobs_ * nact_ * nact_;
        mean_ii /= nocc_ * nocc_ * nact_ * nact_;
        mean_ia /= nocc_ * nvir_ * nact_ * nact_;
        mean_ai /= nvir_ * nocc_ * nact_ * nact_;
        mean_aa /= nvir_ * nvir_ * nact_ * nact_;
        x_squared /= nobs_ * nobs_ * nact_ * nact_;
        x_sq_ii /= nocc_ * nocc_ * nact_ * nact_;
        x_sq_ia /= nocc_ * nvir_ * nact_ * nact_;
        x_sq_ai /= nvir_ * nocc_ * nact_ * nact_;
        x_sq_aa /= nvir_ * nvir_ * nact_ * nact_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        outfile->Printf("      Mean Value (ii): %e, Standard Deviation: %e\n", mean_ii, sqrt(x_sq_ii - mean_ii * mean_ii));
        outfile->Printf("      Mean Value (ia): %e, Standard Deviation: %e\n", mean_ia, sqrt(x_sq_ia - mean_ia * mean_ia));
        outfile->Printf("      Mean Value (ai): %e, Standard Deviation: %e\n", mean_ai, sqrt(x_sq_ai - mean_ai * mean_ai));
        outfile->Printf("      Mean Value (aa): %e, Standard Deviation: %e\n", mean_aa, sqrt(x_sq_aa - mean_aa * mean_aa));//*/

        outfile->Printf("   T2 Residual (V) Tensor\n");
        timer_on("T2 Residual (V) Tensor");
        //form_V_ijab(V_ijab.get(), G_ij.get(), tau.get(), D_Werner.get(), alpha.get(), C.get(), FG.get(), K.get(), F_BraBig.get());
        form_Scuseria_Vijab(V_ijab.get(), T_ijab.get(), t_ia.get(), f.get(), J.get(), F_BraBig.get(), FG.get(), C.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> V_ijab_view{(*V_ijab), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; V_ijab_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(V_ijab_view(i, j, a, b));
                        x_squared += V_ijab_view(i, j, a, b) * V_ijab_view(i, j, a, b);
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("T2 Residual (V) Tensor");

        outfile->Printf("   Update T2\n");
        timer_on("Update T2");
        update_t2(T_ijab.get(), V_ijab.get(), D_ijab.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> T_ijab_view{(*T_ijab), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; T_ijab_view.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(T_ijab_view(i, j, a, b));
                        x_squared += T_ijab_view(i, j, a, b) * T_ijab_view(i, j, a, b);
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("Update T2");

        outfile->Printf("   Update Tau and Tau Tilde\n");
        timer_on("Update Tau and Tau Tilde");
        form_tau(tau.get(), T_ijab.get(), t_ia.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> tau_view_b{(*tau), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; tau_view_b.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(tau_view_b(i, j, a, b));
                        x_squared += tau_view_b(i, j, a, b) * tau_view_b(i, j, a, b);
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));

        form_taut(taut.get(), T_ijab.get(), t_ia.get());
        meanValue = 0.0;
        x_squared = 0.0;
        DiskView<double, 4, 4> taut_view_b{(*taut), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; taut_view_b.set_read_only(true);
        for (int i = 0; i < nact_; i++) {
            for (int j = 0; j < nact_; j++) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        meanValue += abs(taut_view_b(i, j, a, b));
                        x_squared += taut_view_b(i, j, a, b) * taut_view_b(i, j, a, b);
                    }
                }
            }
        }
        meanValue /= nact_ * nact_ * nvir_ * nvir_;
        x_squared /= nact_ * nact_ * nvir_ * nvir_;
        outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", meanValue, sqrt(x_squared - meanValue * meanValue));
        timer_off("Update Tau and Tau Tilde");

        outfile->Printf("\n ===> Computing CCSD-F12b Extra Energy Correction <===\n");
        timer_on("F12b Energy Correction");
        form_CCSDF12B_Energy(tau.get(), F_BraBig.get(), FG.get(), J.get());
        timer_off("F12b Energy Correction");

        // Compute the CCSD-F12b Energy
        outfile->Printf("\n ===> Computing CCSD-F12b Energy Correction <===\n");
        timer_on("F12 Energy Correction");
        form_CCSD_Energy(J.get(), V.get(), tau.get());
        timer_off("F12 Energy Correction");

        // Compare the CCSD energies
        timer_on("CCSD Energy Comparison");
        e_diff = std::abs(E_ccsd_ + E_f12b_ - E_old_);
        rms = get_root_mean_square_amplitude_change(t_ia.get(), T_ijab.get(), t_ia_old.get(), T_ijab_old.get());
        auto E_rhf = Process::environment.globals["CURRENT REFERENCE ENERGY"];
        outfile->Printf("\n   Iteration %d: E(RHF) = %.12f; E(CCSD) = %.12f; E(F12b) = %.12f; E(CCSD-F12b) = %.12f\n", iteration_, E_rhf, E_ccsd_, E_f12b_, E_ccsd_ + E_f12b_ + E_rhf);
        outfile->Printf("\n RMS: %.12f; E_diff: %.12f\n", rms, e_diff);

        iteration_++;
        timer_off("CCSD Iteration");
    }

    if (singles_ == true) {
        outfile->Printf("\n ===> Computing CCSD-F12b Singles Correction <===\n");
        timer_on("CABS Singles Correction");
        form_cabs_singles(f.get());
        timer_off("CABS Singles Correction");
    }

    print_results();

    if (print_ > 1) {
        timer::report();
    }
    timer::finalize();

    // Typically you would build a new wavefunction and populate it with data
    return E_f12_;
}

std::pair<double, double> DiskCCSDF12B::V_Tilde(einsums::Tensor<double, 2>& V_ij, einsums::DiskTensor<double, 4> *C,
                                          einsums::DiskView<double, 2, 4>& G_ij, einsums::DiskView<double, 2, 4>& D_ij,
                                          const int& i, const int& j)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    double V_s, V_t;
    int kd;

    {
        Tensor<double, 2> GD{"G_ijab . D_ijab", nvir_, nvir_};
        einsum(Indices{a, b}, &GD, Indices{a, b}, G_ij.get(), Indices{a, b}, D_ij.get());

        Tensor<double, 0> tmp{"tmp"};
        for (int I = 0; I < nact_; I++) {
            for (int J = I; J < nact_; J++) {
                auto C_IJ = (*C)(I, J, All, All); C_IJ.set_read_only(true);
                einsum(Indices{}, &tmp, Indices{a, b}, C_IJ.get(), Indices{a, b}, GD);
                V_ij(I, J) -= tmp;

                if (I != J) {
                    auto C_JI = (*C)(J, I, All, All); C_JI.set_read_only(true);
                    einsum(Indices{}, &tmp, Indices{a, b}, C_JI.get(), Indices{a, b}, GD);
                    V_ij(J, I) -= tmp;
                }
            }
        }
    }

    ( i == j ) ? ( kd = 1 ) : ( kd = 2 );

    V_s = 0.25 * (T_ijkl(i, j, i, j) + T_ijkl(i, j, j, i)) * kd * (V_ij(i, j) + V_ij(j, i));

    if ( i != j ) {
        V_t = 0.25 * (T_ijkl(i, j, i, j) - T_ijkl(i, j, j, i)) * kd * (V_ij(i, j) - V_ij(j, i));
    }
    return {V_s, V_t};
}

std::pair<double, double> DiskCCSDF12B::B_Tilde(einsums::Tensor<double, 4>& B_ij, einsums::DiskTensor<double, 4> *C,
                                          einsums::DiskView<double, 2, 4>& D_ij, const int& i, const int& j)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    double B_s, B_t;
    int kd;

    {
        Tensor<double, 2> rank2{"Contraction 1", nvir_, nvir_};
        Tensor<double, 0> tmp{"Contraction 2"};
        for (int I = 0; I < nact_; I++) {
            for (int J = I; J < nact_; J++) {
                auto C_IJ = (*C)(I, J, All, All); C_IJ.set_read_only(true);
                einsum(Indices{a, b}, &rank2, Indices{a, b}, C_IJ.get(), Indices{a, b}, D_ij.get());
                einsum(Indices{}, &tmp, Indices{a, b}, rank2, Indices{a, b}, C_IJ.get());
                B_ij(I, J, I, J) -= tmp;

                if (I != J) {
                    auto C_JI = (*C)(J, I, All, All); C_JI.set_read_only(true);
                    einsum(Indices{}, &tmp, Indices{a, b}, rank2, Indices{a, b}, C_JI.get());
                    B_ij(I, J, J, I) -= tmp;
                }
            }
        }
    }

    ( i == j ) ? ( kd = 1 ) : ( kd = 2 );

    B_s = 0.125 * (T_ijkl(i, j, i, j) + T_ijkl(i, j, j, i)) * kd
                 * (B_ij(i, j, i, j) + B_ij(i, j, j, i))
                 * (T_ijkl(i, j, i, j) + T_ijkl(i, j, j, i)) * kd;

    if ( i != j ) {
        B_t = 0.125 * (T_ijkl(i, j, i, j) - T_ijkl(i, j, j, i)) * kd
                     * (B_ij(i, j, i, j) - B_ij(i, j, j, i))
                     * (T_ijkl(i, j, i, j) - T_ijkl(i, j, j, i)) * kd;
    }
    return {B_s, B_t};
}

void DiskCCSDF12B::form_CCSDF12B_Energy(einsums::DiskTensor<double, 4> *tau, einsums::DiskTensor<double, 4> *F, 
                                      einsums::DiskTensor<double, 4> *FG, einsums::DiskTensor<double, 4> *K)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    E_f12b_ = 0.0;

    DiskView<double, 4, 4> tau_view{(*tau), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; tau_view.set_read_only(true);
    Tensor<double, 4> E_F12b_temp{"E_F12b_temp", nact_, nact_, nact_, nact_};
    
    size_t block_size = static_cast<size_t>(std::sqrt((((memory_ * 0.25)/ double_memory_) / nvir_) / nvir_)); // Assume memory has been used elsewhere!
    if (block_size > nvir_) block_size = nvir_;
    int last_block = static_cast<int>(nvir_ % block_size);
    int no_blocks = static_cast<int>((nvir_ / block_size) + 1);
    outfile->Printf("Block Size: %d, Last Block: %d, Number of Blocks: %d\n", block_size, last_block, no_blocks);

#pragma omp parallel for schedule(dynamic) collapse(2) num_threads(nthreads_)
    for (int k = 0; k < nact_; k++) {
        for (int l = 0; l < nact_; l++) {
            //outfile->Printf("Iteration %d, %d", k, l);
            Tensor <double, 2> tmp_kl{"Tmp kl", nvir_, nvir_};
            auto F_pqoo_kl = (*F)(Range{nfrzn_, nobs_}, Range{nfrzn_, nobs_}, k, l); F_pqoo_kl.set_read_only(true);
            for (int a_block = 0; a_block < no_blocks; a_block++) {
                //outfile->Printf("Block %d\n", a_block);
                for (int b_block = 0; b_block < no_blocks; b_block++) {
                    int a_start = a_block * block_size;
                    int a_end = (a_block == no_blocks - 1) ? a_start + last_block : a_start + block_size;
                    int b_start = b_block * block_size;
                    int b_end = (b_block == no_blocks - 1) ? b_start + last_block : b_start + block_size;

                    auto W_vvoo_ab = (*FG)(Range{a_start + nocc_, a_end + nocc_}, Range{b_start + nocc_, b_end + nocc_}, k + nfrzn_, l + nfrzn_); W_vvoo_ab.set_read_only(true);
                    auto K_vvpq_ab = (*K)(Range{a_start + nocc_, a_end + nocc_}, Range{b_start + nocc_, b_end + nocc_}, Range{nfrzn_, nobs_}, Range{nfrzn_, nobs_}); K_vvpq_ab.set_read_only(true); // K here is not K in Werner et al. - Supplied J_pqrs
                    auto tmp_kl_ab = tmp_kl(Range{a_start, a_end}, Range{b_start, b_end});

                    einsum(0.0, Indices{a, b}, &tmp_kl_ab, -1.0, Indices{a, b, r, s}, K_vvpq_ab.get(), Indices{r, s}, F_pqoo_kl.get());
                    sort(1.0, Indices{a, b}, &tmp_kl_ab, 1.0, Indices{a, b}, W_vvoo_ab.get());
                }
            }
            /*if (iteration_ < 2) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        outfile->Printf("F12b Temp: %d, %d, %d, %d = %e, ", k, l, a, b, tmp_kl(a, b));
                    }
                    outfile->Printf("\n");
                    for (int b = 0; b < nvir_; b++) {
                        outfile->Printf("D_ijab: %d, %d, %d, %d = %e, ", k, l, a, b, D_ijab_view(k, l, a, b));
                    }
                    outfile->Printf("\n");
                }
            }*/
            auto E_f12b_temp_kl = E_F12b_temp(All, All, k, l);
            einsum(1.0, Indices{i, j}, &E_f12b_temp_kl, 1.0, Indices{a, b}, tmp_kl, Indices{j, i, a, b}, tau_view.get());
        }
    }

    for (int i = 0; i < nact_; i++) {
        for (int j = 0; j < nact_; j++) {
            E_f12b_ += T_ijkl(i, j, j, i) * E_F12b_temp(i, j, j, i);
            if (i != j) E_f12b_ += T_ijkl(i, j, i, j) * E_F12b_temp(i, j, i, j);
        }
    }
}

void DiskCCSDF12B::form_CCSD_Energy(einsums::DiskTensor<double, 4> *G, einsums::DiskTensor<double, 4> *V, einsums::DiskTensor<double, 4> *tau)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    /* N.B. G_ijab == K_ijab in Alder papers */

    E_ccsd_ = 0.0;

    Tensor<double, 0> E_ccsd_tmp{"CCSD Energy"};
    Tensor<double, 4> tmp{"tmp", nact_, nact_, nvir_, nvir_};
    DiskView<double, 4, 4> tau_view{(*tau), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; tau_view.set_read_only(true);
    DiskView<double, 4, 4> G_ijab_view{(*G), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{nfrzn_, nfrzn_, nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; G_ijab_view.set_read_only(true);

    /* K_ijab -> Correct Implementation */
    sort(0.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, tau_view.get());
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, a, b}, tau_view.get());
    einsum(0.0, Indices{}, &E_ccsd_tmp, 1.0, Indices{i, j, a, b}, G_ijab_view.get(), Indices{i, j, a, b}, tmp);

    // Check the CCSD Energy is Expected
    double runnningTotal = 0.0;
    for (int i = 0; i < nact_; i++) {
        for (int j = 0; j < nact_; j++) {
            for (int a = 0; a < nvir_; a++) {
                for (int b = 0; b < nvir_; b++) {
                    runnningTotal += G_ijab_view(i, j, a, b) * 2.0 * tau_view(i, j, a, b);
                    runnningTotal -= G_ijab_view(i, j, a, b) * tau_view(j, i, a, b);
                }
            }
        }
    }
    double check = E_ccsd_tmp;
    outfile->Printf("   CCSD Energy via running total: %.12f\n", runnningTotal);
    outfile->Printf("   CCSD Energy via einsums: %.12f\n", check);

    DiskView<double, 4, 4> V_view{(*V), Dim<4>{nact_, nact_, nact_, nact_}, Count<4>{nact_, nact_, nact_, nact_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; V_view.set_read_only(true);
    
    for (int i = 0; i < nact_; i++) {
        for (int j = 0; j < nact_; j++) {
            for (int k = 0; k < nact_; k++) {
                for (int l = 0; l < nact_; l++) {
                    E_ccsd_tmp += ((2 * T_ijkl(i, j, k, l)) - T_ijkl(i, j, l, k)) * V_view(i, j, k, l);
                }
            }
            /*if (i != j) {
                E_ccsd_tmp += ((2 * T_ijkl(i, j, j, i)) - T_ijkl(i, j, i, j)) * V_view(i, j, j, i);
                E_ccsd_tmp += ((2 * T_ijkl(j, i, j, i)) - T_ijkl(j, i, i, j)) * V_view(j, i, j, i);
                E_ccsd_tmp += ((2 * T_ijkl(j, i, i, j)) - T_ijkl(j, i, j, i)) * V_view(j, i, i, j);
            }*/
        }
    }

    E_ccsd_ = E_ccsd_tmp;

}

}} // End namespaces
