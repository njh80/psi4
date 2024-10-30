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

#ifndef CCSDF12B_H
#define CCSDF12B_H

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/process.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libqt/qt.h"

#include "psi4/libmints/orbitalspace.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/liboptions/liboptions.h"

#include "einsums.hpp"

namespace psi { namespace ccsd_f12b {

class CCSDF12B : public Wavefunction {
   public: 
    CCSDF12B(SharedWavefunction reference_wavefunction, Options& options);
    ~CCSDF12B() override;

    /* Compute the total CCSD-F12B Energy */
    virtual double compute_energy();

   protected:
    /* Print level */
    int print_;

    /* Number of OMP_THREADS */
    int nthreads_;

    /* Choose to compute CABS Singles correction */
    bool singles_;

    /* Choose CONV or DF F12 computation */
    std::string f12_type_;

    /* Density-fitting Basis Set (DFBS) */
    std::shared_ptr<BasisSet> DFBS_;

    /* Bool to turn on DF */
    bool use_df_ = false;

    /* Bool to read in precomputed F12 integrals */
    bool f12_read_ints_ = false;

    /* Root Mean Square difference tolerance between amplitudes 
       (equivalent to convergence in ccenergy) */
    double rms_tol_;

    /* Energy tolerance (equivalent to e_convergence in ccenergy) */
    double e_tol_;

    /* Maximum number of iterations */
    int maxiter_;

    /* List of orbital spaces: Orbital Basis Set (OBS) 
       and Complimentary Auxiliary Basis Set (CABS) */
    std::vector<OrbitalSpace> bs_;

    /* Number of basis functions in OBS */
    int nobs_;

    /* Number of basis functions in CABS */
    int ncabs_;

    /* Number of basis functions in total */
    int nri_;

    /* Number of occupied orbitals */
    int nocc_;

    /* Number of virtual orbitals */
    int nvir_;

    /* Number of basis functions in DFBS */
    int naux_;

    /* Number of frozen core orbitals */
    int nfrzn_;

    /* Number of active orbitals */
    int nact_;

    /* F12 Correlation Factor, Contracted Gaussian-Type Geminal */
    std::vector<std::pair<double, double>> cgtg_;

    /* $\beta$ in F12 CGTG */
    double beta_;

    /* CCSD Energy */
    double E_ccsd_ = 0.0;

    /* CABS Singles Correction */
    double E_singles_ = 0.0;

    /* CCSD-F12b Energy Correction (Additional Energy term from Eq. 11 of Adler 2007) */
    double E_f12b_ = 0.0;

    /* CCSD Energy from Mitchell Implementation (Unused Here) */
    double E_f12_ = 0.0;

    /* Total CCSD-F12b Energy */
    double E_ccsdf12b_ = 0.0;

    void common_init();

    void print_header();

    /* Initialise amplitudes t_ia, T_ijab, T_ijkl */
    void initialise_amplitudes(einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 4> *T_ijab, 
                               einsums::Tensor<double, 4> *V_ijab, einsums::Tensor<double, 4> *D_iajb);

    /* Save the amplitudes between iterations */
    void save_amplitudes(einsums::Tensor<double, 2> *t_ia_old, einsums::Tensor<double, 4> *T_ijab_old, einsums::Tensor<double, 2> *t_ia, 
                         einsums::Tensor<double, 4> *T_ijab);

    /* Update t_ia to be v_ia / D_ij */
    void update_t1(einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 2> *v_ia, einsums::Tensor<double, 2> *D_ia);
    
    /* Update T_ijkl to be -V_ikjl / D_ijkl */
    void update_t2(einsums::Tensor<double, 4> *T_ijkl, einsums::Tensor<double, 4> *V_ikjl, einsums::Tensor<double, 4> *D_ijkl);

    /* Form $\tau^{ij}_{ab}$ = $T^{ij}_{ab}$ + $t^{i}_{a}$ * $t^{j}_{b}$ */
    void form_tau(einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *t_ia);

    /* Form $\Tilde\tau^{ij}_{ab}$ = (0.5 * $T^{ij}_{ab}$) + $t^{i}_{a}$ * $t^{j}_{b}$ */
    void form_taut(einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *t_ia);

    /* Form the basis sets OBS and CABS */
    void form_basissets();

    /* Form the energy denominator */
    virtual void form_D_ia(einsums::Tensor<double, 2> *D, einsums::Tensor<double, 2> *f);

    /* Form the energy denominator */
    virtual void form_D_ijab(einsums::Tensor<double, 4> *D, einsums::Tensor<double, 2> *f);

    /* Form the CABS Singles correction $\frac{|f^{a'}_{i}}|^2}{e_{a'} - e_{i}}$ */
    virtual void form_cabs_singles(einsums::Tensor<double,2> *f);

    /* Form the F12/3C(FIX) correlation energy */
    virtual void form_f12_energy(einsums::Tensor<double,4> *V, einsums::Tensor<double,4> *X,
                                 einsums::Tensor<double,4> *C, einsums::Tensor<double,4> *B,
                                 einsums::Tensor<double,2> *f, einsums::Tensor<double,4> *G,
                                 einsums::Tensor<double,4> *D);
   
    /* Form the one-electron integrals H = T + V */
    virtual void form_oeints(einsums::Tensor<double, 2> *h);

    /* Form the conventional two-electron integrals */
    virtual void form_teints(const std::string& int_type, einsums::Tensor<double, 4> *ERI,
                             std::vector<char> order);
    
    /* Form the density-fitted two-electron integrals */
    virtual void form_df_teints(const std::string& int_type, einsums::Tensor<double, 4> *ERI,
                                einsums::Tensor<double, 3> *J_inv_AB, std::vector<char> order);
    
    /* Form the Fock matrix */
    virtual void form_fock(einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *k);

    /* Form the density-fitted Fock matrix */
    virtual void form_df_fock(einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *k);

    /* Form the $V^{ij}_{kl}$ or $X^{ij}_{kl}$ tensor */
    virtual void form_V_X(einsums::Tensor<double, 4> *V, einsums::Tensor<double, 4> *X);

    /* Form the density-fitted $V^{ij}_{kl}$ or $X^{ij}_{kl}$ tensor */
    virtual void form_df_V_X(einsums::Tensor<double, 4> *V, einsums::Tensor<double, 4> *X,
                             einsums::Tensor<double, 3> *J_inv_AB);
    
    /* Form the $C^{kl}_{ab}$ tensor */
    virtual void form_C(einsums::Tensor<double, 4> *C, einsums::Tensor<double, 2> *f);

    /* Form the density-fitted $C^{kl}_{ab}$ tensor */
    virtual void form_df_C(einsums::Tensor<double, 4> *C, einsums::Tensor<double, 2> *f,
                           einsums::Tensor<double, 3> *J_inv_AB);
    
    /* Form the $B^{kl}_{mn}$ tensor */
    virtual void form_B(einsums::Tensor<double, 4> *B, einsums::Tensor<double, 2> *f,
                        einsums::Tensor<double, 2> *k);

    /* Form the density-fitted $B^{kl}_{mn}$ tensor */
    virtual void form_df_B(einsums::Tensor<double, 4> *B, einsums::Tensor<double, 2> *f,
                           einsums::Tensor<double, 2> *k, einsums::Tensor<double, 3> *J_inv_AB);

    /* Form the $D^{kl}_{ab} Matrix from Werner */
    virtual void form_D_Werner(einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *tau,
                               einsums::Tensor<double, 2> *t_i);
    
    /* Form an L^{kl}_{ab} matrix */
    virtual void form_L(einsums::Tensor<double, 4> *L, einsums::Tensor<double, 4> *K);

    /* Form $\beta^{kl}_{mn}$ tensor */
    virtual void form_beta(einsums::Tensor<double, 2> *beta, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t,
                           einsums::Tensor<double, 4> *tau);

    /* Form the density-fitted $\beta^{kl}_{mn}$ tensor */
    virtual void form_df_beta(einsums::Tensor<double, 2> *beta, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t, 
                              einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 3> *J_inv_AB);

    /* Form the $A_{ab}$ tensor */                   
    virtual void form_A(einsums::Tensor<double, 2> *A, einsums::Tensor<double, 4> *T);

    /* Form the density-fitted $A_{ab}$ tensor */
    virtual void form_df_A(einsums::Tensor<double, 2> *A, einsums::Tensor<double, 3> *J_inv_AB, einsums::Tensor<double, 4> *T);

    /* Form the $s^{i}_{a}$ tensor */
    virtual void form_s(einsums::Tensor<double, 2> *s, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *f, 
                        einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 2> *A, einsums::Tensor<double, 4> *D);

    /* Form the density fitted $s^{i}_{a}$ tensor */
    virtual void form_df_s(einsums::Tensor<double, 2> *s, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *f,
                           einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 2> *A, einsums::Tensor<double, 4> *D, 
                           einsums::Tensor<double, 3> *J_inv_AB);

    /* Form the $r^{k}_{a}$ tensor */
    virtual void form_r(einsums::Tensor<double, 2> *r, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t);

    /* Form the density fitted $r^{k}_{a}$ tensor */
    virtual void form_df_r(einsums::Tensor<double, 2> *r, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t,
                           einsums::Tensor<double, 3> *J_inv_AB);

    /* Form the $v^{i}_{a}$ singles residual tensor */
    virtual void form_v_ia(einsums::Tensor<double, 2> *v_ia, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *t,
                           einsums::Tensor<double, 2> *beta, einsums::Tensor<double, 2> *r, einsums::Tensor<double, 2> *s);

    /* Form the $X_{ab}$ matrix */
    virtual void form_X_Werner(einsums::Tensor<double, 2> *X, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t,
                               einsums::Tensor<double, 2> *A, einsums::Tensor<double, 2> *r);

    /* Form the density fitted $X_{ab}$ matrix */
    virtual void form_df_X_Werner(einsums::Tensor<double, 2> *X, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t,
                                  einsums::Tensor<double, 2> *A, einsums::Tensor<double, 2> *r, einsums::Tensor<double, 3> *J_inv_AB);

    /* Form the $Y^{kl}_{ab}$ tensor */
    virtual void form_Y_Werner(einsums::Tensor<double, 4> *Y, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *f, 
                               einsums::Tensor<double, 2> *t);

    /* Form the density fitted $Y^{kl}_{ab}$ tensor */
    virtual void form_df_Y_Werner(einsums::Tensor<double, 4> *Y, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *f, 
                                  einsums::Tensor<double, 2> *t, einsums::Tensor<double, 3> *J_inv_AB);

    /* Form the $Z^{kj}_{ab}$ tensor */
    virtual void form_Z_Werner(einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *t);

    /* Form the density fitted $Z^{kj}_{ab}$ tensor */
    virtual void form_df_Z_Werner(einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *t, 
                                  einsums::Tensor<double, 3> *J_inv_AB);

    /* Form the $\alpha^{ij}_{mn}$ constant tensor */
    virtual void form_alpha(einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 2> *t);

    /* Form the density fitted $\alpha^{ij}_{mn}$ constant tensor */
    virtual void form_df_alpha(einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 2> *t,
                               einsums::Tensor<double, 3> *J_inv_AB);

    /* Form the G^{ij}_{ab} matrix */
    virtual void form_G(einsums::Tensor<double, 4> *G, einsums::Tensor<double, 2> *s, einsums::Tensor<double, 2> *t,
                        einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 2> *beta,
                        einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *X, einsums::Tensor<double, 4> *Y,
                        einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *tau);

    /* Form the density fitted $G^{ij}_{ab}$ matrix */
    virtual void form_df_G(einsums::Tensor<double, 4> *G, einsums::Tensor<double, 2> *s, einsums::Tensor<double, 2> *t,
                           einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 2> *beta,
                           einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *X, einsums::Tensor<double, 4> *Y,
                           einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 3> *J_inv_AB);

    /* Form the $V^{ij}_{ab}$ doubles residual tensor */
    virtual void form_V_ijab(einsums::Tensor<double, 4> *V_ijab, einsums::Tensor<double, 4> *G, einsums::Tensor<double, 4> *tau,
                             einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *C);

    /* Form the density fitted $V^{ij}_{ab}$ doubles residual tensor */
    virtual void form_df_V_ijab(einsums::Tensor<double, 4> *V_ijab, einsums::Tensor<double, 4> *G, einsums::Tensor<double, 4> *tau,
                                einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *C, 
                                einsums::Tensor<double, 3> *J_inv_AB);

    void print_results();

    /* Returns the fixed amplitudes value */
    double T_ijkl(const int& p, const int& q, const int& r, const int& s);

    /* Form the $T^{ij}_{ij}\Tilde{V}^{ij}_{ij}$ contirbution to the energy */
    virtual std::pair<double, double> V_Tilde(einsums::Tensor<double, 2>& V_, einsums::Tensor<double, 4> *C,
                                      einsums::TensorView<double, 2>& G_ij, einsums::TensorView<double, 2>& D_ij,
                                      const int& i, const int& j);

    /* Form the $T^{ij}_{ij}\Tilde{B}^{ij}_{ij}T^{ij}_{ij}$ contirbution to the energy */
    virtual std::pair<double, double> B_Tilde(einsums::Tensor<double, 4>& B, einsums::Tensor<double, 4> *C,
                                      einsums::TensorView<double, 2>& D_ij,
                                      const int& i, const int& j);

    /* Form the CCSD F12b correction to the CCSD and Hartree-Fock energy */
    double form_CCSDF12B_Energy(einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 2>  *t_ia);

    /* Density fitted algorithm for the CCSD F12b correction to the CCSD and Hartree-Fock energy */
    double form_CCSDF12B_Energy_df(einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 2>  *t_ia, einsums::Tensor<double, 3> *J_inv_AB);

    /* Form the CCSD Energy Correction to Hartree Fock (E_ccsd_) */
    void form_CCSD_Energy(einsums::Tensor<double, 4> *G, einsums::Tensor<double, 4> *V, einsums::Tensor<double, 4> *tau);

    /* Form Amplitude Root Means Square Change*/
    double get_root_mean_square_amplitude_change(einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 4> *T_ijab, 
                                                 einsums::Tensor<double, 2> *t_ia_old, einsums::Tensor<double, 4> *T_ijab_old);

    /* Converts the AO to MO matrices to einsum::Tensors */
    void convert_C(einsums::Tensor<double,2> *C, OrbitalSpace bs, const int& dim1, const int& dim2,
                   const bool use_frzn);
    void convert_C(einsums::Tensor<double,2> *C, OrbitalSpace bs, const int& dim1, const int& dim2);

    /* Places the computed integral in the einsum::Tensor */
    virtual void set_ERI(einsums::TensorView<double, 4>& ERI_Slice, einsums::Tensor<double, 4> *Slice);
    void set_ERI(einsums::TensorView<double, 3>& ERI_Slice, einsums::Tensor<double, 3> *Slice);

    /* Computes the conventional two-body integrals */
    void two_body_ao_computer(const std::string& int_type, einsums::Tensor<double, 4> *GAO,
                              std::shared_ptr<BasisSet> bs1, std::shared_ptr<BasisSet> bs2,
                              std::shared_ptr<BasisSet> bs3, std::shared_ptr<BasisSet> bs4);

    /* Computes the DF three-index integrals */
    void three_index_ao_computer(const std::string& int_type, einsums::Tensor<double, 3> *Bpq,
                                 std::shared_ptr<BasisSet> bs1, std::shared_ptr<BasisSet> bs2);

    /* Form the integrals containing the DF metric [J_AB]^{-1}(B|PQ) */
    void form_metric_ints(einsums::Tensor<double, 3> *DF_ERI, bool is_fock);
    
    /* Form the integrals containing the explicit correlation (B|\hat{A}_{12}|PQ) */
    void form_oper_ints(const std::string& int_type, einsums::Tensor<double, 3> *DF_ERI);
    void form_oper_ints(const std::string& int_type, einsums::Tensor<double, 2> *DF_ERI);
};

class DiskCCSDF12B : public CCSDF12B {
   public:
    DiskCCSDF12B(SharedWavefunction reference_wavefunction, Options& options);
    ~DiskCCSDF12B() override;

    /* Compute the total CCSD-F12b Energy */
    double compute_energy() override;

   protected:
    /* Form the energy denominator */
    void form_D(einsums::DiskTensor<double, 4> *D, einsums::DiskTensor<double, 2> *f);

   //  /* Form the CABS Singles correction $\frac{|f^{a'}_{i}}|^2}{e_{a'} - e_{i}}$ */
    void form_cabs_singles(einsums::DiskTensor<double,2> *f);

    /* Form the F12b correlation energy */
    void form_f12_energy(einsums::DiskTensor<double,4> *V, einsums::DiskTensor<double,4> *X,
                         einsums::DiskTensor<double,4> *C, einsums::DiskTensor<double,4> *B,
                         einsums::DiskTensor<double,2> *f, einsums::DiskTensor<double,4> *G,
                         einsums::DiskTensor<double,4> *D);

    /* Form the one-electron integrals H = T + V */
    void form_oeints(einsums::DiskTensor<double, 2> *h);

    /* Form the convetional two-electron integrals */
    void form_teints(const std::string& int_type, einsums::DiskTensor<double, 4> *ERI);

    /* Form the density-fitted two-electron integrals */
    void form_df_teints(const std::string& int_type, einsums::DiskTensor<double, 4> *ERI,
                        einsums::Tensor<double, 3> *Metric);

    /* Form the Fock matrix */
    void form_fock(einsums::DiskTensor<double, 2> *f, einsums::DiskTensor<double, 2> *k,
                   einsums::DiskTensor<double, 2> *fk);

    /* Form the DF Fock matrix */
    void form_df_fock(einsums::DiskTensor<double, 2> *f, einsums::DiskTensor<double, 2> *k,
                      einsums::DiskTensor<double, 2> *fk);

    /* Form the $V^{ij}_{kl}$ or $X^{ij}_{kl}$ tensor */
    void form_V_X(einsums::DiskTensor<double, 4> *VX, einsums::DiskTensor<double, 4> *F,
                     einsums::DiskTensor<double, 4> *G_F, einsums::DiskTensor<double, 4> *FG_F2);

    /* Form the $C^{kl}_{ab}$ tensor */
    void form_C(einsums::DiskTensor<double, 4> *C, einsums::DiskTensor<double, 4> *F,
                einsums::DiskTensor<double, 2> *f);

    /* Form the $B^{kl}_{mn}$ tensor */
    void form_B(einsums::DiskTensor<double, 4> *B, einsums::DiskTensor<double, 4> *Uf,
                einsums::DiskTensor<double, 4> *F2, einsums::DiskTensor<double, 4> *F,
                einsums::DiskTensor<double, 2> *f, einsums::DiskTensor<double, 2> *fk,
                einsums::DiskTensor<double, 2> *kk);


    /* Form the $T^{ij}_{ij}\Tilde{V}^{ij}_{ij}$ contirbution to the energy */
    std::pair<double, double> V_Tilde(einsums::Tensor<double, 2>& V_ij, einsums::DiskTensor<double, 4> *C,
                              einsums::DiskView<double, 2, 4>& G_ij, einsums::DiskView<double, 2, 4>& D_ij,
                              const int& i, const int& j);

    /* Form the $T^{ij}_{ij}\Tilde{B}^{ij}_{ij}T^{ij}_{ij}$ contirbution to the energy */
    std::pair<double, double> B_Tilde(einsums::Tensor<double, 4>& B_ij, einsums::DiskTensor<double, 4> *C,
                                      einsums::DiskView<double, 2, 4>& D_ij, const int& i, const int& j);

    /* Places the computed integral in the einsum::DiskTensor */
    void set_ERI(einsums::DiskView<double, 2, 4>& ERI_Slice, einsums::TensorView<double, 2>& Slice);
};

}} // end namespaces
#endif


