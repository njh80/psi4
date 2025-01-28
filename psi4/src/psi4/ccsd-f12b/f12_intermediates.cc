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
Functions
    form_fock (fockMatrix, exchangeMatrix)
        Forms the Fock matrix and the exchange matrix
        Calls form_oeints to form the molecular one-electron integrals from the atomic orbitals in f
        Forms the J and K matrices from the two-electron integrals
        J is outputted as sq|1/r_12|rp from teints with the indices sorted to sr|1/r_12|qp and summed along qp to form the J matrix saved to f twice
        K is outputted as qs|1/r_12|pr and sorted to qr|1/r_12|sp and summed along sp to form the K matrix saved to k
        The J matrix is then subtracted the K matrix and the result is saved to f
    form_df_fock (fockMatrix, exchangeMatrix)
        Forms the Density Fitted Fock matrix and the exchange matrix
        Calls form_oeints to form the molecular one-electron integrals from the atomic orbitals in f
        Next makes metric and "G" operator in (B|PQ) MO basis from form_metric_ints and form_oper_ints
        Then forms J and K matrices from the two-electron integrals
        f is formed from J_oper as B|1/r_12|ij summed on ij with the result einsummed with the metric tensor along B
        k is formed from K_oper as B|1/r_12|Pi summed on i with the result einsummed with the metric tensor along B and j
        The J matrix is then subtracted the K matrix and the result is saved to f
    form_V_X (V, X)
        First generates FG (V) and F2 (X) (i.e. f(r_12)/r_12 and f(r_12)^2) from the two electron integrals in the orbital basis
        Then generates "F" (i.e. f(r_12)) and "G" (i.e. 1/r_12) from the two electron integrals in three orbital dimensions and a single auxiliary dimension
        <oo|f(r_12)|ov> is einsummed with itself into a temporary tensor (to only keep active orbitals), which is sorted and taken away twice from X
        Similarly, V is reduced by this process on G
        Then an F generated in two active and two full orbital dimensions einsummed into the active space reduces X and a similar G reduces V
    form_df_V_X (V, X, J_inv_AB)
        Same as form_V_X but with the Density Fitted integrals meaning that J^-1_AB G^A_pq is used to generate the integrals
    form_C (C, f [Fock Matrix])
        Forms the C matrix from the Fock matrix
        A form_teints call is made to generate a two active, one virtual and one complimentary auxiliary dimension <oo|f(r_12)|vC>
        from <oo|f(r_12)|pc> sorted
        Taking the fock matrix for the virtual and complimentary auxiliary dimensions and einsumming with the <oo|f(r_12)|vC> to keep 
        active and virtual orbitals C is generated
    form_df_C (C, f [Fock Matrix], J_inv_AB)
        Same as form_C but with the Density Fitted integrals meaning that J^-1_AB G^A_pq is used to generate the integrals
    form_B (B, f [Fock Matrix], k [Exchange Matrix])
        Forms the B matrix from the Fock and Exchange matrices
        A form_teints call is made to generate a active space double commutator at B <oo|f(r_12) T_1 f(r_12)|oo>
        Then a <oo|f(r_12)^2|op> and a Fock-Exchange matrix for occupied and all space is generated which are einsummed into the active space
        and sorted to be added to B twice. (|1> is the full basis space)
        Next (<oo|f(r_12)|11> <C|1/r_12|A>) <oo|f(r_12)|11> into the orbital basis is generated and sorted to be subtracted from B twice
        Repeated for <oo|f(r_12)|o1>, <oo|f(r_12)|Co> <o|1/r_12|o> <oo|f(r_12)|Co> is twice sorted and added to B
        <oo|f(r_12)|vq> <p|1/r_12|q> <oo|f(r_12)|vq> is generated and sorted to be subtracted from B twice (note all in the orbital space)
        Similar matrices for <oo|f(r_12)|C1> and <oo|f(r_12)|vC> are generated and sorted to be deducted from B twice
        Finally B is symmetrised by adding half the transpose of B to half of B
    form_df_B (B, f [Fock Matrix], k [Exchange Matrix], J_inv_AB)
        Same but with the Density Fitted integrals meaning that J^-1_AB G^A_pq is used to generate the integrals
    void CCSDF12B::form_D_Werner(einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *tau, 
                             einsums::Tensor<double, 2> *t_i)
    void CCSDF12B::form_L(einsums::Tensor<double, 4> *L, einsums::Tensor<double, 4> *K)
    void CCSDF12B::form_beta(einsums::Tensor<double, 2> *beta, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t,
                         einsums::Tensor<double, 4> *tau)
    void CCSDF12B::form_df_beta(einsums::Tensor<double, 2> *beta, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t, 
                         einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 4> *J_inv_AB)
    void CCSDF12B::form_A(einsums::Tensor<double, 4> *A, einsums::Tensor<double, 4> *T)
    void CCSDF12B::form_df_A(einsums::Tensor<double, 2> *A, einsums::Tensor<double, 4> *J_inv_AB, einsums::Tensor<double, 4> *T)
    void CCSDF12B::form_s(einsums::Tensor<double, 2> *s, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *f,
                          einsums::Tensor<double, 2> *h, einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 4> *A,
                          einsums::Tensor<double, 2> *D)
    void CCSDF12B::form_df_s(einsums::Tensor<double, 2> *s, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *f,
                         einsums::Tensor<double, 2> *h, einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 4> *A,
                         einsums::Tensor<double, 2> *D, einsums::Tensor<double, 3> *J_inv_AB)
    void CCSDF12B::form_r(einsums::Tensor<double, 2> *r, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t)
    void CCSDF12B::form_df_r(einsums::Tensor<double, 2> *r, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t,
                             einsums::Tensor<double, 3> *J_inv_AB)
    void CCSDF12B::form_v_ia(einsums::Tensor<double, 2> *v_ia, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *t,
                              einsums::Tensor<double, 2> *beta, einsums::Tensor<double, 2> *r, einsums::Tensor<double, 2> *s)
    void CCSDF12B::form_X_Werner(einsums::Tensor<double, 2> *X, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t,
                             einsums::Tensor<double, 2> *A, einsums::Tensor<double, 2> *r)
    void CCSDF12B::form_df_X_Werner(einsums::Tensor<double, 2> *X, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t,
                                einsums::Tensor<double, 2> *A, einsums::Tensor<double, 2> *r, einsums::Tensor<double, 3> *J_inv_AB)
    void CCSDF12B::form_Y_Werner(einsums::Tensor<double, 4> *Y, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *f, 
                             einsums::Tensor<double, 2> *t)
    void CCSDF12B::form_df_Y_Werner(einsums::Tensor<double, 4> *Y, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *f, 
                                einsums::Tensor<double, 2> *t, einsums::Tensor<double, 3> *J_inv_AB)
    void CCSDF12B::form_Z_Werner(einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *t)
    void CCSDF12B::form_df_Z_Werner(einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *t, 
                                einsums::Tensor<double, 3> *J_inv_AB)
    void CCSDF12B::form_alpha(einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 2> *t)
    void CCSDF12B::form_df_alpha(einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 2> *t,
                             einsums::Tensor<double, 3> *J_inv_AB)
    void CCSDF12B::form_G(einsums::Tensor<double, 4> *G, einsums::Tensor<double, 2> *s, einsums::Tensor<double, 2> *t,
                      einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 2> *beta,
                      einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 4> *X, einsums::Tensor<double, 4> *Y,
                      einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *tau)
    void CCSDF12B::form_df_G(einsums::Tensor<double, 4> *G, einsums::Tensor<double, 2> *s, einsums::Tensor<double, 2> *t,
                         einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 2> *beta,
                         einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 4> *X, einsums::Tensor<double, 4> *Y,
                         einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 3> *J_inv_AB)
    void CCSDF12B::form_V_ijab(einsums::Tensor<double, 4> *V_ijab, einsums::Tensor<double, 4> *G, einsums::Tensor<double, 4> *tau,
                           einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *C)
    void CCSDF12B::form_df_V_ijab(einsums::Tensor<double, 4> *V_ijab, einsums::Tensor<double, 4> *G, einsums::Tensor<double, 4> *tau,
                              einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *C, 
                              einsums::Tensor<double, 3> *J_inv_AB)
*/

#include "ccsd-f12b.h"

#include "einsums.hpp"

namespace psi { namespace ccsd_f12b {

void CCSDF12B::form_fock(einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *k)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    form_oeints(f);

    Tensor Id = create_identity_tensor("I", nocc_, nocc_);
    {
        outfile->Printf("     Forming J\n");
        auto J = std::make_unique<Tensor<double, 4>>("Coulomb", nri_, nocc_, nri_, nocc_);
        form_teints("J", J.get(), {'O', 'O', 'o', 'o',
                                   'O', 'C', 'o', 'o',
                                   'C', 'C', 'o', 'o'});

        Tensor<double, 4> J_sorted{"pqiI", nri_, nri_, nocc_, nocc_};
        sort(Indices{p, q, i, I}, &J_sorted, Indices{p, i, q, I}, J);
        J.reset();
        einsum(1.0, Indices{p, q}, &(*f), 2.0, Indices{p, q, i, I}, J_sorted, Indices{i, I}, Id);
    }

    {
        outfile->Printf("     Forming K\n");
        auto K = std::make_unique<Tensor<double, 4>>("Exhange", nri_, nocc_, nocc_, nri_);
        form_teints("K", K.get(), {'O', 'o', 'o', 'O',
                                   'O', 'o', 'o', 'C',
                                   'C', 'o', 'o', 'C'});

        Tensor<double, 4> K_sorted{"pqiI", nri_, nri_, nocc_, nocc_};
        sort(Indices{p, q, i, I}, &K_sorted, Indices{p, i, I, q}, K);
        K.reset();
        einsum(Indices{p, q}, &(*k), Indices{p, q, i, I}, K_sorted, Indices{i, I}, Id);
    }

    tensor_algebra::element([](double const &val1, double const &val2)
                            -> double { return val1 - val2; },
                            &(*f), *k);
}

void CCSDF12B::form_df_fock(einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *k)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    form_oeints(f);

    {
        auto Metric = std::make_unique<Tensor<double, 3>>("(B|PQ) MO", naux_, nri_, nri_);
        form_metric_ints(Metric.get(), true);
        auto Oper = std::make_unique<Tensor<double, 3>>("(B|PQ) MO", naux_, nocc_, nri_);
        form_oper_ints("G", Oper.get());

        {
            outfile->Printf("     Forming J\n");
            Tensor Id = create_identity_tensor("I", nocc_, nocc_);
            Tensor J_Metric = (*Metric)(Range{0, naux_}, Range{0, nri_}, Range{0, nri_});
            Tensor J_Oper = (*Oper)(Range{0, naux_}, Range{0, nocc_}, Range{0, nocc_});

            Tensor<double, 1> tmp{"B", naux_};
            einsum(Indices{B}, &tmp, Indices{B, i, j}, J_Oper, Indices{i, j}, Id);
            einsum(1.0, Indices{P, Q}, &(*f), 2.0, Indices{B, P, Q}, J_Metric, Indices{B}, tmp);
        }

        {
            outfile->Printf("     Forming K\n");
            Tensor K_Metric = (*Metric)(Range{0, naux_}, Range{0, nri_}, Range{0, nocc_});
            Tensor K_Oper = (*Oper)(Range{0, naux_}, Range{0, nocc_}, Range{0, nri_});

            Tensor<double, 3> tmp{"", naux_, nocc_, nri_};
            sort(Indices{B, i, P}, &tmp, Indices{B, P, i}, K_Metric);
            einsum(Indices{P, Q}, &(*k), Indices{B, i, P}, tmp, Indices{B, i, Q}, K_Oper);
        }
    }

    tensor_algebra::element([](double const &val1, double const &val2)
                            -> double { return val1 - val2; },
                            &(*f), *k);
}

void CCSDF12B::form_V_X(einsums::Tensor<double, 4> *V, einsums::Tensor<double, 4> *X)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    form_teints("FG", V, {'o', 'o', 'o', 'o'});
    form_teints("F2", X, {'o', 'o', 'o', 'o'});

    {
        Tensor<double, 4> F_oooc{"<oo|F|oC>", nact_, nact_, nocc_, ncabs_};
        form_teints("F", &F_oooc, {'o', 'o', 'o', 'C'});

        Tensor<double, 4> tmp{"Temp", nact_, nact_, nact_, nact_};
        {
            einsum(Indices{i, j, k, l}, &tmp, Indices{i, j, m, q}, F_oooc, Indices{k, l, m, q}, F_oooc);
            sort(1.0, Indices{i, j, k, l}, &(*X), -1.0, Indices{i, j, k, l}, tmp);
            sort(1.0, Indices{i, j, k, l}, &(*X), -1.0, Indices{j, i, l, k}, tmp);
        }

        Tensor<double, 4> G_oooc{"<oo|oC>", nact_, nact_, nocc_, ncabs_};
        form_teints("G", &G_oooc, {'o', 'o', 'o', 'C'});

        einsum(Indices{i, j, k, l}, &tmp, Indices{i, j, m, q}, G_oooc, Indices{k, l, m, q}, F_oooc);
        sort(1.0, Indices{i, j, k, l}, &(*V), -1.0, Indices{i, j, k, l}, tmp);
        sort(1.0, Indices{i, j, k, l}, &(*V), -1.0, Indices{j, i, l, k}, tmp);
    }

    {
        Tensor<double, 4> F_oopq{"<oo|F|OO>", nact_, nact_, nobs_, nobs_};
        form_teints("F", &F_oopq, {'o', 'O', 'o', 'O'});
        einsum(1.0, Indices{i, j, k, l}, &(*X), -1.0, Indices{i, j, p, q}, F_oopq, Indices{k, l, p, q}, F_oopq);

        Tensor<double, 4> G_oopq{"<oo|OO>", nact_, nact_, nobs_, nobs_};
        form_teints("G", &G_oopq, {'o', 'O', 'o', 'O'});
        einsum(1.0, Indices{i, j, k, l}, &(*V), -1.0, Indices{i, j, p, q}, G_oopq, Indices{k, l, p, q}, F_oopq);
    }
}

void CCSDF12B::form_df_V_X(einsums::Tensor<double, 4> *V, einsums::Tensor<double, 4> *X,
                           einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    form_df_teints("FG", V, J_inv_AB, {'o', 'o', 'o', 'o'});
    form_df_teints("F2", X, J_inv_AB, {'o', 'o', 'o', 'o'});

    {
        Tensor<double, 4> F_oooc{"<oo|F|oC>", nact_, nact_, nocc_, ncabs_};
        form_df_teints("F", &F_oooc, J_inv_AB, {'o', 'o', 'o', 'C'});

        Tensor<double, 4> tmp{"Temp", nact_, nact_, nact_, nact_};
        {
            einsum(Indices{i, j, k, l}, &tmp, Indices{i, j, m, q}, F_oooc, Indices{k, l, m, q}, F_oooc);
            sort(1.0, Indices{i, j, k, l}, &(*X), -1.0, Indices{i, j, k, l}, tmp);
            sort(1.0, Indices{i, j, k, l}, &(*X), -1.0, Indices{j, i, l, k}, tmp);
        }

        Tensor<double, 4> G_oooc{"<oo|oC>", nact_, nact_, nocc_, ncabs_};
        form_df_teints("G", &G_oooc, J_inv_AB, {'o', 'o', 'o', 'C'});

        einsum(Indices{i, j, k, l}, &tmp, Indices{i, j, m, q}, G_oooc, Indices{k, l, m, q}, F_oooc);
        sort(1.0, Indices{i, j, k, l}, &(*V), -1.0, Indices{i, j, k, l}, tmp);
        sort(1.0, Indices{i, j, k, l}, &(*V), -1.0, Indices{j, i, l, k}, tmp);
    }

    {
        Tensor<double, 4> F_oopq{"<oo|F|OO>", nact_, nact_, nobs_, nobs_};
        form_df_teints("F", &F_oopq, J_inv_AB, {'o', 'O', 'o', 'O'});
        einsum(1.0, Indices{i, j, k, l}, &(*X), -1.0, Indices{i, j, p, q}, F_oopq, Indices{k, l, p, q}, F_oopq);

        Tensor<double, 4> G_oopq{"<oo|OO>", nact_, nact_, nobs_, nobs_};
        form_df_teints("G", &G_oopq, J_inv_AB, {'o', 'O', 'o', 'O'});
        einsum(1.0, Indices{i, j, k, l}, &(*V), -1.0, Indices{i, j, p, q}, G_oopq, Indices{k, l, p, q}, F_oopq);
    }
}

void CCSDF12B::form_C(einsums::Tensor<double, 4> *C, einsums::Tensor<double, 2> *f)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    Tensor<double, 4> F_oovc{"<oo|F|vC>", nact_, nact_, nvir_, ncabs_};
    {
        Tensor<double, 4> F_oopc{"<oo|F|OC>", nact_, nact_, nobs_, ncabs_};
        form_teints("F", &F_oopc, {'o', 'O', 'o', 'C'});
        F_oovc = F_oopc(All, All, Range{nocc_, nobs_}, All);
    }

    Tensor f_vc = (*f)(Range{nocc_, nobs_}, Range{nobs_, nri_});
    Tensor<double, 4> tmp{"Temp", nact_, nact_, nvir_, nvir_};

    einsum(Indices{k, l, a, b}, &tmp, Indices{k, l, a, q}, F_oovc, Indices{b, q}, f_vc);
    sort(Indices{k, l, a, b}, &(*C), Indices{k, l, a, b}, tmp);
    sort(1.0, Indices{k, l, a, b}, &(*C), 1.0, Indices{l, k, b, a}, tmp);
}

void CCSDF12B::form_df_C(einsums::Tensor<double, 4> *C, einsums::Tensor<double, 2> *f,
                         einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    Tensor<double, 4> F_oovc{"<oo|F|vC>", nact_, nact_, nvir_, ncabs_};
    {
        Tensor<double, 4> F_oopc{"<oo|F|OC>", nact_, nact_, nobs_, ncabs_};
        form_df_teints("F", &F_oopc, J_inv_AB, {'o', 'O', 'o', 'C'});
        F_oovc = F_oopc(All, All, Range{nocc_, nobs_}, All);
    }

    Tensor f_vc = (*f)(Range{nocc_, nobs_}, Range{nobs_, nri_});
    Tensor<double, 4> tmp{"Temp", nact_, nact_, nvir_, nvir_};

    einsum(Indices{k, l, a, b}, &tmp, Indices{k, l, a, q}, F_oovc, Indices{b, q}, f_vc);
    sort(Indices{k, l, a, b}, &(*C), Indices{k, l, a, b}, tmp);
    sort(1.0, Indices{k, l, a, b}, &(*C), 1.0, Indices{l, k, b, a}, tmp);
}

void CCSDF12B::form_B(einsums::Tensor<double, 4> *B, einsums::Tensor<double, 2> *f,
                    einsums::Tensor<double, 2> *k)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    form_teints("Uf", B, {'o', 'o', 'o', 'o'});

    Tensor<double, 4> tmp{"klmn | lkmn", nact_, nact_, nact_, nact_};
    {
        Tensor<double, 4> F2_ooo1{"<oo|F2|o1>", nact_, nact_, nact_, nri_};
        form_teints("F2", &F2_ooo1, {'o', 'o', 'o', 'O',
                                     'o', 'o', 'o', 'C'});

        Tensor<double, 2> fk_o1{"Fock-Exchange Matrix", nact_, nri_};
        {
            auto f_o1 = (*f)(Range{nfrzn_, nocc_}, All);
            auto k_o1 = (*k)(Range{nfrzn_, nocc_}, All);
            tensor_algebra::element([](double const &val1, double const &val2, double const &val3)
                                    -> double { return val2 + val3; },
                                    &fk_o1, f_o1, k_o1);
        }

        einsum(Indices{l, k, n, m}, &tmp, Indices{l, k, n, I}, F2_ooo1, Indices{m, I}, fk_o1);
        sort(1.0, Indices{k, l, m, n}, &(*B), 1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), 1.0, Indices{l, k, n, m}, tmp);
    }

    auto F = std::make_unique<Tensor<double, 4>>("<oo|F|11>", nact_, nact_, nri_, nri_);
    form_teints("F", F.get(), {'o', 'O', 'o', 'O',
                               'o', 'C', 'o', 'O',
                               'o', 'C', 'o', 'C'});

    {
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, nri_, nri_};

        einsum(Indices{l, k, P, A}, &rank4, Indices{l, k, P, C}, *F, Indices{C, A}, *k);
        einsum(Indices{l, k, n, m}, &tmp, Indices{l, k, P, A}, rank4, Indices{n, m, P, A}, *F);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{l, k, n, m}, tmp);
    }

    {
        Tensor F_ooo1 = (*F)(All, All, Range{0, nocc_}, All);
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, nocc_, nri_};

        einsum(Indices{l, k, j, A}, &rank4, Indices{l, k, j, C}, F_ooo1, Indices{C, A}, *f);
        einsum(Indices{l, k, n, m}, &tmp, Indices{l, k, j, A}, rank4, Indices{n, m, j, A}, F_ooo1);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{l, k, n, m}, tmp);
    }

    TensorView<double, 4> F_ooco_temp{(*F), Dim<4>{nact_, nact_, ncabs_, nocc_}, Offset<4>{0, 0, nobs_, 0}};
    {
        Tensor F_ooco = F_ooco_temp;
        Tensor f_oo   = (*f)(Range{0, nocc_}, Range{0, nocc_});
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, ncabs_, nocc_};

        einsum(Indices{l, k, p, i}, &rank4, Indices{l, k, p, j}, F_ooco, Indices{j, i}, f_oo);
        einsum(Indices{l, k, n, m}, &tmp, Indices{l, k, p, i}, rank4, Indices{n, m, p, i}, F_ooco);
        sort(1.0, Indices{k, l, m, n}, &(*B), 1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), 1.0, Indices{l, k, n, m}, tmp);
    }


    TensorView<double, 4> F_oovq_temp{(*F), Dim<4>{nact_, nact_, nvir_, nobs_}, Offset<4>{0, 0, nocc_, 0}};
    {
        Tensor F_oovq = F_oovq_temp;
        Tensor f_pq = (*f)(Range{0, nobs_}, Range{0, nobs_});
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, nvir_, nobs_};

        einsum(Indices{l, k, b, p}, &rank4, Indices{l, k, b, r}, F_oovq, Indices{r, p}, f_pq);
        einsum(Indices{l, k, n, m}, &tmp, Indices{l, k, b, p}, rank4, Indices{n, m, b, p}, F_oovq);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{l, k, n, m}, tmp);
    }

    {
        Tensor F_ooco = F_ooco_temp;
        Tensor F_ooc1 = (*F)(All, All, Range{nobs_, nri_}, All);
        Tensor f_o1   = (*f)(Range{0, nocc_}, All);
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, ncabs_, nocc_};

        einsum(Indices{l, k, p, j}, &rank4, Indices{l, k, p, I}, F_ooc1, Indices{j, I}, f_o1);
        einsum(0.0, Indices{l, k, n, m}, &tmp, 2.0, Indices{l, k, p, j}, rank4, Indices{n, m, p, j}, F_ooco);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{l, k, n, m}, tmp);
    }

    {
        Tensor F_oovq = F_oovq_temp;
        Tensor F_oovc = (*F)(All, All, Range{nocc_, nobs_}, Range{nobs_, nri_});
        Tensor f_pc   = (*f)(Range{0, nobs_}, Range{nobs_, nri_});
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, nvir_, ncabs_};

        einsum(Indices{l, k, b, q}, &rank4, Indices{l, k, b, r}, F_oovq, Indices{r, q}, f_pc);
        einsum(0.0, Indices{l, k, n, m}, &tmp, 2.0, Indices{l, k, b, q}, rank4, Indices{n, m, b, q}, F_oovc);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{l, k, n, m}, tmp);
    }

    auto B_nosymm = (*B)(All, All, All, All);
    sort(0.5, Indices{m, n, k, l}, &(*B), 0.5, Indices{k, l, m, n}, B_nosymm);
}

void CCSDF12B::form_df_B(einsums::Tensor<double, 4> *B, einsums::Tensor<double, 2> *f,
                         einsums::Tensor<double, 2> *k, einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    form_df_teints("Uf", B, J_inv_AB, {'o', 'o', 'o', 'o'});

    Tensor<double, 4> tmp{"klmn | lkmn", nact_, nact_, nact_, nact_};
    {
        Tensor<double, 4> F2_ooo1{"<oo|F2|o1>", nact_, nact_, nact_, nri_};
        form_df_teints("F2", &F2_ooo1, J_inv_AB, {'o', 'o', 'o', 'O',
                                                  'o', 'o', 'o', 'C'});

        Tensor<double, 2> fk_o1{"Fock-Exchange Matrix", nact_, nri_};
        {
            auto f_o1 = (*f)(Range{nfrzn_, nocc_}, All);
            auto k_o1 = (*k)(Range{nfrzn_, nocc_}, All);
            tensor_algebra::element([](double const &val1, double const &val2, double const &val3)
                                    -> double { return val2 + val3; },
                                    &fk_o1, f_o1, k_o1);
        }

        einsum(Indices{l, k, n, m}, &tmp, Indices{l, k, n, I}, F2_ooo1, Indices{m, I}, fk_o1);
        sort(1.0, Indices{k, l, m, n}, B, 1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, B, 1.0, Indices{l, k, n, m}, tmp);
    }

    auto F = std::make_unique<Tensor<double, 4>>("<oo|F|11>", nact_, nact_, nri_, nri_);
    form_df_teints("F", F.get(), J_inv_AB, {'o', 'O', 'o', 'O',
                                            'o', 'O', 'o', 'C',
                                            'o', 'C', 'o', 'O',
                                            'o', 'C', 'o', 'C'});

    {
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, nri_, nri_};

        einsum(Indices{l, k, P, A}, &rank4, Indices{l, k, P, C}, *F, Indices{C, A}, *k);
        einsum(Indices{l, k, n, m}, &tmp, Indices{l, k, P, A}, rank4, Indices{n, m, P, A}, *F);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{l, k, n, m}, tmp);
    }

    {
        Tensor F_ooo1 = (*F)(All, All, Range{0, nocc_}, All);
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, nocc_, nri_};

        einsum(Indices{l, k, j, A}, &rank4, Indices{l, k, j, C}, F_ooo1, Indices{C, A}, *f);
        einsum(Indices{l, k, n, m}, &tmp, Indices{l, k, j, A}, rank4, Indices{n, m, j, A}, F_ooo1);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{l, k, n, m}, tmp);
    }

    TensorView<double, 4> F_ooco_temp{(*F), Dim<4>{nact_, nact_, ncabs_, nocc_}, Offset<4>{0, 0, nobs_, 0}};
    {
        Tensor F_ooco = F_ooco_temp;
        Tensor f_oo   = (*f)(Range{0, nocc_}, Range{0, nocc_});
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, ncabs_, nocc_};

        einsum(Indices{l, k, p, i}, &rank4, Indices{l, k, p, j}, F_ooco, Indices{j, i}, f_oo);
        einsum(Indices{l, k, n, m}, &tmp, Indices{l, k, p, i}, rank4, Indices{n, m, p, i}, F_ooco);
        sort(1.0, Indices{k, l, m, n}, &(*B), 1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), 1.0, Indices{l, k, n, m}, tmp);
    }


    TensorView<double, 4> F_oovq_temp{(*F), Dim<4>{nact_, nact_, nvir_, nobs_}, Offset<4>{0, 0, nocc_, 0}};
    {
        Tensor F_oovq = F_oovq_temp;
        Tensor f_pq = (*f)(Range{0, nobs_}, Range{0, nobs_});
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, nvir_, nobs_};

        einsum(Indices{l, k, b, p}, &rank4, Indices{l, k, b, r}, F_oovq, Indices{r, p}, f_pq);
        einsum(Indices{l, k, n, m}, &tmp, Indices{l, k, b, p}, rank4, Indices{n, m, b, p}, F_oovq);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{l, k, n, m}, tmp);
    }

    {
        Tensor F_ooco = F_ooco_temp;
        Tensor F_ooc1 = (*F)(All, All, Range{nobs_, nri_}, All);
        Tensor f_o1   = (*f)(Range{0, nocc_}, All);
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, ncabs_, nocc_};

        einsum(Indices{l, k, p, j}, &rank4, Indices{l, k, p, I}, F_ooc1, Indices{j, I}, f_o1);
        einsum(0.0, Indices{l, k, n, m}, &tmp, 2.0, Indices{l, k, p, j}, rank4, Indices{n, m, p, j}, F_ooco);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{l, k, n, m}, tmp);
    }

    {
        Tensor F_oovq = F_oovq_temp;
        Tensor F_oovc = (*F)(All, All, Range{nocc_, nobs_}, Range{nobs_, nri_});
        Tensor f_pc   = (*f)(Range{0, nobs_}, Range{nobs_, nri_});
        Tensor<double, 4> rank4{"Contraction 1", nact_, nact_, nvir_, ncabs_};

        einsum(Indices{l, k, b, q}, &rank4, Indices{l, k, b, r}, F_oovq, Indices{r, q}, f_pc);
        einsum(0.0, Indices{l, k, n, m}, &tmp, 2.0, Indices{l, k, b, q}, rank4, Indices{n, m, b, q}, F_oovc);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{k, l, m, n}, tmp);
        sort(1.0, Indices{k, l, m, n}, &(*B), -1.0, Indices{l, k, n, m}, tmp);
    }

    auto B_nosymm = (*B)(All, All, All, All);
    sort(0.5, Indices{m, n, k, l}, &(*B), 0.5, Indices{k, l, m, n}, B_nosymm);
}

void CCSDF12B::form_D_Werner(einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *tau, 
                             einsums::Tensor<double, 2> *t_i)
{
    // Unfortunately Werner et al. (1992) uses D to mean D^ij = C^ij + E^ij + E^ji [transpose], where C^ij = T^ij + t^i t^j or tau
    // This has nothing to do with the inverse fock energy matrix, which is also called D.
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    // For the standard case forming D_ijab
    const bool full_basis = ((*D).dim(2) == nobs_)? true : false;
    const auto otherStart = (full_basis) ? nfrzn_ : nocc_; // All Orbital Basis Functions {p}, or Active/Frozen Orbital Basis Functions {i}

    if (full_basis) { // If just virtuals, r and s cannot == i or j therefore there is no E^{ij} contribution
        for (auto i = 0; i < nact_; i++) {
            for (auto j = 0; j < nact_; j++) {
                for (auto r = 0; r < nobs_ - nfrzn_; r++) {
                    for (auto s = 0; s < nobs_ - nfrzn_; s++) { // Eij rc = dri tjc ; Eji rc transpose = drj tic transpose ; Eji rc transpose = drj tic transpose, but... tjc = tic ???
                        if (r == i && s >= nact_) {
                            (*D)(i, j, r, s) += (*t_i)(j, s); 
                        }
                        if (s == j && r >= nact_) {
                            (*D)(i, j, r, s) += (*t_i)(i, r);
                        }
                        if (r >= nact_ && s >= nact_) { // Add virtual contributions
                            (*D)(i, j, r, s) += (*tau)(i, j, r, s);
                        } else { // Add explicitly correlated contributions
                            (*D)(i, j, r, s) += T_ijkl(i, j, r, s);
                        }
                    }
                }
            }
        }
    } else { // Can add virtual contributions to all elements as dimensions identical
        sort(1.0, Indices{k, l, a, b}, &(*D), 1.0, Indices{k, l, a, b}, *tau);
    }
}

void CCSDF12B::form_L(einsums::Tensor<double, 4> *L, einsums::Tensor<double, 4> *K)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    /* Note that a and b are indicative, the function can handle receiving ijak inputs etc. */
    sort(1.0, Indices{i, j, a, b}, &(*L), 2.0, Indices{i, j, a, b}, *K);
    sort(1.0, Indices{i, j, a, b}, &(*L), -1.0, Indices{j, i, a, b}, *K);
}

void CCSDF12B::form_beta(einsums::Tensor<double, 2> *beta, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t, 
                         einsums::Tensor<double, 4> *tau)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    outfile->Printf("Forming beta\n");

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    form_teints("K", K_pqrs.get(), {'O', 'O', 'O', 'O'});
    outfile->Printf("Formed K_pqrs\n");
    Tensor<double, 2> tmp_trace {"Trace Term", nact_, nact_};
    {
        Tensor K_oovv = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        outfile->Printf("Forming K_oovv\n");
        Tensor<double, 4> L_oovv {"L_oovv", nact_, nact_, nvir_, nvir_};
        outfile->Printf("Forming L_oovv\n");
        form_L(&L_oovv, &K_oovv);
        outfile->Printf("Formed L_oovv\n");
        /* Trace Term */
        einsum(Indices{k, i}, &tmp_trace, Indices{k, l, a, b}, L_oovv, Indices{i, l, a, b}, *tau);
        outfile->Printf("Einsum L_oovv to trace\n");
    }

    Tensor f_ov = (*f)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
    Tensor f_oo = (*f)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_});

    sort(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, i}, f_oo); // Populate beta with fock matrix
    einsum(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, a}, f_ov, Indices{i, a}, *t); // Add f^k t^i to beta (subscripts a on both)
    outfile->Printf("Beta plus fock \n");
    
    {
        Tensor<double, 2> tmp_sum {"Sum Term", nact_, nact_};
        Tensor K_ooov = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        outfile->Printf("Forming K_ooov\n");
        Tensor<double, 4> L_ooov {"L_ooov", nact_, nact_, nact_, nvir_};
        form_L(&L_ooov, &K_ooov);
        outfile->Printf("Forming L_ooov\n");
        einsum(1.0, Indices{k, i}, &tmp_sum, 1.0, Indices{l, a}, *t, Indices{l, k, i, a}, L_ooov);
        sort(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, i}, tmp_sum);
        outfile->Printf("Einsumed L_ooov\n");
    }

    for (auto i = 0; i < nact_; i++) {
        (*beta)(i, i) += tmp_trace(i, i);
    }
}

void CCSDF12B::form_df_beta(einsums::Tensor<double, 2> *beta, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t, 
                            einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    outfile->Printf("Forming beta\n");

    Tensor f_ov = (*f)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
    Tensor f_oo = (*f)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_});

    sort(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, i}, f_oo); // Populate beta with fock matrix
    outfile->Printf("Beta plus fock \n");
    einsum(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, a}, f_ov, Indices{i, a}, *t); // Add f^k t^i to beta (subscripts a on both)
    outfile->Printf("Beta plus fock amplitudes \n");

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    form_df_teints("K", K_pqrs.get(), J_inv_AB, {'O', 'O', 'O', 'O'});
    outfile->Printf("Formed K_pqrs\n");
    Tensor<double, 2> tmp_trace {"Trace Term", nact_, nact_};
    {
        Tensor K_oovv = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        outfile->Printf("Forming K_oovv\n");
        Tensor<double, 4> L_oovv {"L_oovv", nact_, nact_, nvir_, nvir_};
        outfile->Printf("Forming L_oovv\n");
        form_L(&L_oovv, &K_oovv);
        outfile->Printf("Formed L_oovv\n");
        /* Trace Term */
        einsum(Indices{k, i}, &tmp_trace, Indices{k, l, a, b}, L_oovv, Indices{i, l, a, b}, *tau);
        outfile->Printf("Einsum L_oovv to trace\n");
    }

    for (auto i = 0; i < nact_; i++) {
        outfile->Printf("%d\n", i);
        (*beta)(i, i) += tmp_trace(i, i);
    }
    outfile->Printf("Formed beta\n");

    {
        Tensor<double, 2> tmp_sum {"Sum Term", nact_, nact_};
        Tensor K_ooov = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        outfile->Printf("Forming K_ooov\n");
        Tensor<double, 4> L_ooov {"L_ooov", nact_, nact_, nact_, nvir_};
        form_L(&L_ooov, &K_ooov);
        outfile->Printf("Forming L_ooov\n");
        einsum(1.0, Indices{k, i}, &tmp_sum, 1.0, Indices{l, a}, *t, Indices{l, k, i, a}, L_ooov);
        sort(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, i}, tmp_sum);
        outfile->Printf("Einsumed L_ooov\n");
    }
    outfile->Printf("Formed beta\n");
}

void CCSDF12B::form_A(einsums::Tensor<double, 2> *A, einsums::Tensor<double, 4> *T)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto K_ijpq = std::make_unique<Tensor<double, 4>>("K_ijpq", nact_, nact_, nobs_, nobs_);
    form_teints("K", K_ijpq.get(), {'o', 'o', 'O', 'O'});
    Tensor K_oovv = (*K_ijpq)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
    Tensor<double, 4> L_oovv {"L_oovv", nact_, nact_, nvir_, nvir_};
    form_L(&L_oovv, &K_oovv);
    einsum(1.0, Indices{a, b}, &(*A), 1.0, Indices{k, l, a, b}, L_oovv, Indices{l, k, a, b}, *T);
    
}

void CCSDF12B::form_df_A(einsums::Tensor<double, 2> *A, einsums::Tensor<double, 3> *J_inv_AB, einsums::Tensor<double, 4> *T)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto K_ijpq = std::make_unique<Tensor<double, 4>>("K_ijpq", nact_, nact_, nobs_, nobs_);
    form_df_teints("K", K_ijpq.get(), J_inv_AB, {'o', 'o', 'O', 'O'});
    Tensor K_oovv = (*K_ijpq)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
    Tensor<double, 4> L_oovv {"L_oovv", nact_, nact_, nvir_, nvir_};
    form_L(&L_oovv, &K_oovv);
    einsum(1.0, Indices{a, b}, &(*A), 1.0, Indices{k, l, a, b}, L_oovv, Indices{l, k, a, b}, *T);
    
}

void CCSDF12B::form_s(einsums::Tensor<double, 2> *s, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *f, 
                      einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 2> *A, einsums::Tensor<double, 4> *D)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto h_xx = std::make_unique<Tensor<double, 2>>("h_xx", nri_, nri_); 
    form_oeints(h_xx.get());
    Tensor h_vv = (*h_xx)(Range{nocc_, nobs_}, Range{nocc_, nobs_});

    Tensor f_oo = (*f)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_});
    sort(1.0, Indices{i, a}, &(*s), 1.0, Indices{i, a}, f_oo);

    Tensor<double, 2> tmp {"Temp", nvir_, nvir_};
    sort(1.0, Indices{b, a}, &tmp, 1.0, Indices{b, a}, h_vv);
    sort(1.0, Indices{b, a}, &tmp, -1.0, Indices{b, a}, *A);
    einsum(1.0, Indices{i, a}, &(*s), 1.0, Indices{a, b}, tmp, Indices{i, b}, *t_ia);

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    form_teints("K", K_pqrs.get(), {'O', 'O', 'O', 'O'});
    {
        Tensor K_ooov = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, a}, &(*s), 2.0, Indices{i, k, p, q}, *D, Indices{p, q, k, a}, K_ooov);
        einsum(1.0, Indices{i, a}, &(*s), -1.0, Indices{k, i, p, q}, *D, Indices{p, q, k, a}, K_ooov);
        Tensor<double, 4> L_ooov {"L_ooov", nact_, nvir_, nvir_, nvir_};
        form_L(&L_ooov, &K_ooov);
        einsum(1.0, Indices{i, a}, &(*s), -1.0, Indices{k, l, a, b}, *T_ijab, Indices{k, l, i, b}, L_ooov);
    }
}

void CCSDF12B::form_df_s(einsums::Tensor<double, 2> *s, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *f, 
                         einsums::Tensor<double, 2> *A, einsums::Tensor<double, 2> *t_ia, einsums::Tensor<double, 4> *D,
                         einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto h_xx = std::make_unique<Tensor<double, 2>>("h_xx", nri_, nri_);
    form_oeints(h_xx.get());
    Tensor h_vv = (*h_xx)(Range{nocc_, nobs_}, Range{nocc_, nobs_});

    Tensor f_oo = (*f)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_});
    sort(1.0, Indices{i, a}, &(*s), 1.0, Indices{i, a}, f_oo);

    Tensor<double, 2> tmp {"Temp", nvir_, nvir_};
    sort(1.0, Indices{b, a}, &tmp, 1.0, Indices{b, a}, h_vv);
    sort(1.0, Indices{b, a}, &tmp, -1.0, Indices{b, a}, *A);
    einsum(1.0, Indices{i, a}, &(*s), 1.0, Indices{b, a}, tmp, Indices{i, b}, *t_ia);

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    form_df_teints("K", K_pqrs.get(), J_inv_AB, {'O', 'O', 'O', 'O'});
    {
        Tensor K_ooov = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, a}, &(*s), 2.0, Indices{i, k, p, q}, *D, Indices{p, q, k, a}, K_ooov);
        einsum(1.0, Indices{i, a}, &(*s), -1.0, Indices{k, i, p, q}, *D, Indices{p, q, k, a}, K_ooov);
        Tensor<double, 4> L_ooov {"L_ooov", nact_, nvir_, nvir_, nvir_};
        form_L(&L_ooov, &K_ooov);
        einsum(1.0, Indices{i, a}, &(*s), -1.0, Indices{k, l, a, b}, *T_ijab, Indices{k, l, i, b}, L_ooov);
    }
}

void CCSDF12B::form_r(einsums::Tensor<double, 2> *r, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    Tensor f_ov = (*f)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
    sort(1.0, Indices{i, a}, &(*r), 1.0, Indices{i, a}, f_ov);
    auto K_oopq = std::make_unique<Tensor<double, 4>>("K_oopq", nact_, nact_, nobs_, nobs_);
    form_teints("K", K_oopq.get(), {'o', 'o', 'O', 'O'});
    Tensor K_oovv = (*K_oopq)(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
    Tensor<double, 4> L_oovv {"L_oovv", nact_, nact_, nvir_, nvir_};
    form_L(&L_oovv, &K_oovv);
    einsum(1.0, Indices{i, a}, &(*r), 1.0, Indices{i, j, a, b}, L_oovv, Indices{j, b}, *t);
}

void CCSDF12B::form_df_r(einsums::Tensor<double, 2> *r, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t,
                      einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    Tensor f_ov = (*f)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
    sort(1.0, Indices{i, a}, &(*r), 1.0, Indices{i, a}, f_ov);
    auto K_oopq = std::make_unique<Tensor<double, 4>>("K_oopq", nact_, nact_, nobs_, nobs_);
    form_df_teints("K", K_oopq.get(), J_inv_AB, {'o', 'o', 'O', 'O'});
    Tensor K_oovv = (*K_oopq)(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
    Tensor<double, 4> L_oovv {"L_oovv", nact_, nact_, nvir_, nvir_};
    form_L(&L_oovv, &K_oovv);
    einsum(1.0, Indices{i, a}, &(*r), 1.0, Indices{i, j, a, b}, L_oovv, Indices{j, b}, *t);
}

void CCSDF12B::form_v_ia(einsums::Tensor<double, 2> *v_ia, einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *t,
                         einsums::Tensor<double, 2> *beta, einsums::Tensor<double, 2> *r, einsums::Tensor<double, 2> *s)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    Tensor<double, 4> tmp {"Temp", nact_, nact_, nvir_, nvir_};
    sort(1.0, Indices{i, a}, &(*v_ia), 1.0, Indices{i, a}, *s);
    sort(1.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, *T_ijab);
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, b, a}, *T_ijab);
    einsum(1.0, Indices{i, a}, &(*v_ia), 1.0, Indices{i, j, a, b}, tmp, Indices{j, b}, *r);
    einsum(1.0, Indices{i, a}, &(*v_ia), -1.0, Indices{k, i}, *beta, Indices{k, a}, *t);
}

void CCSDF12B::form_X_Werner(einsums::Tensor<double, 2> *X, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t,
                             einsums::Tensor<double, 2> *A, einsums::Tensor<double, 2> *r)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    Tensor f_vv = (*f)(Range{nocc_, nobs_}, Range{nocc_, nobs_});
    sort(1.0, Indices{a, b}, &(*X), 1.0, Indices{a, b}, f_vv);
    sort(1.0, Indices{a, b}, &(*X), -1.0, Indices{a, b}, *A);
    einsum(1.0, Indices{a, b}, &(*X), -1.0, Indices{k, a}, *r, Indices{k, b}, *t);
    
    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", nact_, nobs_, nobs_, nobs_);
    auto K_opqr = std::make_unique<Tensor<double, 4>>("K_opqr", nact_, nobs_, nobs_, nobs_);
    form_teints("J", J_opqr.get(), {'o', 'O', 'O', 'O'});
    form_teints("K", K_opqr.get(), {'o', 'O', 'O', 'O'});
    Tensor J_ovvv = (*J_opqr)(All, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
    Tensor K_ovvv = (*K_opqr)(All, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
    einsum(1.0, Indices{a, b}, &(*X), 2.0, Indices{a, b, i, c}, J_ovvv, Indices{i, c}, *t);
    einsum(1.0, Indices{a, b}, &(*X), -1.0, Indices{a, i, b, c}, K_ovvv, Indices{i, c}, *t);
}

void CCSDF12B::form_df_X_Werner(einsums::Tensor<double, 2> *X, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t,
                                einsums::Tensor<double, 2> *A, einsums::Tensor<double, 2> *r, einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    Tensor f_vv = (*f)(Range{nocc_, nobs_}, Range{nocc_, nobs_});
    sort(1.0, Indices{a, b}, &(*X), 1.0, Indices{a, b}, f_vv);
    sort(1.0, Indices{a, b}, &(*X), -1.0, Indices{a, b}, *A);
    einsum(1.0, Indices{a, b}, &(*X), -1.0, Indices{k, a}, *r, Indices{k, b}, *t);
    
    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", nact_, nobs_, nobs_, nobs_);
    auto K_opqr = std::make_unique<Tensor<double, 4>>("K_opqr", nact_, nobs_, nobs_, nobs_);
    form_df_teints("J", J_opqr.get(), J_inv_AB, {'o', 'O', 'O', 'O'});
    form_df_teints("K", K_opqr.get(), J_inv_AB, {'o', 'O', 'O', 'O'});
    Tensor J_ovvv = (*J_opqr)(All, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
    Tensor K_ovvv = (*K_opqr)(All, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
    einsum(1.0, Indices{a, b}, &(*X), 2.0, Indices{a, b, i, c}, J_ovvv, Indices{i, c}, *t);
    einsum(1.0, Indices{a, b}, &(*X), -1.0, Indices{a, i, b, c}, K_ovvv, Indices{i, c}, *t);
}

void CCSDF12B::form_Y_Werner(einsums::Tensor<double, 4> *Y, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *t,
                             einsums::Tensor<double, 2> *f)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", nact_, nobs_, nobs_, nobs_);
    auto K_opqr = std::make_unique<Tensor<double, 4>>("K_opqr", nact_, nobs_, nobs_, nobs_);

    form_teints("J", J_opqr.get(), {'o', 'O', 'O', 'O'});
    form_teints("K", K_opqr.get(), {'o', 'O', 'O', 'O'});
    
    {
        Tensor J_oovv = (*J_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor K_oovv = (*K_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        sort(1.0, Indices{i, j, a, b}, &(*Y), 1.0, Indices{i, j, a, b}, K_oovv);
        sort(1.0, Indices{i, j, a, b}, &(*Y), -0.5, Indices{i, j, a, b}, J_oovv);
    }

    {
        Tensor J_ovvv = (*J_opqr)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor K_ovvv = (*K_opqr)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, j, a, b}, &(*Y), -0.5, Indices{i, a, b, c}, J_ovvv, Indices{j, c}, *t);
        einsum(1.0, Indices{i, j, a, b}, &(*Y), 1.0, Indices{i, a, b, c}, K_ovvv, Indices{j, c}, *t);
    }

    Tensor<double, 4> tmp {"Temp", nact_, nact_, nvir_, nvir_};
    sort(1.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, *taut);
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, b, a}, *taut);
    {
        Tensor K_oovv = (*K_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor<double, 4> L_oovv {"L_oovv", nact_, nvir_, nvir_, nvir_};
        form_L(&L_oovv, &K_oovv);
        einsum(1.0, Indices{i, j, a, b}, &(*Y), 0.5, Indices{i, j, a, b}, L_oovv, Indices{i, j, a, b}, tmp);
    }

    Tensor f_ov = (*f)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
    einsum(1.0, Indices{i, j, a, b}, &(*Y), 1.0, Indices{i, a}, f_ov, Indices{j, b}, *t);

    {
        Tensor K_ooov = (*K_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor<double, 4> L_ooov {"L_ooov", nact_, nvir_, nvir_, nvir_};
        form_L(&L_ooov, &K_ooov);
        einsum(1.0, Indices{k, j, a, b}, &(*Y), -0.5, Indices{k, l, j, a}, L_ooov, Indices{l, b}, *t);
    }

}

void CCSDF12B::form_df_Y_Werner(einsums::Tensor<double, 4> *Y, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *f, 
                                einsums::Tensor<double, 2> *t, einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", nact_, nobs_, nobs_, nobs_);
    auto K_opqr = std::make_unique<Tensor<double, 4>>("K_opqr", nact_, nobs_, nobs_, nobs_);

    form_df_teints("J", J_opqr.get(), J_inv_AB, {'o', 'O', 'O', 'O'});
    form_df_teints("K", K_opqr.get(), J_inv_AB, {'o', 'O', 'O', 'O'});
    
    {
        Tensor J_oovv = (*J_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor K_oovv = (*K_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        sort(1.0, Indices{i, j, a, b}, &(*Y), 1.0, Indices{i, j, a, b}, K_oovv);
        sort(1.0, Indices{i, j, a, b}, &(*Y), -0.5, Indices{i, j, a, b}, J_oovv);
    }

    {
        Tensor J_ovvv = (*J_opqr)(All, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor K_ovvv = (*K_opqr)(All, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, j, a, b}, &(*Y), -0.5, Indices{i, a, b, c}, J_ovvv, Indices{j, c}, *t);
        einsum(1.0, Indices{i, j, a, b}, &(*Y), 1.0, Indices{i, a, b, c}, K_ovvv, Indices{j, c}, *t);
    }

    Tensor<double, 4> tmp {"Temp", nact_, nact_, nvir_, nvir_};
    sort(1.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, *taut);
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, b, a}, *taut);
    {
        Tensor K_oovv = (*K_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor<double, 4> L_oovv {"L_oovv", nact_, nvir_, nvir_, nvir_};
        form_L(&L_oovv, &K_oovv);
        einsum(1.0, Indices{i, j, a, b}, &(*Y), 0.5, Indices{i, j, a, b}, L_oovv, Indices{i, j, a, b}, tmp);
    }

    Tensor f_ov = (*f)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
    einsum(1.0, Indices{i, j, a, b}, &(*Y), 1.0, Indices{i, a}, f_ov, Indices{j, b}, *t);

    {
        Tensor K_ooov = (*K_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor<double, 4> L_ooov {"L_ooov", nact_, nvir_, nvir_, nvir_};
        form_L(&L_ooov, &K_ooov);
        einsum(1.0, Indices{k, j, a, b}, &(*Y), -0.5, Indices{k, l, j, a}, L_ooov, Indices{l, b}, *t);
    }
}

void CCSDF12B::form_Z_Werner(einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *t)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", nact_, nobs_, nobs_, nobs_);
    auto K_oopq = std::make_unique<Tensor<double, 4>>("K_oopq", nact_, nact_, nobs_, nobs_);

    form_teints("J", J_opqr.get(), {'o', 'O', 'O', 'O'});
    form_teints("K", K_oopq.get(), {'o', 'O', 'o', 'O'});

    {
        Tensor J_oovv = (*J_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        sort(1.0, Indices{i, j, a, b}, &(*Z), 1.0, Indices{i, j, a, b}, J_oovv);
    }
    {
        Tensor J_ovvv = (*J_opqr)(All, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, j, a, b}, &(*Z), 1.0, Indices{a, i, b, c}, J_ovvv, Indices{j, c}, *t);
    }
    {
        Tensor K_oovv = (*K_oopq)(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{k, j, a, b}, &(*Z), -1.0, Indices{l, k, a, b}, K_oovv, Indices{j, l, a, b}, *taut);
    }
    {
        Tensor K_ooov = (*K_oopq)(All, All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{k, j, a, b}, &(*Z), -1.0, Indices{l, k, j, a}, K_ooov, Indices{l, b}, *t);
    }

}

void CCSDF12B::form_df_Z_Werner(einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *t, 
                                einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", nact_, nobs_, nobs_, nobs_);
    auto K_oopq = std::make_unique<Tensor<double, 4>>("K_oopq", nact_, nact_, nobs_, nobs_);

    form_df_teints("J", J_opqr.get(), J_inv_AB, {'o', 'O', 'O', 'O'});
    form_df_teints("K", K_oopq.get(), J_inv_AB, {'o', 'O', 'o', 'O'});

    {
        Tensor J_oovv = (*J_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        sort(1.0, Indices{i, j, a, b}, &(*Z), 1.0, Indices{i, j, a, b}, J_oovv);
    }
    {
        Tensor J_ovvv = (*J_opqr)(All, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, j, a, b}, &(*Z), 1.0, Indices{a, i, b, c}, J_ovvv, Indices{j, c}, *t);
    }
    {
        Tensor K_oovv = (*K_oopq)(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{k, j, a, b}, &(*Z), -1.0, Indices{l, k, a, b}, K_oovv, Indices{j, l, a, b}, *taut);
    }
    {
        Tensor K_ooov = (*K_oopq)(All, All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{k, j, a, b}, &(*Z), -1.0, Indices{l, k, j, a}, K_ooov, Indices{l, b}, *t);
    }

}

void CCSDF12B::form_alpha(einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 2> *t)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto K_oopq = std::make_unique<Tensor<double, 4>>("K_oopq", nact_, nact_, nobs_, nobs_);
    form_teints("K", K_oopq.get(), {'o', 'O', 'o', 'O'});

    {
        Tensor K_oooo = (*K_oopq)(All, All, Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_});
        sort(1.0, Indices{i, j, k, l}, &(*alpha), 1.0, Indices{i, j, k, l}, K_oooo);
    }
    {
        Tensor K_oovv = (*K_oopq)(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, j, k, l}, &(*alpha), 1.0, Indices{i, j, a, b}, *tau, Indices{l, k, a, b}, K_oovv);
    }
    {
        Tensor K_ooov = (*K_oopq)(All, All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, j, k, l}, &(*alpha), 1.0, Indices{i, a}, *t, Indices{k, l, j, a}, K_ooov);
        einsum(1.0, Indices{i, j, k, l}, &(*alpha), 1.0, Indices{j, a}, *t, Indices{l, k, i, a}, K_ooov);
    }
}

void CCSDF12B::form_df_alpha(einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 2> *t,
                             einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto K_oopq = std::make_unique<Tensor<double, 4>>("K_oopq", nact_, nact_, nobs_, nobs_);
    form_df_teints("K", K_oopq.get(), J_inv_AB, {'o', 'O', 'o', 'O'});

    {
        Tensor K_oooo = (*K_oopq)(All, All, Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_});
        sort(1.0, Indices{i, j, k, l}, &(*alpha), 1.0, Indices{i, j, k, l}, K_oooo);
    }
    {
        Tensor K_oovv = (*K_oopq)(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, j, k, l}, &(*alpha), 1.0, Indices{i, j, a, b}, *tau, Indices{l, k, a, b}, K_oovv);
    }
    {
        Tensor K_ooov = (*K_oopq)(All, All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, j, k, l}, &(*alpha), 1.0, Indices{i, a}, *t, Indices{k, l, j, a}, K_ooov);
        einsum(1.0, Indices{i, j, k, l}, &(*alpha), 1.0, Indices{j, a}, *t, Indices{l, k, i, a}, K_ooov);
    }
}

void CCSDF12B::form_G(einsums::Tensor<double, 4> *G, einsums::Tensor<double, 2> *s, einsums::Tensor<double, 2> *t,
                      einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 2> *beta,
                      einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *X, einsums::Tensor<double, 4> *Y,
                      einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *tau)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    form_teints("K", K_pqrs.get(), {'O', 'O', 'O', 'O'});

    Tensor<double, 4> tmp {"Temp", nact_, nact_, nvir_, nvir_};
    sort(1.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, *T_ijab);
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, b, a}, *T_ijab);

    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, a}, *s, Indices{j, b}, *t);
    einsum(1.0, Indices{i, j, a, b}, &(*G), -1.0, Indices{k, i}, *beta, Indices{k, j, a, b}, *tau);
    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, j, a, b}, *T_ijab, Indices{a, b}, *X);
    einsum(1.0, Indices{j, i, a, b}, &(*G), -1.0, Indices{k, i, a, b}, *T_ijab, Indices{k, j, a, b}, *Z);
    einsum(1.0, Indices{i, j, a, b}, &(*G), -0.5, Indices{k, i, a, b}, *T_ijab, Indices{k, j, a, b}, *Z);
    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, k, a, b}, tmp, Indices{k, j, a, b}, *Y);

    Tensor<double, 4> tmp2 {"Temp2", nact_, nact_, nact_, nvir_};
    {
        Tensor K_vpqo = (*K_pqrs)(Range{nocc_, nobs_}, All, Range{nfrzn_, nocc_}, All);
        einsum(1.0, Indices{i, j, k, a}, &tmp2, 1.0, Indices{i, j, p, q}, *D, Indices{a, p, q, k}, K_vpqo);
    }
    {
        Tensor K_ooov = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
        sort(1.0, Indices{i, j, k, a}, &tmp2, 1.0, Indices{i, j, k, a}, K_ooov);
    }
    einsum(1.0, Indices{i, j, a, b}, &(*G), -1.0, Indices{i, j, k, a}, tmp2, Indices{k, b}, *t);
}

void CCSDF12B::form_df_G(einsums::Tensor<double, 4> *G, einsums::Tensor<double, 2> *s, einsums::Tensor<double, 2> *t,
                         einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 2> *beta,
                         einsums::Tensor<double, 4> *T_ijab, einsums::Tensor<double, 2> *X, einsums::Tensor<double, 4> *Y,
                         einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    form_df_teints("K", K_pqrs.get(), J_inv_AB, {'O', 'O', 'O', 'O'});

    Tensor<double, 4> tmp {"Temp", nact_, nact_, nvir_, nvir_};
    sort(1.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, *T_ijab);
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, b, a}, *T_ijab);

    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, a}, *s, Indices{j, b}, *t);
    einsum(1.0, Indices{i, j, a, b}, &(*G), -1.0, Indices{k, i}, *beta, Indices{k, j, a, b}, *tau);
    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, j, a, b}, *T_ijab, Indices{a, b}, *X);
    einsum(1.0, Indices{j, i, a, b}, &(*G), -1.0, Indices{k, i, a, b}, *T_ijab, Indices{k, j, a, b}, *Z);
    einsum(1.0, Indices{i, j, a, b}, &(*G), -0.5, Indices{k, i, a, b}, *T_ijab, Indices{k, j, a, b}, *Z);
    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, k, a, b}, tmp, Indices{k, j, a, b}, *Y);

    Tensor<double, 4> tmp2 {"Temp2", nact_, nact_, nact_, nvir_};
    {
        Tensor K_vpqo = (*K_pqrs)(Range{nocc_, nobs_}, All, Range{nfrzn_, nocc_}, All);
        einsum(1.0, Indices{i, j, k, a}, &tmp2, 1.0, Indices{i, j, p, q}, *D, Indices{a, p, q, k}, K_vpqo);
    }
    {
        Tensor K_ooov = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nocc_, nobs_});
        sort(1.0, Indices{i, j, k, a}, &tmp2, 1.0, Indices{i, j, k, a}, K_ooov);
    }
    einsum(1.0, Indices{i, j, a, b}, &(*G), -1.0, Indices{i, j, k, a}, tmp2, Indices{k, b}, *t);
}

void CCSDF12B::form_V_ijab(einsums::Tensor<double, 4> *V_ijab, einsums::Tensor<double, 4> *G, einsums::Tensor<double, 4> *tau,
                           einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *C)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    auto W_oopq = std::make_unique<Tensor<double, 4>>("W_oopq", nact_, nact_, nobs_, nobs_);
    auto F_pqoo = std::make_unique<Tensor<double, 4>>("F_pqoo", nobs_, nobs_, nact_, nact_);
    form_teints("K", K_pqrs.get(), {'O', 'O', 'O', 'O'});
    form_teints("FG", W_oopq.get(), {'o', 'O', 'o', 'O'});
    form_teints("F", F_pqoo.get(), {'O', 'o', 'O', 'o'});

    Tensor<double, 4> tmp {"Temp", nact_, nact_, nvir_, nvir_};

    /* Include a the extra terms for CCSD-F12b */
    {
        Tensor W_oovv = (*W_oopq)(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        sort(1.0, Indices{i, j, a, b}, &tmp, 1.0, Indices{i, j, a, b}, W_oovv);
    }
    {
        Tensor K_vvpq = (*K_pqrs)(Range{nocc_, nobs_}, Range{nocc_, nobs_}, All, All);
        einsum(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{a, b, p, q}, K_vvpq, Indices{p, q, i, j}, F_pqoo);
    }
    
    sort(1.0, Indices{i, j, a, b}, &tmp, 1.0, Indices{i, j, a, b}, *C);

    for (auto i = 0; i < nact_; i++)
    {
        for (auto j = 0; j < nact_; j++)
        {
            for (auto a = 0; a < nvir_; a++)
            {
                for (auto b = 0; b < nvir_; b++)
                { // T_ijkl(i, i, i, i) as T_ijkl(i, j, j, i) + T_ijkl(i, j, i, j) = T_ijkl(i, i, i, i)
                    (*V_ijab)(i, j, a, b) = T_ijkl(i, i, i, i) * tmp(i, j, a, b);
                }
            }
        }
    }

    sort(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{i, j, a, b}, tmp);
    
    /* Normal Residual terms */

    {
        Tensor K_oovv = (*K_pqrs)(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        sort(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{i, j, a, b}, K_oovv);
    }
    {
        Tensor K_vpqv = (*K_pqrs)(Range{nocc_, nobs_}, All, All, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{i, j, p, q}, *D, Indices{a, p, q, b}, K_vpqv);
    }

    einsum(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{i, j, k, l}, *alpha, Indices{k, l, a, b}, *tau);
    sort(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{i, j, a, b}, *G);
    sort(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{j, i, b, a}, *G); // Transpose

}

void CCSDF12B::form_df_V_ijab(einsums::Tensor<double, 4> *V_ijab, einsums::Tensor<double, 4> *G, einsums::Tensor<double, 4> *tau,
                              einsums::Tensor<double, 4> *D, einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *C, 
                              einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    auto W_oopq = std::make_unique<Tensor<double, 4>>("W_oopq", nact_, nact_, nobs_, nobs_);
    auto F_pqoo = std::make_unique<Tensor<double, 4>>("F_pqoo", nobs_, nobs_, nact_, nact_);
    form_df_teints("K", K_pqrs.get(), J_inv_AB, {'O', 'O', 'O', 'O'});
    form_df_teints("FG", W_oopq.get(), J_inv_AB, {'o', 'O', 'o', 'O'});
    form_df_teints("F", F_pqoo.get(), J_inv_AB, {'O', 'o', 'O', 'o'});

    Tensor<double, 4> tmp {"Temp", nact_, nact_, nvir_, nvir_};

    /* Include a the extra terms for CCSD-F12b */
    {
        Tensor W_oovv = (*W_oopq)(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        sort(1.0, Indices{i, j, a, b}, &tmp, 1.0, Indices{i, j, a, b}, W_oovv);
    }
    {
        Tensor K_vvpq = (*K_pqrs)(Range{nocc_, nobs_}, Range{nocc_, nobs_}, All, All);
        einsum(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{a, b, p, q}, K_vvpq, Indices{p, q, i, j}, F_pqoo);
    }
    
    sort(1.0, Indices{i, j, a, b}, &tmp, 1.0, Indices{i, j, a, b}, *C);

    for (auto i = 0; i < nact_; i++)
    {
        for (auto j = 0; j < nact_; j++)
        {
            for (auto a = 0; a < nvir_; a++)
            {
                for (auto b = 0; b < nvir_; b++)
                {
                    (*V_ijab)(i, j, a, b) = T_ijkl(i, i, i, i) * tmp(i, j, a, b);
                }
            }
        }
    }

    sort(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{i, j, a, b}, tmp);
    
    /* Normal Residual terms */

    {
        Tensor K_oovv = (*K_pqrs)(All, All, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        sort(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{i, j, a, b}, K_oovv);
    }
    {
        Tensor K_vpqv = (*K_pqrs)(Range{nocc_, nobs_}, All, All, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{i, j, p, q}, *D, Indices{a, p, q, b}, K_vpqv);
    }

    einsum(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{i, j, k, l}, *alpha, Indices{k, l, a, b}, *tau);
    sort(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{i, j, a, b}, *G);
    sort(1.0, Indices{i, j, a, b}, &(*V_ijab), 1.0, Indices{j, i, b, a}, *G); // Transpose

}

////////////////////////////////
//* Disk Algorithm (CONV/DF) *//
////////////////////////////////

void DiskCCSDF12B::form_fock(einsums::DiskTensor<double, 2> *f, einsums::DiskTensor<double, 2> *k,
                           einsums::DiskTensor<double, 2> *fk)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    form_oeints(f);
    double mean = 0.0;
    double mean_ii = 0.0;
    double mean_ia = 0.0;
    double mean_ai = 0.0;
    double mean_aa = 0.0;
    double mean_ix = 0.0;
    double mean_xi = 0.0;
    double mean_ax = 0.0;
    double mean_xa = 0.0;
    double mean_xx = 0.0;
    double x_sq = 0.0;
    double x_sq_ii = 0.0;
    double x_sq_ia = 0.0;
    double x_sq_ai = 0.0;
    double x_sq_aa = 0.0;
    double x_sq_ix = 0.0;
    double x_sq_xi = 0.0;
    double x_sq_ax = 0.0;
    double x_sq_xa = 0.0;
    double x_sq_xx = 0.0;
    auto f_view = (*f)(All, All);
    for (int i = 0; i < nri_; i++) {
        for (int a = 0; a < nri_; a++) {
            mean += abs(f_view(i, a));
            x_sq += f_view(i, a) * f_view(i, a);
            if (i < nocc_ && a < nocc_) {
                mean_ii += abs(f_view(i, a));
                x_sq_ii += f_view(i, a) * f_view(i, a);
            } else if (i < nocc_ && a < nobs_) {
                mean_ia += abs(f_view(i, a));
                x_sq_ia += f_view(i, a) * f_view(i, a);
            } else if (i < nobs_ && a < nocc_) {
                mean_ai += abs(f_view(i, a));
                x_sq_ai += f_view(i, a) * f_view(i, a);
            } else if (i < nobs_ && a < nobs_) {
                mean_aa += abs(f_view(i, a));
                x_sq_aa += f_view(i, a) * f_view(i, a);
            } else if (i >= nobs_ && a < nocc_) {
                mean_xi += abs(f_view(i, a));
                x_sq_xi += f_view(i, a) * f_view(i, a);
            } else if (i < nocc_ && a >= nobs_) {
                mean_ix += abs(f_view(i, a));
                x_sq_ix += f_view(i, a) * f_view(i, a);
            } else if (i < nobs_ && a >= nobs_) {
                mean_ax += abs(f_view(i, a));
                x_sq_ax += f_view(i, a) * f_view(i, a);
            } else if (i >= nobs_ && a < nobs_) {
                mean_xa += abs(f_view(i, a));
                x_sq_xa += f_view(i, a) * f_view(i, a);
            } else if (i >= nobs_ && a >= nobs_) {
                mean_xx += abs(f_view(i, a));
                x_sq_xx += f_view(i, a) * f_view(i, a);
            }
        }
    }
    mean /= nri_ * nri_;
    mean_ii /= nocc_ * nocc_;
    mean_ia /= nocc_ * nvir_;
    mean_ai /= nvir_ * nocc_;
    mean_aa /= nvir_ * nvir_;
    mean_ix /= nocc_ * ncabs_;
    mean_xi /= ncabs_ * nocc_;
    mean_ax /= nvir_ * ncabs_;
    mean_xa /= nvir_ * ncabs_;
    mean_xx /= ncabs_ * ncabs_;
    x_sq /= nri_ * nri_;
    x_sq_ii /= nocc_ * nocc_;
    x_sq_ia /= nocc_ * nvir_;
    x_sq_ai /= nvir_ * nocc_;
    x_sq_aa /= nvir_ * nvir_;
    x_sq_ix /= nocc_ * ncabs_;
    x_sq_xi /= ncabs_ * nocc_;
    x_sq_ax /= nvir_ * ncabs_;
    x_sq_xa /= nvir_ * ncabs_;
    x_sq_xx /= ncabs_ * ncabs_;
    
    outfile->Printf("     Mean: %e, Standard Deviation: %e\n", mean, sqrt(x_sq - mean * mean));
    outfile->Printf("     Mean_ii: %e, Standard Deviation: %e\n", mean_ii, sqrt(x_sq_ii - mean_ii * mean_ii));
    outfile->Printf("     Mean_ia: %e, Standard Deviation: %e\n", mean_ia, sqrt(x_sq_ia - mean_ia * mean_ia));
    outfile->Printf("     Mean_ai: %e, Standard Deviation: %e\n", mean_ai, sqrt(x_sq_ai - mean_ai * mean_ai));
    outfile->Printf("     Mean_aa: %e, Standard Deviation: %e\n", mean_aa, sqrt(x_sq_aa - mean_aa * mean_aa));
    outfile->Printf("     Mean_ix: %e, Standard Deviation: %e\n", mean_ix, sqrt(x_sq_ix - mean_ix * mean_ix));
    outfile->Printf("     Mean_xi: %e, Standard Deviation: %e\n", mean_xi, sqrt(x_sq_xi - mean_xi * mean_xi));
    outfile->Printf("     Mean_ax: %e, Standard Deviation: %e\n", mean_ax, sqrt(x_sq_ax - mean_ax * mean_ax));
    outfile->Printf("     Mean_xa: %e, Standard Deviation: %e\n", mean_xa, sqrt(x_sq_xa - mean_xa * mean_xa));
    outfile->Printf("     Mean_xx: %e, Standard Deviation: %e\n", mean_xx, sqrt(x_sq_xx - mean_xx * mean_xx));

    {
        outfile->Printf("     Forming J\n");
        auto J = DiskTensor<double, 4>{state::data(), "Coulomb", nri_, nocc_, nri_, nocc_};
        if (!J.existed()) form_teints("Jf", &J);

        for (int i = 0; i < nocc_; i++) {
            auto J_view = J(All, i, All, i); J_view.set_read_only(true);
            sort(1.0, Indices{p, q}, &f_view.get(), 2.0, Indices{p, q}, J_view.get());
        }

        auto fk_view = (*fk)(All, All);
        sort(Indices{p, q}, &fk_view.get(), Indices{p, q}, f_view.get());
    }

    {
        outfile->Printf("     Forming K\n");
        auto K = DiskTensor<double, 4>{state::data(), "Exchange", nri_, nocc_, nocc_, nri_};
        if (!K.existed()) form_teints("Kf", &K);

        auto k_view = (*k)(All, All);
        for (int i = 0; i < nocc_; i++) {
            auto K_view = K(All, i, i, All); K_view.set_read_only(true);
            sort(1.0, Indices{p, q}, &k_view.get(), 1.0, Indices{p, q}, K_view.get());
        }

        sort(1.0, Indices{p, q}, &f_view.get(), -1.0, Indices{p, q}, k_view.get());
    }
}

void DiskCCSDF12B::form_df_fock(einsums::DiskTensor<double, 2> *f, einsums::DiskTensor<double, 2> *k,
                              einsums::DiskTensor<double, 2> *fk)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    form_oeints(f);
    double mean = 0.0;
    double mean_ii = 0.0;
    double mean_ia = 0.0;
    double mean_ai = 0.0;
    double mean_aa = 0.0;
    double mean_ix = 0.0;
    double mean_ax = 0.0;
    double mean_xx = 0.0;
    double x_sq = 0.0;
    double x_sq_ii = 0.0;
    double x_sq_ia = 0.0;
    double x_sq_ai = 0.0;
    double x_sq_aa = 0.0;
    double x_sq_ix = 0.0;
    double x_sq_ax = 0.0;
    double x_sq_xx = 0.0;
    auto f_view = (*f)(All, All);
    for (int i = 0; i < nri_; i++) {
        for (int a = 0; a < nri_; a++) {
            mean += f_view(i, a);
            x_sq += f_view(i, a) * f_view(i, a);
            if (i < nocc_ && a < nocc_) {
                mean_ii += f_view(i, a);
                x_sq_ii += f_view(i, a) * f_view(i, a);
            } else if (i < nocc_ && a < nobs_) {
                mean_ia += f_view(i, a);
                x_sq_ia += f_view(i, a) * f_view(i, a);
            } else if (i < nobs_ && a < nocc_) {
                mean_ai += f_view(i, a);
                x_sq_ai += f_view(i, a) * f_view(i, a);
            } else if (i < nobs_ && a < nobs_) {
                mean_aa += f_view(i, a);
                x_sq_aa += f_view(i, a) * f_view(i, a);
            } else if ((i < nocc_ && a >= nobs_) || (i >= nobs_ && a < nocc_)) {
                mean_ix += f_view(i, a);
                x_sq_ix += f_view(i, a) * f_view(i, a);
            } else if ((i < nobs_ && a >= nobs_) || (i >= nobs_ && a >= nocc_)) {
                mean_ax += f_view(i, a);
                x_sq_ax += f_view(i, a) * f_view(i, a);
            } else if (i >= nobs_ && a >= nobs_) {
                mean_xx += f_view(i, a);
                x_sq_xx += f_view(i, a) * f_view(i, a);
            }
        }
    }
    mean /= nri_ * nri_;
    mean_ii /= nocc_ * nocc_;
    mean_ia /= nocc_ * nvir_;
    mean_ai /= nvir_ * nocc_;
    mean_aa /= nvir_ * nvir_;
    mean_ix /= nocc_ * ncabs_;
    mean_ax /= nocc_ * ncabs_;
    mean_xx /= ncabs_ * ncabs_;
    x_sq /= nri_ * nri_;
    x_sq_ii /= nocc_ * nocc_;
    x_sq_ia /= nocc_ * nvir_;
    x_sq_ai /= nvir_ * nocc_;
    x_sq_aa /= nvir_ * nvir_;
    x_sq_ix /= nocc_ * ncabs_;
    x_sq_ax /= nocc_ * ncabs_;
    x_sq_xx /= ncabs_ * ncabs_;
    
    outfile->Printf("     Mean: %e, Standard Deviation: %e\n", mean, sqrt(x_sq - mean * mean));
    outfile->Printf("     Mean_ii: %e, Standard Deviation: %e\n", mean_ii, sqrt(x_sq_ii - mean_ii * mean_ii));
    outfile->Printf("     Mean_ia: %e, Standard Deviation: %e\n", mean_ia, sqrt(x_sq_ia - mean_ia * mean_ia));
    outfile->Printf("     Mean_ai: %e, Standard Deviation: %e\n", mean_ai, sqrt(x_sq_ai - mean_ai * mean_ai));
    outfile->Printf("     Mean_aa: %e, Standard Deviation: %e\n", mean_aa, sqrt(x_sq_aa - mean_aa * mean_aa));
    outfile->Printf("     Mean_ix: %e, Standard Deviation: %e\n", mean_ix, sqrt(x_sq_ix - mean_ix * mean_ix));
    outfile->Printf("     Mean_ax: %e, Standard Deviation: %e\n", mean_ax, sqrt(x_sq_ax - mean_ax * mean_ax));
    outfile->Printf("     Mean_xx: %e, Standard Deviation: %e\n", mean_xx, sqrt(x_sq_xx - mean_xx * mean_xx));

    {
        auto Metric = std::make_unique<Tensor<double, 3>>("(B|PQ) MO", naux_, nri_, nri_);
        form_metric_ints(Metric.get(), true);
        auto Oper = std::make_unique<Tensor<double, 3>>("(B|PQ) MO", naux_, nocc_, nri_);
        form_oper_ints("G", Oper.get());

        {
            outfile->Printf("     Forming J\n");
            Tensor<double, 1> tmp{"B", naux_};
            {
                Tensor Id = create_identity_tensor("I", nocc_, nocc_);
                Tensor J_Oper = (*Oper)(Range{0, naux_}, Range{0, nocc_}, Range{0, nocc_});
                einsum(Indices{B}, &tmp, Indices{B, i, j}, J_Oper, Indices{i, j}, Id);
            }

            Tensor J_Metric = (*Metric)(Range{0, naux_}, Range{0, nri_}, Range{0, nri_});
            einsum(1.0, Indices{P, Q}, &f_view.get(), 2.0, Indices{B, P, Q}, J_Metric, Indices{B}, tmp);

            auto fk_view = (*fk)(All, All);
            sort(Indices{p, q}, &fk_view.get(), Indices{p, q}, f_view.get());
        }

        {
            outfile->Printf("     Forming K\n");
            Tensor<double, 3> tmp{"", naux_, nocc_, nri_};
            {
                Tensor K_Metric = (*Metric)(Range{0, naux_}, Range{0, nri_}, Range{0, nocc_});
                sort(Indices{B, i, P}, &tmp, Indices{B, P, i}, K_Metric);
            }

            auto k_view = (*k)(All, All);
            Tensor K_Oper = (*Oper)(Range{0, naux_}, Range{0, nocc_}, Range{0, nri_});
            einsum(Indices{P, Q}, &k_view.get(), Indices{B, i, P}, tmp, Indices{B, i, Q}, K_Oper);

            sort(1.0, Indices{p, q}, &f_view.get(), -1.0, Indices{p, q}, k_view.get());
        }
    }
}

void DiskCCSDF12B::form_V_X(einsums::DiskTensor<double, 4> *VX, einsums::DiskTensor<double, 4> *F,
                          einsums::DiskTensor<double, 4> *G_F, einsums::DiskTensor<double, 4> *FG_F2)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    Tensor<double, 0> tmp1{"terms IJIJ"};
    Tensor<double, 0> tmp2{"terms IJJI"};

    for (int I = 0; I < nact_; I++) {
        for (int J = I; J < nact_; J++) {
            // Terms 2 and 3
            {
                DiskView<double, 2, 4> G_F_IJ_oc{(*G_F), Dim<2>{nact_, ncabs_}, Count<4>{1, 1, nact_, ncabs_}, Offset<4>{I, J, nfrzn_, nobs_}, Stride<4>{1, 1, 1, 1}}; G_F_IJ_oc.set_read_only(true);
                DiskView<double, 2, 4> G_F_JI_oc{(*G_F), Dim<2>{nact_, ncabs_}, Count<4>{1, 1, nact_, ncabs_}, Offset<4>{J, I, nfrzn_, nobs_}, Stride<4>{1, 1, 1, 1}}; G_F_JI_oc.set_read_only(true);
                DiskView<double, 2, 4> F_IJ_oc{(*F), Dim<2>{nact_, ncabs_}, Count<4>{1, 1, nact_, ncabs_}, Offset<4>{I, J, nfrzn_, nobs_}, Stride<4>{1, 1, 1, 1}}; F_IJ_oc.set_read_only(true);
                DiskView<double, 2, 4> F_JI_oc{(*F), Dim<2>{nact_, ncabs_}, Count<4>{1, 1, nact_, ncabs_}, Offset<4>{J, I, nfrzn_, nobs_}, Stride<4>{1, 1, 1, 1}}; F_JI_oc.set_read_only(true);

                einsum(Indices{}, &tmp1, Indices{m, q}, G_F_IJ_oc.get(), Indices{m, q}, F_IJ_oc.get());
                einsum(1.0, Indices{}, &tmp1, 1.0, Indices{m, q}, G_F_JI_oc.get(), Indices{m, q}, F_JI_oc.get());

                if (I != J) {
                    einsum(Indices{}, &tmp2, Indices{m, q}, G_F_IJ_oc.get(), Indices{m, q}, F_JI_oc.get());
                    einsum(1.0, Indices{}, &tmp2, 1.0, Indices{m, q}, G_F_JI_oc.get(), Indices{m, q}, F_IJ_oc.get());
                }
            }

            // Term 4
            {
                auto G_F_IJ_pq = (*G_F)(I, J, Range{nfrzn_, nobs_}, Range{nfrzn_, nobs_}); G_F_IJ_pq.set_read_only(true);
                auto G_F_JI_pq = (*G_F)(J, I, Range{nfrzn_, nobs_}, Range{nfrzn_, nobs_}); G_F_JI_pq.set_read_only(true);
                auto F_IJ_pq = (*F)(I, J, Range{nfrzn_, nobs_}, Range{nfrzn_, nobs_}); F_IJ_pq.set_read_only(true);

                einsum(1.0, Indices{}, &tmp1, 1.0, Indices{p, q}, F_IJ_pq.get(), Indices{p, q}, G_F_IJ_pq.get());

                if (I != J) {
                    einsum(1.0, Indices{}, &tmp2, 1.0, Indices{p, q}, F_IJ_pq.get(), Indices{p, q}, G_F_JI_pq.get());
                }
            }

            int start = 0, stop = nact_;
            if ((*VX).name() == "X Intermediate Tensor") { start = nfrzn_, stop = nocc_; }
            DiskView<double, 2, 4> FG_F2_IJ{(*FG_F2), Dim<2>{nact_, nact_}, Count<4>{1, 1, nact_, nact_}, Offset<4>{I, J, 0, start}, Stride<4>{1, 1, 1, 1}}; FG_F2_IJ.set_read_only(true); // Term 1

            DiskView<double, 2, 4> VX_IJ{(*VX), Dim<2>{nact_, nact_}, Count<4>{1, 1, nact_, nact_}, Offset<4>{I, J, start, start}, Stride<4>{1, 1, 1, 1}};
            VX_IJ(I, J) = FG_F2_IJ(I, J) - tmp1;
            if (I != J) VX_IJ(J, I) = FG_F2_IJ(J, I) - tmp2;
        }
    }
}

void DiskCCSDF12B::form_C(einsums::DiskTensor<double, 4> *C, einsums::DiskTensor<double, 4> *F,
                    einsums::DiskTensor<double, 2> *f)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto f_vc = (*f)(Range{nocc_, nobs_}, Range{nobs_, nri_}); f_vc.set_read_only(true);

    for (int I = 0; I < nact_; I++) {
        for (int J = I; J < nact_; J++) {
            auto F_IJ_vc = (*F)(I, J, Range{nocc_, nobs_}, Range{nobs_, nri_}); F_IJ_vc.set_read_only(true);
            auto F_JI_vc = (*F)(J, I, Range{nocc_, nobs_}, Range{nobs_, nri_}); F_JI_vc.set_read_only(true);

            // IJAB
            {
                auto C_IJ = (*C)(I, J, All, All);
                einsum(Indices{a, b}, &C_IJ.get(), Indices{a, q}, F_IJ_vc.get(), Indices{b, q}, f_vc.get());
                einsum(1.0, Indices{a, b}, &C_IJ.get(), 1.0, Indices{a, q}, f_vc.get(), Indices{b, q}, F_JI_vc.get());
            }

            // JIAB
            if (I != J) {
                auto C_JI = (*C)(J, I, All, All);
                einsum(Indices{a, b}, &C_JI.get(), Indices{a, q}, F_JI_vc.get(), Indices{b, q}, f_vc.get());
                einsum(1.0, Indices{a, b}, &C_JI.get(), 1.0, Indices{a, q}, f_vc.get(), Indices{b, q}, F_IJ_vc.get());
            }
        }
    }
}

void DiskCCSDF12B::form_B(einsums::DiskTensor<double, 4> *B, einsums::DiskTensor<double, 4> *Uf,
                    einsums::DiskTensor<double, 4> *F2, einsums::DiskTensor<double, 4> *F,
                    einsums::DiskTensor<double, 2> *f, einsums::DiskTensor<double, 2> *fk,
                    einsums::DiskTensor<double, 2> *kk)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    // Must fill IJIJ, IJJI and JIJI, JIIJ
    Tensor<double, 0> tmp1{"terms IJIJ"};
    Tensor<double, 0> tmp2{"terms IJJI"};

    // Term 1 and Term 2
    {
        for (int I = 0; I < nact_; I++) {
            for (int J = I; J < nact_; J++) {
                auto fk_I_1 = (*fk)(I + nfrzn_, All); fk_I_1.set_read_only(true);
                auto fk_J_1 = (*fk)(J + nfrzn_, All); fk_J_1.set_read_only(true);

                // IJIJ and JIJI
                {
                    auto F2_JIJ_1 = (*F2)(J, I, J, All); F2_JIJ_1.set_read_only(true);
                    auto F2_IJI_1 = (*F2)(I, J, I, All); F2_IJI_1.set_read_only(true);

                    einsum(Indices{}, &tmp1, Indices{A}, fk_I_1.get(), Indices{A}, F2_JIJ_1.get());
                    einsum(1.0, Indices{}, &tmp1, 1.0, Indices{A}, F2_IJI_1.get(), Indices{A}, fk_J_1.get());
                }

                // IJJI and JIIJ
                if (I != J) {
                    auto F2_IJJ_1 = (*F2)(I, J, J, All); F2_IJJ_1.set_read_only(true);
                    auto F2_JII_1 = (*F2)(J, I, I, All); F2_JII_1.set_read_only(true);

                    einsum(Indices{}, &tmp2, Indices{A}, fk_I_1.get(), Indices{A}, F2_IJJ_1.get());
                    einsum(1.0, Indices{}, &tmp2, 1.0, Indices{A}, F2_JII_1.get(), Indices{A}, fk_J_1.get());
                }

                auto Uf_IJ = (*Uf)(I, J, All, All); Uf_IJ.set_read_only(true); // Term 1

                auto B_IJ = (*B)(I, J, All, All);
                auto B_JI = (*B)(J, I, All, All);
                B_IJ(I, J) = B_JI(J, I) = Uf_IJ(I, J) + tmp1;
                if (I != J) B_IJ(J, I) = B_JI(I, J) = Uf_IJ(J, I) + tmp2;
            }
        }
    }

    // Term 3
    {
        Tensor<double, 2> rank2{"Contraction 1", nri_, nri_};
        auto k_view = (*kk)(All, All); k_view.set_read_only(true);

        for (int I = 0; I < nact_; I++) {
            for (int J = I; J < nact_; J++) {
                auto F_IJ_11 = (*F)(I, J, All, All); F_IJ_11.set_read_only(true);
                auto F_JI_11 = (*F)(J, I, All, All); F_JI_11.set_read_only(true);

                // IJIJ and IJJI
                {
                    einsum(Indices{P, Q}, &rank2, Indices{P, R}, F_IJ_11.get(), Indices{R, Q}, k_view.get());
                    einsum(Indices{}, &tmp1, Indices{P, Q}, rank2, Indices{P, Q}, F_IJ_11.get());

                    if (I != J) {
                        einsum(Indices{}, &tmp2, Indices{P, Q}, rank2, Indices{P, Q}, F_JI_11.get());
                    }
                }

                // JIJI and JIIJ
                {
                    einsum(Indices{P, Q}, &rank2, Indices{P, R}, F_JI_11.get(), Indices{R, Q}, k_view.get());
                    einsum(1.0, Indices{}, &tmp1, 1.0, Indices{P, Q}, rank2, Indices{P, Q}, F_JI_11.get());

                    if (I != J) {
                        einsum(1.0, Indices{}, &tmp2, 1.0, Indices{P, Q}, rank2, Indices{P, Q}, F_IJ_11.get());
                    }
                }

                auto B_IJ = (*B)(I, J, All, All);
                auto B_JI = (*B)(J, I, All, All);
                B_IJ(I, J) = B_JI(J, I) -= tmp1;
                if (I != J) B_IJ(J, I) = B_JI(I, J) -= tmp2;
            }
        }
    }

    // Term 4
    {
        Tensor<double, 2> rank2{"Contraction 1", nocc_, nri_};
        auto f_view = (*f)(All, All); f_view.set_read_only(true);

        for (int I = 0; I < nact_; I++) {
            for (int J = I; J < nact_; J++) {
                auto F_IJ_o1 = (*F)(I, J, Range{0, nocc_}, All); F_IJ_o1.set_read_only(true);
                auto F_JI_o1 = (*F)(J, I, Range{0, nocc_}, All); F_JI_o1.set_read_only(true);

                // IJIJ and IJJI
                {
                    einsum(Indices{j, Q}, &rank2, Indices{j, R}, F_IJ_o1.get(), Indices{R, Q}, f_view.get());
                    einsum(Indices{}, &tmp1, Indices{j, Q}, rank2, Indices{j, Q}, F_IJ_o1.get());

                    if (I != J) {
                        einsum(Indices{}, &tmp2, Indices{j, Q}, rank2, Indices{j, Q}, F_JI_o1.get());
                    }
                }

                // JIJI and JIIJ
                {
                    einsum(Indices{j, Q}, &rank2, Indices{j, R}, F_JI_o1.get(), Indices{R, Q}, f_view.get());
                    einsum(1.0, Indices{}, &tmp1, 1.0, Indices{j, Q}, rank2, Indices{j, Q}, F_JI_o1.get());

                    if (I != J) {
                        einsum(1.0, Indices{}, &tmp2, 1.0, Indices{j, Q}, rank2, Indices{j, Q}, F_IJ_o1.get());
                    }
                }

                auto B_IJ = (*B)(I, J, All, All);
                auto B_JI = (*B)(J, I, All, All);
                B_IJ(I, J) = B_JI(J, I) -= tmp1;
                if (I != J) B_IJ(J, I) = B_JI(I, J) -= tmp2;
            }
        }
    }

    // Term 5 and Term 7
    {
        Tensor<double, 2> rank2{"Contraction 1", ncabs_, nocc_};
        auto f_oo = (*f)(Range{0, nocc_}, Range{0, nocc_}); f_oo.set_read_only(true);
        auto f_o1 = (*f)(Range{0, nocc_}, All); f_o1.set_read_only(true);

        for (int I = 0; I < nact_; I++) {
            for (int J = I; J < nact_; J++) {
                auto F_IJ_co = (*F)(I, J, Range{nobs_, nri_}, Range{0, nocc_}); F_IJ_co.set_read_only(true);
                auto F_JI_co = (*F)(J, I, Range{nobs_, nri_}, Range{0, nocc_}); F_JI_co.set_read_only(true);

                // Term 5
                {
                    einsum(Indices{p, i}, &rank2, Indices{p, j}, F_IJ_co.get(), Indices{i, j}, f_oo.get());
                    einsum(Indices{}, &tmp1, Indices{p, i}, rank2, Indices{p, i}, F_IJ_co.get());

                    if (I != J) {
                        einsum(Indices{}, &tmp2, Indices{p, i}, rank2, Indices{p, i}, F_JI_co.get());
                    }

                    einsum(Indices{p, i}, &rank2, Indices{p, j}, F_JI_co.get(), Indices{i, j}, f_oo.get());
                    einsum(1.0, Indices{}, &tmp1, 1.0, Indices{p, i}, rank2, Indices{p, i}, F_JI_co.get());

                    if (I != J) {
                        einsum(1.0, Indices{}, &tmp2, 1.0, Indices{p, i}, rank2, Indices{p, i}, F_IJ_co.get());
                    }
                }

                // Term 7
                {
                    auto F_IJ_c1 = (*F)(I, J, Range{nobs_, nri_}, All); F_IJ_c1.set_read_only(true);
                    auto F_JI_c1 = (*F)(J, I, Range{nobs_, nri_}, All); F_JI_c1.set_read_only(true);

                    einsum(Indices{p, j}, &rank2, Indices{p, A}, F_IJ_c1.get(), Indices{j, A}, f_o1.get());
                    einsum(1.0, Indices{}, &tmp1, -2.0, Indices{p, j}, rank2, Indices{p, j}, F_IJ_co.get());

                    if (I != J) {
                        einsum(1.0, Indices{}, &tmp2, -2.0, Indices{p, j}, rank2, Indices{p, j}, F_JI_co.get());
                    }

                    einsum(Indices{p, j}, &rank2, Indices{p, A}, F_JI_c1.get(), Indices{j, A}, f_o1.get());
                    einsum(1.0, Indices{}, &tmp1, -2.0, Indices{p, j}, rank2, Indices{p, j}, F_JI_co.get());

                    if (I != J) {
                        einsum(1.0, Indices{}, &tmp2, -2.0, Indices{p, j}, rank2, Indices{p, j}, F_IJ_co.get());
                    }
                }

                auto B_IJ = (*B)(I, J, All, All);
                auto B_JI = (*B)(J, I, All, All);
                B_IJ(I, J) = B_JI(J, I) += tmp1;
                if (I != J) B_IJ(J, I) = B_JI(I, J) += tmp2;
            }
        }
    }

    // Term 6 and Term 8
    {
        Tensor<double, 2> rank2{"Contraction 1", nvir_, nobs_};
        auto f_pq = (*f)(Range{0, nobs_}, Range{0, nobs_}); f_pq.set_read_only(true);
        auto f_pc   = (*f)(Range{0, nobs_}, Range{nobs_, nri_}); f_pc.set_read_only(true);

        for (int I = 0; I < nact_; I++) {
            for (int J = I; J < nact_; J++) {
                auto F_IJ_vq = (*F)(I, J, Range{nocc_, nobs_}, Range{0, nobs_}); F_IJ_vq.set_read_only(true);
                auto F_JI_vq = (*F)(J, I, Range{nocc_, nobs_}, Range{0, nobs_}); F_JI_vq.set_read_only(true);

                // Term 6
                {
                    einsum(Indices{b, q}, &rank2, Indices{b, p}, F_IJ_vq.get(), Indices{p, q}, f_pq.get());
                    einsum(Indices{}, &tmp1, Indices{b, q}, rank2, Indices{b, q}, F_IJ_vq.get());

                    if (I != J) {
                        einsum(Indices{}, &tmp2, Indices{b, q}, rank2, Indices{b, q}, F_JI_vq.get());
                    }

                    einsum(Indices{b, q}, &rank2, Indices{b, p}, F_JI_vq.get(), Indices{p, q}, f_pq.get());
                    einsum(1.0, Indices{}, &tmp1, 1.0, Indices{b, q}, rank2, Indices{b, q}, F_JI_vq.get());

                    if (I != J) {
                        einsum(1.0, Indices{}, &tmp2, 1.0, Indices{b, q}, rank2, Indices{b, q}, F_IJ_vq.get());
                    }
                }

                // Term 8
                {
                    auto F_IJ_vc = (*F)(I, J, Range{nocc_, nobs_}, Range{nobs_, nri_}); F_IJ_vc.set_read_only(true);
                    auto F_JI_vc = (*F)(J, I, Range{nocc_, nobs_}, Range{nobs_, nri_}); F_JI_vc.set_read_only(true);

                    einsum(Indices{b, q}, &rank2, Indices{b, w}, F_IJ_vc.get(), Indices{q, w}, f_pc.get());
                    einsum(1.0, Indices{}, &tmp1, 2.0, Indices{b, q}, rank2, Indices{b, q}, F_IJ_vq.get());

                    if (I != J) {
                        einsum(1.0, Indices{}, &tmp2, 2.0, Indices{b, q}, rank2, Indices{b, q}, F_JI_vq.get());
                    }

                    einsum(Indices{b, q}, &rank2, Indices{b, w}, F_JI_vc.get(), Indices{q, w}, f_pc.get());
                    einsum(1.0, Indices{}, &tmp1, 2.0, Indices{b, q}, rank2, Indices{b, q}, F_JI_vq.get());

                    if (I != J) {
                        einsum(1.0, Indices{}, &tmp2, 2.0, Indices{b, q}, rank2, Indices{b, q}, F_IJ_vq.get());
                    }
                }

                auto B_IJ = (*B)(I, J, All, All);
                auto B_JI = (*B)(J, I, All, All);
                B_IJ(I, J) = B_JI(J, I) -= tmp1;
                if (I != J) B_IJ(J, I) = B_JI(I, J) -= tmp2;
            }
        }
    }
}

void DiskCCSDF12B::form_D_Werner(einsums::DiskTensor<double, 4> *D, einsums::DiskTensor<double, 4> *tau, 
                             einsums::DiskTensor<double, 2> *t_i)
{
    // Unfortunately Werner et al. (1992) uses D to mean D^ij = C^ij + E^ij + E^ji [transpose], where C^ij = T^ij + t^i t^j or tau
    // This has nothing to do with the inverse fock energy matrix, which is also called D. (E^ij_rc = \delta_ri t_jc)
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    //if ((*D).dim(2) == nobs_ - nfrzn_) { // If just virtuals, r and s cannot == i or j therefore there is no E^{ij} contribution
    // For the standard case forming D_ijab 
    DiskView<double, 4, 4> D_view{(*D), Dim<4>{nact_, nact_, D->dim(2), D->dim(3)}, Count<4> {nact_, nact_, D->dim(2), D->dim(3)}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
    DiskView<double, 4, 4> tau_view{(*tau), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4> {nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; tau_view.set_read_only(true);
    DiskView<double, 2, 2> t_i_view{(*t_i), Dim<2>{nact_, nvir_}, Count<2> {nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_i_view.set_read_only(true);
    for (auto i = 0; i < nact_; i++) {
        for (auto j = 0; j < nact_; j++) {
            for (auto r = 0; r < nobs_ - nfrzn_; r++) {
                for (auto s = 0; s < nobs_ - nfrzn_; s++) { // Eij rc = dri tjc
                    D_view(i, j, r, s) = 0.0;
                    if (r == i && s >= nact_) {
                        D_view(i, j, r, s) += t_i_view(j, s-nact_); 
                    }
                    if (s == j && r >= nact_) {
                        D_view(i, j, r, s) += t_i_view(i, r-nact_);
                    }
                    if (r >= nact_ && s >= nact_) { // Add virtual contributions
                        D_view(i, j, r, s) += tau_view(i, j, r-nact_, s-nact_);
                    }
                }
            }
        }
    }
/*    } else { // Can add virtual contributions to all elements as dimensions identical
        for (auto i = 0; i < nact_; i++) {
            for (auto j = 0; j < nact_; j++) {
                auto tau_IJ = (*tau)(i, j, All, All); tau_IJ.set_read_only(true);
                
                //IJAB
                {
                    auto D_IJ = (*D)(i, j, All, All);
                    sort(0.0, Indices{a, b}, &D_IJ.get(), 1.0, Indices{a, b}, tau_IJ.get());
                    outfile->Printf(" D_IJ (0,0,0,0) = %20.12f, tau_IJ (0,0,0,0) = %20.12f\n", D_IJ(0, 0), tau_IJ(0, 0));
                    outfile->Printf(" D_IJ (0,0,%d,%d) = %20.12f, tau_IJ (0,0,%d,%d) = %20.12f\n", nvir_ - 1, nvir_ - 1, D_IJ(nvir_ - 1, nvir_ - 1), nvir_ - 1, nvir_ - 1, tau_IJ(nvir_ - 1, nvir_ - 1));
                }
                //JIAB
                if (i != j) {
                    auto D_JI = (*D)(j, i, All, All);
                    sort(0.0, Indices{a, b}, &D_JI.get(), 1.0, Indices{a, b}, tau_IJ.get());
                    outfile->Printf(" D_JI (0,0,0,0) = %20.12f, tau_IJ (0,0,0,0) = %20.12f\n", D_JI(0, 0), tau_IJ(0, 0));
                    outfile->Printf(" D_JI (0,0,%d,%d) = %20.12f, tau_IJ (0,0,%d,%d) = %20.12f\n", nvir_ - 1, nvir_ - 1, D_JI(nvir_ - 1, nvir_ - 1), nvir_ - 1, nvir_ - 1, tau_IJ(nvir_ - 1, nvir_ - 1));
                }
            }
        }
    }*/
}

void DiskCCSDF12B::form_L(einsums::DiskTensor<double, 4> *L_oovv, einsums::DiskTensor<double, 4> *L_ooov, einsums::DiskTensor<double, 4> *K)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    //DiskView<double, 4, 4> K_view{(*K), Dim<4>{nvir_, nact_, nact_, nobs_ - nfrzn_}, Count<4>{nvir_, nact_, nact_, nobs_ - nfrzn_}, Offset<4>{nocc_, nfrzn_, nfrzn_, nfrzn_}, Stride<4>{1, 1, 1, 1}}; K_view.set_read_only(true);
    DiskView<double, 4, 4> K_view{(*K), Dim<4>{nact_, nact_, nvir_, nobs_ - nfrzn_}, Count<4>{nact_, nact_, nvir_, nobs_ - nfrzn_}, Offset<4>{nfrzn_, nfrzn_, nocc_, nfrzn_}, Stride<4>{1, 1, 1, 1}}; K_view.set_read_only(true);
    DiskView<double, 4, 4> L_oovv_view{(*L_oovv), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
    DiskView<double, 4, 4> L_ooov_view{(*L_ooov), Dim<4>{nact_, nact_, nact_, nvir_}, Count<4>{nact_, nact_, nact_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};

    /* a is a or k -> The memory is hopefully saved by doing these simultaneously */
    for (int i = 0; i < nact_; i++) {
        for (int j = i; j < nact_; j++) {
            for (int a = 0; a < nobs_ - nfrzn_; a++) {
                if (a < nact_) { // Swaps from K notation K^kl_ab = (ak|lb) to L notation L^kl_ab with indices in klab order
                    for (int b = 0; b < nvir_; b++) {
                        L_ooov_view(i, j, a, b) = (2.0 * K_view(i, j, a, b)) - K_view(j, i, a, b);
                        if (i != j) L_ooov_view(j, i, a, b) = (2.0 * K_view(j, i, a, b)) - K_view(i, j, a, b);
                        //L_ooov_view(i, j, a, b) = (2.0 * K_view(b, j, i, a)) - K_view(b, i, j, a);
                        //if (i != j) L_ooov_view(j, i, a, b) = (2.0 * K_view(b, i, j, a)) - K_view(b, j, i, a);
                        //if (L_oovv_view(0, 0, a, b) > 0.0001) outfile->Printf("L_ooov(0, 0, 0, %d) = %e - %e = %e, ", a, 2.0 * K_view(b, i, j, a), K_view(b, j, i, a), L_ooov_view(0, 0, 0, b));
                    }
                    //outfile->Printf("\n");
                } else {
                    for (int b = 0; b < nvir_; b++) {
                        L_oovv_view(i, j, a - nocc_, b) = (2.0 * K_view(i, j, a, b)) - K_view(j, i, a, b);
                        if (i != j) L_oovv_view(j, i, a - nocc_, b) = (2.0 * K_view(j, i, a, b)) - K_view(i, j, a, b);
                        //L_oovv_view(i, j, a - nocc_, b) = (2.0 * K_view(a, j, i, b)) - K_view(a, i, j, b);
                        //if (i != j) L_oovv_view(j, i, a - nocc_, b) = (2.0 * K_view(a, i, j, b)) - K_view(a, j, i, b);
                        //if (L_oovv_view(0, 0, a, b) > 0.0001) outfile->Printf("L_oovv(0, 0, %d, %d) = %e - %e = %e, ", a, b, 2.0 * K_view(b, i, j, a), K_view(b, j, i, a), L_oovv_view(0, 0, a, b));
                    }
                    //outfile->Printf("\n");
                }
            }
        }
    }
}

void DiskCCSDF12B::form_beta(einsums::DiskTensor<double, 2> *beta, einsums::DiskTensor<double, 2> *f, einsums::DiskTensor<double, 2> *t, 
                         einsums::DiskTensor<double, 4> *tau, einsums::DiskTensor<double, 4> *L_oovv, einsums::DiskTensor<double, 4> *L_ooov)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;
    
    DiskView<double, 2, 2> f_ov{(*f), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{nfrzn_, nocc_}, Stride<2>{1, 1}}; f_ov.set_read_only(true);
    DiskView<double, 2, 2> f_oo{(*f), Dim<2>{nact_, nact_}, Count<2>{nact_, nact_}, Offset<2>{nfrzn_, nfrzn_}, Stride<2>{1, 1}}; f_oo.set_read_only(true);
    DiskView<double, 2, 2> t_view{(*t), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_view.set_read_only(true);
    DiskView<double, 2, 2> beta_view{(*beta), Dim<2>{nact_, nact_}, Count<2>{nact_, nact_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
    Tensor<double, 0> tmp {"Temp Value"};

    for (int k = 0; k < nact_; k++) {
        for (int i = 0; i < nact_; i++) {
            DiskView<double, 2, 4> L_ooov_ki{(*L_ooov), Dim<2>{nact_, nvir_}, Count<4>{nact_, 1, 1, nvir_}, Offset<4>{0, k, i, 0}, Stride<4>{1, 1, 1, 1}}; L_ooov_ki.set_read_only(true);
            
            beta_view(k, i) = 0.0;
            einsum(0.0, Indices{}, &tmp, 1.0, Indices{l, a}, t_view.get(), Indices{l, a}, L_ooov_ki.get());
            beta_view(k, i) += tmp;

            DiskView<double, 3, 4> L_oovv_kl{(*L_oovv), Dim<3>{nact_, nvir_, nvir_}, Count<4>{1, nact_, nvir_, nvir_}, Offset<4>{k, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; L_oovv_kl.set_read_only(true);
            DiskView<double, 3, 4> tau_kl{(*tau), Dim<3>{nact_, nvir_, nvir_}, Count<4>{nact_, 1, nvir_, nvir_}, Offset<4>{0, i, 0, 0}, Stride<4>{1, 1, 1, 1}}; tau_kl.set_read_only(true);
            einsum(Indices{k, i}, &beta_view.get(), Indices{l, a, b}, L_oovv_kl.get(), Indices{l, a, b}, tau_kl.get());
        }
    }

    sort(1.0, Indices{k, i}, &beta_view.get(), 1.0, Indices{k, i}, f_oo.get()); // Populate beta with fock matrix
    einsum(1.0, Indices{k, i}, &beta_view.get(), 1.0, Indices{k, a}, f_ov.get(), Indices{i, a}, t_view.get()); // Add f^k t^i to beta (subscripts a on both)

    //if (iteration_ < 2) outfile->Printf("Beta = %e\n", beta_view(0, 0));
}

void DiskCCSDF12B::form_A(einsums::DiskTensor<double, 2> *A, einsums::DiskTensor<double, 4> *T, einsums::DiskTensor<double, 4> *L_oovv)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto A_view = (*A)(All, All);
    A_view.zero();
    Tensor<double, 0> tmp {"Temp Value"};
    for (int a = 0; a < nvir_; a++) {
        for (int b = 0; b < nvir_; b++) {
            DiskView<double, 3, 4> L_oovv_ab{(*L_oovv), Dim<3>{nact_, nact_, nvir_}, Count<4>{nact_, nact_, 1, nvir_}, Offset<4>{0, 0, a, 0}, Stride<4>{1, 1, 1, 1}}; L_oovv_ab.set_read_only(true);
            DiskView<double, 3, 4> T_ab{(*T), Dim<3>{nact_, nact_, nvir_}, Count<4>{nact_, nact_, nvir_, 1}, Offset<4>{0, 0, 0, b}, Stride<4>{1, 1, 1, 1}}; T_ab.set_read_only(true);
            
            einsum(0.0, Indices{}, &tmp, 1.0, Indices{k, l, c}, L_oovv_ab.get(), Indices{l, k, c}, T_ab.get());
            A_view(a, b) += tmp;
            //if (iteration_ < 2 && A_view(a, b) > 0.00001) outfile->Printf("A(%d, %d) = %e, ", a, b, A_view(a, b));
        }
        //if (iteration_ < 2) outfile->Printf("\n");
    }
}

void DiskCCSDF12B::form_s(einsums::DiskTensor<double, 2> *s, einsums::DiskTensor<double, 4> *T_ijab, einsums::DiskTensor<double, 2> *f, 
                          einsums::DiskTensor<double, 2> *t_ia, einsums::DiskTensor<double, 2> *A, einsums::DiskTensor<double, 4> *D,
                          einsums::DiskTensor<double, 4> *K_pqrs, einsums::DiskTensor<double, 4> *L_ooov)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    auto h_xx = std::make_unique<DiskTensor<double, 2>>(state::data(), "h_xx", nri_, nri_); 
    form_oeints(h_xx.get());
    auto h_vv = (*h_xx)(Range{nocc_, nobs_}, Range{nocc_, nobs_});

    /*if (iteration_ < 2) {
        for (int b = 0; b < nvir_; b++) {
            for (int a = 0; a < nvir_; a++) {
                if (h_vv(b, a) > 0.00001) outfile->Printf("h_vv(%d, %d) = %e, ", b, a, h_vv(b, a));
            }
            outfile->Printf("\n");
        }
    }*/

    DiskView<double, 2, 2> f_ov{(*f), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{nfrzn_, nocc_}, Stride<2>{1, 1}}; f_ov.set_read_only(true);
    DiskView<double, 2, 2> s_view{(*s), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
    DiskView<double, 2, 2> t_view{(*t_ia), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_view.set_read_only(true);
    DiskView<double, 2, 2> A_view{(*A), Dim<2>{nvir_, nvir_}, Count<2>{nvir_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; A_view.set_read_only(true);
    s_view.zero();
    sort(1.0, Indices{i, a}, &s_view.get(), 1.0, Indices{i, a}, f_ov.get());
    double mean = 0.0;
    double stdDev = 0.0;
    for (int i = 0; i < nact_; i++) {
        for (int a = 0; a < nvir_; a++) {
            mean += s_view(i, a);
            stdDev += s_view(i, a) * s_view(i, a);
        }
    }

    mean /= nact_ * nvir_;
    stdDev = sqrt(stdDev / (nact_ * nvir_) - mean * mean);
    outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", mean, stdDev);

    Tensor<double, 2> tmp {"Temp", nvir_, nvir_};
    sort(0.0, Indices{b, a}, &tmp, 1.0, Indices{a, b}, h_vv.get());
    sort(1.0, Indices{b, a}, &tmp, -1.0, Indices{b, a}, A_view.get());
    einsum(1.0, Indices{i, a}, &s_view.get(), 1.0, Indices{a, b}, tmp, Indices{i, b}, t_view.get());

    for (int i = 0; i < nact_; i++) {
        for (int a = 0; a < nvir_; a++) {
            if (iteration_ < 2) outfile->Printf("s(%d, %d) = %e\n", i, a, s_view(i, a));
            Tensor<double, 0> tmp_a {"Temp_a"};
            Tensor<double, 0> tmp_b {"Sum Term"};
            DiskView<double, 3, 4> D_ik{(*D), Dim<3>{nact_, nobs_ - nfrzn_, nobs_ - nfrzn_}, Count<4>{1, nact_, nobs_ - nfrzn_, nobs_ - nfrzn_}, Offset<4>{i, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; D_ik.set_read_only(true);
            DiskView<double, 3, 4> D_ki{(*D), Dim<3>{nact_, nobs_ - nfrzn_, nobs_ - nfrzn_}, Count<4>{nact_, 1, nobs_ - nfrzn_, nobs_ - nfrzn_}, Offset<4>{0, i, 0, 0}, Stride<4>{1, 1, 1, 1}}; D_ki.set_read_only(true);
            // K_pqrs is actually K_rpqs therefore we want K_vpqv
            //DiskView<double, 3, 4> K_pqvo_ak{(*K_pqrs), Dim<3>{nobs_ - nfrzn_, nobs_ - nfrzn_, nact_}, Count<4>{1, nobs_ - nfrzn_, nobs_ - nfrzn_, nact_}, Offset<4>{a + nocc_, nfrzn_, nfrzn_, nfrzn_}, Stride<4>{1, 1, 1, 1}}; K_pqvo_ak.set_read_only(true);
            DiskView<double, 3, 4> K_pqvo_ak{(*K_pqrs), Dim<3>{nobs_ - nfrzn_, nobs_ - nfrzn_, nact_}, Count<4>{nobs_ - nfrzn_, nobs_ - nfrzn_, 1, nact_}, Offset<4>{nfrzn_, nfrzn_, a + nocc_, nfrzn_}, Stride<4>{1, 1, 1, 1}}; K_pqvo_ak.set_read_only(true);

            einsum(0.0, Indices{}, &tmp_a, 2.0, Indices{k, p, q}, D_ik.get(), Indices{p, q, k}, K_pqvo_ak.get());
            einsum(1.0, Indices{}, &tmp_a, -1.0, Indices{k, p, q}, D_ki.get(), Indices{p, q, k}, K_pqvo_ak.get());
            //einsum(0.0, Indices{}, &tmp_a, 2.0, Indices{k, p, q}, D_ik.get(), Indices{q, p, k}, K_pqvo_ak.get());
            //einsum(1.0, Indices{}, &tmp_a, -1.0, Indices{k, p, q}, D_ki.get(), Indices{q, p, k}, K_pqvo_ak.get());
            s_view(i, a) += tmp_a;
            if (iteration_ < 2) outfile->Printf("s(%d, %d) = %e\n", i, a, s_view(i, a));

            DiskView<double, 3, 4> L_ooov_ki{(*L_ooov), Dim<3>{nact_, nact_, nvir_}, Count<4>{nact_, nact_, 1, nvir_}, Offset<4>{0, 0, i, 0}, Stride<4>{1, 1, 1, 1}}; L_ooov_ki.set_read_only(true);
            DiskView<double, 3, 4> T_ka{(*T_ijab), Dim<3>{nact_, nact_, nvir_}, Count<4>{nact_, nact_, 1, nvir_}, Offset<4>{0, 0, a, 0}, Stride<4>{1, 1, 1, 1}}; T_ka.set_read_only(true);
                
            einsum(0.0, Indices{}, &tmp_b, -1.0, Indices{l, k, b}, T_ka.get(), Indices{k, l, b}, L_ooov_ki.get());
            s_view(i, a) += tmp_b;
            if (iteration_ < 2) outfile->Printf("s(%d, %d) = %e\n", i, a, s_view(i, a));
        }
    }
    mean = 0.0;
    stdDev = 0.0;
    for (int i = 0; i < nact_; i++) {
        for (int a = 0; a < nvir_; a++) {
            mean += s_view(i, a);
            stdDev += s_view(i, a) * s_view(i, a);
        }
    }

    mean /= nact_ * nvir_;
    stdDev = sqrt(stdDev / (nact_ * nvir_) - mean * mean);
    outfile->Printf("      Mean Value: %e, Standard Deviation: %e\n", mean, stdDev);
}

void DiskCCSDF12B::form_r(einsums::DiskTensor<double, 2> *r, einsums::DiskTensor<double, 2> *f, einsums::DiskTensor<double, 2> *t,
                          einsums::DiskTensor<double, 4> *L_oovv)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    DiskView<double, 2, 2> r_view{(*r), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
    DiskView<double, 2, 2> t_view{(*f), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_view.set_read_only(true);
    DiskView<double, 2, 2> f_ov{(*f), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{nfrzn_, nocc_}, Stride<2>{1, 1}}; f_ov.set_read_only(true);
    r_view.zero();
    sort(1.0, Indices{i, a}, &r_view.get(), 1.0, Indices{i, a}, f_ov.get());
    for (int i = 0; i < nact_; i++) {
        for (int a = 0; a < nvir_; a++) {
            DiskView<double, 2, 4> L_oovv_ia{(*L_oovv), Dim<2>{nact_, nvir_}, Count<4>{1, nact_, 1, nvir_}, Offset<4>{i, 0, a, 0}, Stride<4>{1, 1, 1, 1}}; L_oovv_ia.set_read_only(true);
            Tensor<double, 0> tmp {"Temp Value"};
            einsum(0.0, Indices{}, &tmp, 1.0, Indices{j, b}, L_oovv_ia.get(), Indices{j, b}, t_view.get());
            r_view(i, a) += tmp;
            //if (iteration_ < 2) outfile->Printf("r(%d, %d) = %e, ", i, a, r_view(i, a));
        }
        //if (iteration_ < 2) outfile->Printf("\n");
    }
}

void DiskCCSDF12B::form_v_ia(einsums::DiskTensor<double, 2> *v_ia, einsums::DiskTensor<double, 4> *T_ijab, einsums::DiskTensor<double, 2> *t,
                         einsums::DiskTensor<double, 2> *beta, einsums::DiskTensor<double, 2> *r, einsums::DiskTensor<double, 2> *s)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    Tensor<double, 4> tmp {"Temp", nact_, nact_, nvir_, nvir_};
    DiskView<double, 2, 2> v_ia_view{(*v_ia), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
    DiskView<double, 2, 2> beta_view{(*beta), Dim<2>{nact_, nact_}, Count<2>{nact_, nact_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; beta_view.set_read_only(true);
    DiskView<double, 2, 2> r_view{(*r), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; r_view.set_read_only(true);
    DiskView<double, 2, 2> s_view{(*s), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; s_view.set_read_only(true);
    DiskView<double, 4, 4> T_ijab_view{(*T_ijab), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
    DiskView<double, 2, 2> t_view{(*t), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_view.set_read_only(true);
    v_ia_view.zero();

    sort(1.0, Indices{i, a}, &v_ia_view.get(), 1.0, Indices{i, a}, s_view.get());
    sort(1.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, T_ijab_view.get());
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, b, a}, T_ijab_view.get());
    einsum(1.0, Indices{i, a}, &v_ia_view.get(), 1.0, Indices{i, j, a, b}, tmp, Indices{j, b}, r_view.get());
    einsum(1.0, Indices{i, a}, &v_ia_view.get(), -1.0, Indices{k, i}, beta_view.get(), Indices{k, a}, t_view.get());
    /*if (iteration_ < 2) {
        for (int i = 0; i < nact_; i++) {
            for (int a = 0; a < nvir_; a++) {
                outfile->Printf("v_ia(%d, %d) = %e, ", i, a, v_ia_view(i, a));
            }
            outfile->Printf("\n");
        }
    }*/
}

void DiskCCSDF12B::form_X_Werner(einsums::DiskTensor<double, 2> *X, einsums::DiskTensor<double, 2> *f, einsums::DiskTensor<double, 2> *t,
                                 einsums::DiskTensor<double, 2> *A, einsums::DiskTensor<double, 2> *r, einsums::DiskTensor<double, 4> *J,
                                 einsums::DiskTensor<double, 4> *K)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    DiskView<double, 2, 2> X_view{(*X), Dim<2>{nvir_, nvir_}, Count<2>{nvir_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}};
    DiskView<double, 2, 2> A_view{(*A), Dim<2>{nvir_, nvir_}, Count<2>{nvir_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; A_view.set_read_only(true);
    DiskView<double, 2, 2> r_view{(*r), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; r_view.set_read_only(true);
    DiskView<double, 2, 2> t_view{(*t), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_view.set_read_only(true);
    auto f_vv = (*f)(Range{nocc_, nobs_}, Range{nocc_, nobs_});
    X_view.zero();

    sort(1.0, Indices{a, b}, &X_view.get(), 1.0, Indices{a, b}, f_vv.get());
    sort(1.0, Indices{a, b}, &X_view.get(), -1.0, Indices{a, b}, A_view.get());
    einsum(1.0, Indices{a, b}, &X_view.get(), -1.0, Indices{k, a}, r_view.get(), Indices{k, b}, t_view.get());
    
    for (int a = 0; a < nvir_; a++) {
        for (int b = 0; b < nvir_; b++) {
            DiskView<double, 2, 4> J_ovvv_ab{(*J), Dim<2>{nact_, nvir_}, Count<4>{1, nact_, 1, nvir_}, Offset<4>{a + nocc_, nfrzn_, b + nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; J_ovvv_ab.set_read_only(true);
            //DiskView<double, 2, 4> K_ovvv_ab{(*K), Dim<2>{nact_, nvir_}, Count<4>{1, 1, nact_, nvir_}, Offset<4>{a + nocc_, b + nocc_, nfrzn_, nocc_}, Stride<4>{1, 1, 1, 1}}; K_ovvv_ab.set_read_only(true);
            DiskView<double, 2, 4> K_ovvv_ab{(*K), Dim<2>{nact_, nvir_}, Count<4>{nact_, 1, 1, nvir_}, Offset<4>{nfrzn_, b + nocc_, a + nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; K_ovvv_ab.set_read_only(true);
            Tensor<double, 0> tmp {"Temp Value"};
            einsum(0.0, Indices{}, &tmp, 2.0, Indices{i, c}, J_ovvv_ab.get(), Indices{i, c}, t_view.get());
            einsum(1.0, Indices{}, &tmp, -1.0, Indices{i, c}, K_ovvv_ab.get(), Indices{i, c}, t_view.get());
            X_view(a, b) += tmp;
            //if (iteration_ < 2) outfile->Printf("X(%d, %d) = %e, ", a, b, X_view(a, b));
        }
        //if (iteration_ < 2) outfile->Printf("\n");
    }
}

void DiskCCSDF12B::form_Y_Werner(einsums::DiskTensor<double, 4> *Y, einsums::DiskTensor<double, 4> *taut, einsums::DiskTensor<double, 2> *t,
                             einsums::DiskTensor<double, 2> *f, einsums::DiskTensor<double, 4> *J, einsums::DiskTensor<double, 4> *K,
                             einsums::DiskTensor<double, 4> *L_oovv, einsums::DiskTensor<double, 4> *L_ooov)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    
    DiskView<double, 2, 2> f_ov{(*f), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{nfrzn_, nocc_}, Stride<2>{1, 1}}; f_ov.set_read_only(true);
    DiskView<double, 2, 2> t_view{(*t), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_view.set_read_only(true);
    DiskView<double, 4, 4> Y_view{(*Y), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
    einsum(0.0, Indices{i, j, a, b}, &Y_view.get(), 1.0, Indices{i, a}, f_ov.get(), Indices{j, b}, t_view.get());
    
    for (int i = 0; i < nact_; i++) {
        for (int j = 0; j < nact_; j++) {
            DiskView<double, 3, 4> taut_kj{(*taut), Dim<3>{nact_, nvir_, nvir_}, Count<4>{nact_, 1, nvir_, nvir_}, Offset<4>{0, j, 0, 0}, Stride<4>{1, 1, 1, 1}}; taut_kj.set_read_only(true);
            DiskView<double, 3, 4> taut_jk{(*taut), Dim<3>{nact_, nvir_, nvir_}, Count<4>{1, nact_, nvir_, nvir_}, Offset<4>{j, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; taut_jk.set_read_only(true);
            DiskView<double, 3, 4> L_oovv_i{(*L_oovv), Dim<3>{nact_, nvir_, nvir_}, Count<4>{1, nact_, nvir_, nvir_}, Offset<4>{i, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; L_oovv_i.set_read_only(true);
            auto Y_IJ = (*Y)(i, j, All, All);
            Tensor<double, 3> tmp {"Temp Value", nact_, nvir_, nvir_};
            sort(0.0, Indices{k, c, b}, &tmp, 2.0, Indices{k, c, b}, taut_kj.get());
            sort(1.0, Indices{k, c, b}, &tmp, -1.0, Indices{k, c, b}, taut_jk.get());
            einsum(1.0, Indices{a, b}, &Y_IJ.get(), 0.5, Indices{k, a, c}, L_oovv_i.get(), Indices{k, c, b}, tmp);
            
            DiskView<double, 2, 4> L_ooov_ij{(*L_ooov), Dim<2>{nact_, nvir_}, Count<4>{1, nact_, 1, nvir_}, Offset<4>{i, 0, j, 0}, Stride<4>{1, 1, 1, 1}}; L_ooov_ij.set_read_only(true);
            auto J_oovv_ij = (*J)(i, Range{nocc_, nobs_}, j, Range{nocc_, nobs_}); J_oovv_ij.set_read_only(true);
            //auto K_oovv_ij = (*K)(Range{nocc_, nobs_}, j, i, Range{nocc_, nobs_}); K_oovv_ij.set_read_only(true);
            auto K_oovv_ij = (*K)(i, j, Range{nocc_, nobs_}, Range{nocc_, nobs_}); K_oovv_ij.set_read_only(true);

            sort(1.0, Indices{a, b}, &Y_IJ.get(), -0.5, Indices{a, b}, J_oovv_ij.get());
            sort(1.0, Indices{a, b}, &Y_IJ.get(), 1.0, Indices{a, b}, K_oovv_ij.get());
            einsum(1.0, Indices{a, b}, &Y_IJ.get(), -0.5, Indices{l, a}, L_ooov_ij.get(), Indices{l, b}, t_view.get());
        }
    }
    {
        for (int a = 0; a < nvir_; a++) {
            for (int b = 0; b < nvir_; b++) {
                DiskView<double, 2, 4> Y_ab{(*Y), Dim<2>{nact_, nact_}, Count<4>{nact_, nact_, 1, 1}, Offset<4>{0, 0, a, b}, Stride<4>{1, 1, 1, 1}}; Y_ab.set_read_only(true);
                DiskView<double, 2, 4> J_ovvv_ab{(*J), Dim<2>{nact_, nvir_}, Count<4>{1, nact_, 1, nvir_}, Offset<4>{a + nocc_, nfrzn_, b + nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; J_ovvv_ab.set_read_only(true);
                //DiskView<double, 2, 4> K_ovvv_ab{(*K), Dim<2>{nact_, nvir_}, Count<4>{1, 1, nact_, nvir_}, Offset<4>{a + nocc_, b + nocc_, nfrzn_, nocc_}, Stride<4>{1, 1, 1, 1}}; K_ovvv_ab.set_read_only(true);
                DiskView<double, 2, 4> K_ovvv_ab{(*K), Dim<2>{nact_, nvir_}, Count<4>{nact_, 1, 1, nvir_}, Offset<4>{nfrzn_, b + nocc_, a + nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; K_ovvv_ab.set_read_only(true);

                einsum(1.0, Indices{i, j}, &Y_ab.get(), -0.5, Indices{i, c}, J_ovvv_ab.get(), Indices{j, c}, t_view.get());
                einsum(1.0, Indices{i, j}, &Y_ab.get(), 1.0, Indices{i, c}, K_ovvv_ab.get(), Indices{j, c}, t_view.get());
                //if (iteration_ < 2) outfile->Printf("Y(%d, %d) = %e, ", a, b, Y_ab(0, 0));
            }
            //if (iteration_ < 2) outfile->Printf("\n");
        }
    }
}

void DiskCCSDF12B::form_Z_Werner(einsums::DiskTensor<double, 4> *Z, einsums::DiskTensor<double, 4> *taut, einsums::DiskTensor<double, 2> *t,
                                 einsums::DiskTensor<double, 4> *J, einsums::DiskTensor<double, 4> *K)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    DiskView<double, 2, 2> t_view{(*t), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_view.set_read_only(true);

    for (int i = 0; i < nact_; i++) {
        for (int j = 0; j < nact_; j++) {
            DiskView<double, 3, 4> taut_j{(*taut), Dim<3>{nact_, nvir_, nvir_}, Count<4>{1, nact_, nvir_, nvir_}, Offset<4>{j, 0, 0, 0}, Stride<4>{1, 1, 1, 1}}; taut_j.set_read_only(true);
            auto J_oovv_ij = (*J)(i, Range{nocc_, nobs_}, j, Range{nocc_, nobs_}); J_oovv_ij.set_read_only(true);
            DiskView<double, 3, 4> J_ovvv_i{(*J), Dim<3>{nvir_, nvir_, nvir_}, Count<4>{nvir_, 1, nvir_, nvir_}, Offset<4>{nocc_, i, nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; J_ovvv_i.set_read_only(true);
            //DiskView<double, 3, 4> K_oovv_i{(*K), Dim<3>{nvir_, nact_, nvir_}, Count<4>{nvir_, 1, nact_, nvir_}, Offset<4>{nocc_, i, nfrzn_, nocc_}, Stride<4>{1, 1, 1, 1}}; K_oovv_i.set_read_only(true);
            //DiskView<double, 2, 4> K_ooov_ij{(*K), Dim<2>{nvir_, nact_}, Count<4>{nvir_, 1, nact_, 1}, Offset<4>{nocc_, i, nfrzn_, j}, Stride<4>{1, 1, 1, 1}}; K_ooov_ij.set_read_only(true);
            DiskView<double, 3, 4> K_oovv_i{(*K), Dim<3>{nact_, nvir_, nvir_}, Count<4>{nact_, 1, nvir_, nvir_}, Offset<4>{nfrzn_, i, nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; K_oovv_i.set_read_only(true);
            DiskView<double, 2, 4> K_ooov_ij{(*K), Dim<2>{nact_, nvir_}, Count<4>{nact_, 1, nvir_, 1}, Offset<4>{nfrzn_, i, nocc_, j}, Stride<4>{1, 1, 1, 1}}; K_ooov_ij.set_read_only(true);
            auto t_j = (*t)(j, All); t_j.set_read_only(true); // lk ab -> {alkb}, lkj a -> lk aj -> {alkj}

            auto Z_IJ = (*Z)(i, j, All, All);
            Z_IJ.zero();
            sort(1.0, Indices{a, b}, &Z_IJ.get(), 1.0, Indices{a, b}, J_oovv_ij.get());
            einsum(1.0, Indices{a, b}, &Z_IJ.get(), 1.0, Indices{a, b, c}, J_ovvv_i.get(), Indices{c}, t_j.get());
            einsum(1.0, Indices{a, b}, &Z_IJ.get(), -1.0, Indices{l, a, c}, K_oovv_i.get(), Indices{l, c, b}, taut_j.get());
            einsum(1.0, Indices{a, b}, &Z_IJ.get(), -1.0, Indices{l, a}, K_ooov_ij.get(), Indices{l, b}, t_view.get());
            //einsum(1.0, Indices{a, b}, &Z_IJ.get(), -1.0, Indices{a, l, b}, K_oovv_i.get(), Indices{l, a, b}, taut_j.get());
            //einsum(1.0, Indices{a, b}, &Z_IJ.get(), -1.0, Indices{a, l}, K_ooov_ij.get(), Indices{l, b}, t_view.get());
            /*if (iteration_ < 2) {
                for (int a = 0; a < nvir_; a++) {
                    for (int b = 0; b < nvir_; b++) {
                        outfile->Printf("Z(0, 0, %d, %d) = %e, ", a, b, Z_IJ(a, b));
                    }
                    outfile->Printf("\n");
                }
            }*/
        }
    }
}

void DiskCCSDF12B::form_alpha(einsums::DiskTensor<double, 4> *alpha, einsums::DiskTensor<double, 4> *tau, einsums::DiskTensor<double, 2> *t,
                              einsums::DiskTensor<double, 4> *K)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    for (int i = 0; i < nact_; i++) {
        for (int j = 0; j < nact_; j++) {
            auto tau_ij = (*tau)(i, j, All, All); tau_ij.set_read_only(true);
            auto t_i = (*t)(i, All); t_i.set_read_only(true);
            auto t_j = (*t)(j, All); t_j.set_read_only(true);
            //DiskView<double, 2, 4> K_oooo_ij{(*K), Dim<2>{nact_, nact_}, Count<4>{1, nact_, nact_, 1}, Offset<4>{i, 0, 0, j}, Stride<4>{1, 1, 1, 1}}; K_oooo_ij.set_read_only(true);
            //DiskView<double, 3, 4> K_ooov_j{(*K), Dim<3>{nvir_, nact_, nact_}, Count<4>{nvir_, nact_, nact_, 1}, Offset<4>{nocc_, 0, 0, j}, Stride<4>{1, 1, 1, 1}}; K_ooov_j.set_read_only(true); //klj b -> kl bj -> {bklj}
            //DiskView<double, 3, 4> K_ooov_i{(*K), Dim<3>{nvir_, nact_, nact_}, Count<4>{nvir_, nact_, nact_, 1}, Offset<4>{nocc_, 0, 0, i}, Stride<4>{1, 1, 1, 1}}; K_ooov_i.set_read_only(true); //lki b -> lk bi -> {blki}
            //DiskView<double, 4, 4> K_oovv_{(*K), Dim<4>{nvir_, nact_, nact_, nvir_}, Count<4>{nvir_, nact_, nact_, nvir_}, Offset<4>{nocc_, 0, 0, nocc_}, Stride<4>{1, 1, 1, 1}}; K_oovv_.set_read_only(true);

            DiskView<double, 2, 4> K_oooo_ij{(*K), Dim<2>{nact_, nact_}, Count<4>{nact_, nact_, 1, 1}, Offset<4>{0, 0, i, j}, Stride<4>{1, 1, 1, 1}}; K_oooo_ij.set_read_only(true);
            DiskView<double, 3, 4> K_ooov_j{(*K), Dim<3>{nact_, nact_, nvir_}, Count<4>{nact_, nact_, nvir_, 1}, Offset<4>{0, 0, nocc_, j}, Stride<4>{1, 1, 1, 1}}; K_ooov_j.set_read_only(true); //klj b -> kl bj -> {bklj}
            DiskView<double, 3, 4> K_ooov_i{(*K), Dim<3>{nact_, nact_, nvir_}, Count<4>{nact_, nact_, nvir_, 1}, Offset<4>{0, 0, nocc_, i}, Stride<4>{1, 1, 1, 1}}; K_ooov_i.set_read_only(true); //lki b -> lk bi -> {blki}
            DiskView<double, 4, 4> K_oovv_{(*K), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; K_oovv_.set_read_only(true);

            DiskView<double, 2, 4> alpha_IJ{(*alpha), Dim<2>{nact_, nact_}, Count<4>{1, 1, nact_, nact_}, Offset<4>{i, j, 0, 0}, Stride<4>{1, 1, 1, 1}};
            alpha_IJ.zero();
            sort(1.0, Indices{k, l}, &alpha_IJ.get(), 1.0, Indices{l, k}, K_oooo_ij.get());
            for (int k = 0; k < nact_; k++) {
                for (int l = 0; l < nact_; l++) {
                    //DiskView<double, 2, 4> K_oovv_kl{(*K), Dim<2>{nvir_, nvir_}, Count<4>{nvir_, 1, 1, nvir_}, Offset<4>{nocc_, k, l, nocc_}, Stride<4>{1, 1, 1, 1}}; K_oovv_kl.set_read_only(true);
                    DiskView<double, 2, 4> K_oovv_kl{(*K), Dim<2>{nvir_, nvir_}, Count<4>{1, 1, nvir_, nvir_}, Offset<4>{l, k, nocc_, nocc_}, Stride<4>{1, 1, 1, 1}}; K_oovv_kl.set_read_only(true);
                    einsum(1.0, Indices{k, l}, &alpha_IJ.get(), 1.0, Indices{a, b}, tau_ij.get(), Indices{a, b}, K_oovv_kl.get());
                }
            }

            einsum(1.0, Indices{k, l}, &alpha_IJ.get(), 1.0, Indices{a}, t_i.get(), Indices{k, l, a}, K_ooov_j.get());
            einsum(1.0, Indices{k, l}, &alpha_IJ.get(), 1.0, Indices{a}, t_j.get(), Indices{l, k, a}, K_ooov_i.get());
            //einsum(1.0, Indices{k, l}, &alpha_IJ.get(), 1.0, Indices{a}, t_i.get(), Indices{a, l, k}, K_ooov_j.get());
            //einsum(1.0, Indices{k, l}, &alpha_IJ.get(), 1.0, Indices{a}, t_j.get(), Indices{a, k, l}, K_ooov_i.get());
            //if (iteration_ < 2) outfile->Printf("alpha(0, 0) = %e\n", alpha_IJ(i, i));
        }
    }
}

void DiskCCSDF12B::form_G(einsums::DiskTensor<double, 4> *G, einsums::DiskTensor<double, 2> *s, einsums::DiskTensor<double, 2> *t,
                      einsums::DiskTensor<double, 4> *D, einsums::DiskTensor<double, 2> *beta, einsums::DiskTensor<double, 4> *T_ijab, 
                      einsums::DiskTensor<double, 2> *X, einsums::DiskTensor<double, 4> *Y, einsums::DiskTensor<double, 4> *Z, 
                      einsums::DiskTensor<double, 4> *tau, einsums::DiskTensor<double, 4> *K)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    DiskView<double, 4, 4> G_view{(*G), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
    DiskView<double, 2, 2> s_view{(*s), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; s_view.set_read_only(true);
    DiskView<double, 2, 2> t_view{(*t), Dim<2>{nact_, nvir_}, Count<2>{nact_, nvir_}, Offset<2>{0, 0}, Stride<2>{1, 1}}; t_view.set_read_only(true);
    einsum(0.0, Indices{i, j, a, b}, &G_view.get(), 1.0, Indices{i, a}, s_view.get(), Indices{j, b}, t_view.get());
    auto X_view = (*X)(All, All); X_view.set_read_only(true);
    outfile->Printf("Computing G\n");

#pragma omp parallel for schedule(dynamic) collapse(2) num_threads(nthreads_)
    for (int i = 0; i < nact_; i++) {
        for (int j = 0; j < nact_; j++) {
            auto T_ij = (*T_ijab)(i, j, All, All); T_ij.set_read_only(true);
            
            auto G_ij = (*G)(i, j, All, All);
            einsum(1.0, Indices{a, b}, &G_ij.get(), 1.0, Indices{a, c}, T_ij.get(), Indices{c, b}, X_view.get());

            for (int a = 0; a < nvir_; a++) {
                auto D_ij = (*D)(i, j, All, All); D_ij.set_read_only(true);
                //DiskView<double, 3, 4> K_vpqo_a{(*K), Dim<3>{nobs_, nobs_, nact_}, Count<4>{1, nobs_, nobs_, nact_}, Offset<4>{a + nocc_, nfrzn_, nfrzn_, nfrzn_}, Stride<4>{1, 1, 1, 1}}; K_vpqo_a.set_read_only(true);
                //DiskView<double, 1, 4> K_ooov_ija{(*K), Dim<1>{nact_}, Count<4>{1, 1, 1, nact_}, Offset<4>{a + nocc_, j + nfrzn_, i + nfrzn_, nfrzn_}, Stride<4>{1, 1, 1, 1}}; K_ooov_ija.set_read_only(true);
                DiskView<double, 3, 4> K_vpqo_a{(*K), Dim<3>{nobs_-nfrzn_, nobs_-nfrzn_, nact_}, Count<4>{nobs_-nfrzn_, nobs_-nfrzn_, 1, nact_}, Offset<4>{nfrzn_, nfrzn_, a + nocc_, nfrzn_}, Stride<4>{1, 1, 1, 1}}; K_vpqo_a.set_read_only(true);
                DiskView<double, 1, 4> K_ooov_ija{(*K), Dim<1>{nact_}, Count<4>{1, 1, 1, nact_}, Offset<4>{i + nfrzn_, j + nfrzn_, a + nocc_, nfrzn_}, Stride<4>{1, 1, 1, 1}}; K_ooov_ija.set_read_only(true);
                DiskView<double, 2, 4> T_ia{(*T_ijab), Dim<2>{nact_, nvir_}, Count<4>{1, nact_, 1, nvir_}, Offset<4>{i, 0, a, 0}, Stride<4>{1, 1, 1, 1}}; T_ia.set_read_only(true);
                DiskView<double, 2, 4> T_ja{(*T_ijab), Dim<2>{nact_, nvir_}, Count<4>{nact_, 1, 1, nvir_}, Offset<4>{0, i, a, 0}, Stride<4>{1, 1, 1, 1}}; T_ja.set_read_only(true);
                DiskView<double, 3, 4> Y_j{(*Y), Dim<3>{nact_, nvir_, nvir_}, Count<4>{nact_, 1, nvir_, nvir_}, Offset<4>{0, j, 0, 0}, Stride<4>{1, 1, 1, 1}}; Y_j.set_read_only(true);
                DiskView<double, 1, 2> beta_i{(*beta), Dim<1>{nact_}, Count<2>{nact_, 1}, Offset<2>{0, i}, Stride<2>{1, 1}}; beta_i.set_read_only(true);
                DiskView<double, 2, 4> tau_ja{(*tau), Dim<2>{nact_, nvir_}, Count<4>{nact_, 1, 1, nvir_}, Offset<4>{0, j, a, 0}, Stride<4>{1, 1, 1, 1}}; tau_ja.set_read_only(true);
                DiskView<double, 3, 4> Z_j{(*Z), Dim<3>{nact_, nvir_, nvir_}, Count<4>{nact_, 1, nvir_, nvir_}, Offset<4>{0, j, 0, 0}, Stride<4>{1, 1, 1, 1}}; Z_j.set_read_only(true);
                DiskView<double, 2, 4> Z_ja{(*Z), Dim<2>{nact_, nvir_}, Count<4>{nact_, 1, nvir_, 1}, Offset<4>{0, j, 0, a}, Stride<4>{1, 1, 1, 1}}; Z_ja.set_read_only(true);
                DiskView<double, 3, 4> T_i{(*T_ijab), Dim<3>{nact_, nvir_, nvir_}, Count<4>{nact_, 1, nvir_, nvir_}, Offset<4>{0, i, 0, 0}, Stride<4>{1, 1, 1, 1}}; T_i.set_read_only(true);
                auto G_ija = (*G)(i, j, a, All);

                Tensor<double, 1> tmp2 {"Temp Value", nact_};
                Tensor<double, 2> tmp {"Temp Value", nact_, nvir_};
                //einsum(1.0, Indices{k}, &tmp2, 1.0, Indices{p, q}, D_ij.get(), Indices{q, p, k}, K_vpqo_a.get());
                einsum(1.0, Indices{k}, &tmp2, 1.0, Indices{p, q}, D_ij.get(), Indices{p, q, k}, K_vpqo_a.get());
                sort(1.0, Indices{k}, &tmp2, 1.0, Indices{k}, K_ooov_ija.get());
                einsum(1.0, Indices{b}, &G_ija.get(), -1.0, Indices{k}, tmp2, Indices{k, b}, t_view.get());

                sort(1.0, Indices{k, c}, &tmp, 2.0, Indices{k, c}, T_ia.get());
                sort(1.0, Indices{k, c}, &tmp, -1.0, Indices{k, c}, T_ja.get());
                einsum(1.0, Indices{b}, &G_ija.get(), 1.0, Indices{k, c}, tmp, Indices{k, c, b}, Y_j.get());

                einsum(1.0, Indices{b}, &G_ija.get(), -1.0, Indices{k}, beta_i.get(), Indices{k, b}, tau_ja.get());
                einsum(1.0, Indices{b}, &G_ija.get(), -0.5, Indices{k, c}, T_ja.get(), Indices{k, c, b}, Z_j.get());
                einsum(1.0, Indices{b}, &G_ija.get(), -1.0, Indices{k, b, c}, T_i.get(), Indices{k, c}, Z_ja.get());
            }

            /*if (iteration_ < 2) {
                for (int b = 0; b < nvir_; b++) {
                    for (int a = 0; a < nvir_; a++) {
                        outfile->Printf("G(0, 0, %d, %d) = %e, ", b, a, G_ij(b, a));
                    }
                    outfile->Printf("\n");
                }
            }*/
        }
    }
}

void DiskCCSDF12B::form_V_ijab(einsums::DiskTensor<double, 4> *V_ijab, einsums::DiskTensor<double, 4> *G, einsums::DiskTensor<double, 4> *tau,
                           einsums::DiskTensor<double, 4> *D, einsums::DiskTensor<double, 4> *alpha, einsums::DiskTensor<double, 4> *C,
                           einsums::DiskTensor<double, 4> *FG, einsums::DiskTensor<double, 4> *K, einsums::DiskTensor<double, 4> *F)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    /* Include a the extra terms for CCSD-F12b */
    DiskView<double, 4, 4> V_ijab_view{(*V_ijab), Dim<4>{nact_, nact_, nvir_, nvir_}, Count<4>{nact_, nact_, nvir_, nvir_}, Offset<4>{0, 0, 0, 0}, Stride<4>{1, 1, 1, 1}};
    V_ijab_view.zero();

    size_t block_size = static_cast<size_t>(std::sqrt((((memory_ * 0.5)/ double_memory_) / (nobs_ - nfrzn_)) / (nobs_ - nfrzn_))); // Assume memory has been used elsewhere!
    int last_block = static_cast<int>(nvir_ % block_size);
    int no_blocks = static_cast<int>((nvir_ / block_size) + 1);
    /*
#pragma omp parallel for schedule(dynamic) collapse(2) num_threads(nthreads_)
    for (int i = 0; i < nact_; i++) {
        for (int j = 0; j < nact_; j++) {
            for (int a_block = 0; a_block < no_blocks; a_block++) {
                for (int b_block = 0; b_block < no_blocks; b_block++) {
                    int a_start = a_block * block_size;
                    int a_end = (a_block == no_blocks - 1) ? a_start + last_block : a_start + block_size;
                    int b_start = b_block * block_size;
                    int b_end = (b_block == no_blocks - 1) ? b_start + last_block : b_start + block_size;
                    auto F_pqoo_ij = (*F)(Range{nfrzn_, nobs_}, Range{nfrzn_, nobs_}, i, j); F_pqoo_ij.set_read_only(true);
                    //auto K_pvvq_ab = (*K)(Range{nfrzn_, nobs_}, Range{b_start, b_end}, Range{a_start, a_end}, Range{nfrzn_, nobs_}); K_pvvq_ab.set_read_only(true);
                    auto K_pvvq_ab = (*K)(Range{nfrzn_, nobs_}, Range{nfrzn_, nobs_}, Range{a_start, a_end}, Range{b_start, b_end}); K_pvvq_ab.set_read_only(true);
                    auto FG_oovv_ab = (*FG)(i, j, Range{a_start, a_end}, Range{b_start, b_end}); FG_oovv_ab.set_read_only(true);
                    auto C_ab = (*C)(i, j, Range{a_start, a_end}, Range{b_start, b_end}); C_ab.set_read_only(true);
                    
                    Tensor <double, 2> tmp {"Temp Value", a_end - a_start, b_end - b_start}; // As last block is different size
                    sort(1.0, Indices{a, b}, &tmp, 1.0, Indices{a, b}, FG_oovv_ab.get());
                    sort(1.0, Indices{a, b}, &tmp, 1.0, Indices{a, b}, C_ab.get());
                    //einsum(1.0, Indices{a, b}, &tmp, -1.0, Indices{p, b, a, q}, K_pvvq_ab.get(), Indices{p, q}, F_pqoo_ij.get());
                    einsum(1.0, Indices{a, b}, &tmp, -1.0, Indices{p, q, a, b}, K_pvvq_ab.get(), Indices{p, q}, F_pqoo_ij.get());

                    for (int a = a_start; a < a_end; a++) {
                        for (int b = b_start; b < b_end; b++) {
                            if (i != j) {
                                V_ijab_view(i, j, a, b) += (T_ijkl(i, j, j, i) * tmp(a, b)) + (T_ijkl(i, j, i, j) * tmp(a, b));
                            } else {
                                V_ijab_view(i, j, a, b) += T_ijkl(i, i, i, i) * tmp(a, b);
                            }
                        }
                    }
                }
            }
        }
    }*/

    /* Normal Residual terms */
    outfile->Printf("Normal terms\n");
#pragma omp parallel for schedule(dynamic) collapse(2) num_threads(nthreads_)
    for (int i = 0; i < nact_; i++) {
        for (int j = 0; j < nact_; j++) {
            for (int a_block = 0; a_block < no_blocks; a_block++) {
                for (int b_block = 0; b_block < no_blocks; b_block++) {
                    int a_start = a_block * block_size;
                    int a_end = (a_block == no_blocks - 1) ? a_start + last_block : a_start + block_size;
                    int b_start = b_block * block_size;
                    int b_end = (b_block == no_blocks - 1) ? b_start + last_block : b_start + block_size;
                    DiskView<double, 4, 4> tau_ab{(*tau), Dim<4>{nact_, nact_, a_end - a_start, b_end - b_start}, Count<4>{nact_, nact_, a_end - a_start, b_end - b_start}, Offset<4>{0, 0, a_start, b_start}, Stride<4>{1, 1, 1, 1}}; tau_ab.set_read_only(true);
                    DiskView<double, 2, 4> alpha_ij{(*alpha), Dim<2>{nact_, nact_}, Count<4>{1, 1, nact_, nact_}, Offset<4>{i, j, 0, 0}, Stride<4>{1, 1, 1, 1}}; alpha_ij.set_read_only(true);
                    auto D_ij = (*D)(i, j, All, All); D_ij.set_read_only(true);
                    //auto K_vpqv_ab = (*K)(Range{a_start + nocc_, a_end + nocc_}, Range{nfrzn_, nobs_}, Range{nfrzn_, nobs_}, Range{b_start + nocc_, b_end + nocc_}); K_vpqv_ab.set_read_only(true);
                    auto K_vpqv_ab = (*K)(Range{nfrzn_, nobs_}, Range{nfrzn_, nobs_}, Range{a_start + nocc_, a_end + nocc_}, Range{b_start + nocc_, b_end + nocc_}); K_vpqv_ab.set_read_only(true);
                    auto G_IJ = (*G)(i, j, Range{a_start, a_end}, Range{b_start, b_end}); G_IJ.set_read_only(true);
                    auto G_JI = (*G)(j, i, Range{b_start, b_end}, Range{a_start, a_end}); G_JI.set_read_only(true);
                    //auto K_oovv_ij = (*K)(Range{a_start + nocc_, a_end + nocc_}, j, i, Range{b_start + nocc_, b_end + nocc_}); K_oovv_ij.set_read_only(true);
                    auto K_oovv_ij = (*K)(i, j, Range{a_start + nocc_, a_end + nocc_}, Range{b_start + nocc_, b_end + nocc_}); K_oovv_ij.set_read_only(true);
                    
                    auto V_ijab_ij = (*V_ijab)(i, j, Range{a_start, a_end}, Range{b_start, b_end});
                    einsum(1.0, Indices{a, b}, &V_ijab_ij.get(), 1.0, Indices{p, q}, D_ij.get(), Indices{p, q, a, b}, K_vpqv_ab.get());
                    //einsum(1.0, Indices{a, b}, &V_ijab_ij.get(), 1.0, Indices{p, q}, D_ij.get(), Indices{a, q, p, b}, K_vpqv_ab.get());
                    einsum(1.0, Indices{a, b}, &V_ijab_ij.get(), 1.0, Indices{k, l}, alpha_ij.get(), Indices{k, l, a, b}, tau_ab.get());
                    sort(1.0, Indices{a, b}, &V_ijab_ij.get(), 1.0, Indices{a, b}, G_IJ.get());
                    sort(1.0, Indices{a, b}, &V_ijab_ij.get(), 1.0, Indices{b, a}, G_JI.get());
                    sort(1.0, Indices{a, b}, &V_ijab_ij.get(), 1.0, Indices{a, b}, K_oovv_ij.get());
                }
            }
        }
    }
}

}} // end namespaces