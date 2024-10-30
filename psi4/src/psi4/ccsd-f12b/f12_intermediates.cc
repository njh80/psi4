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
    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const bool full_basis = ((*D).dim(2) == nobs_)? true : false;
    const auto start = (use_frzn) ? 0 : nfrzn_;
    const auto otherStart = (full_basis) ? start : nocc_; // All Orbital Basis Functions {p}, or Active/Frozen Orbital Basis Functions {i}

    if (full_basis) { // If just virtuals, r and s cannot == i or j therefore there is no E^{ij} contribution
        for (auto i = start; i < nocc_; i++) {
            for (auto j = start; j < nocc_; j++) {
                for (auto r = start; r < nobs_; r++) {
                    for (auto s = start; s < nobs_; s++) { // Eij rc = dri tjc ; Eji rc transpose = drj tic transpose ; Eji rc transpose = drj tic transpose, but... tjc = tic ???
                        if (r == i && s >= nocc_) {
                            (*D)(i, j, r, s) += (*t_i)(j, s); 
                        }
                        if (s == j && r >= nocc_) {
                            (*D)(i, j, r, s) += (*t_i)(i, r);
                        }
                        if (r >= nocc_ && s >= nocc_) { // Add virtual contributions
                            (*D)(i, j, r, s) += (*tau)(i, j, r, s);
                        } else { // Add explicitly correlated contributions
                            (*D)(i, j, r, s) += T_ijkl(i, j, r, s);
                        }
                    }
                }
            }
        }
    } else { // Can add virtual contributions to all elements as dimensions identical
        sort(1.0, Indices{i, j, a, b}, &(*D), 1.0, Indices{i, j, a, b}, *tau);
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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto start = (use_frzn) ? 0 : nfrzn_;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    form_teints("K", K_pqrs.get(), {'O', 'O', 'O', 'O'});
    Tensor<double, 2> tmp_trace {"Trace Term", dim1, dim1};
    {
        Tensor K_oovv = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor<double, 4> L_oovv {"L_oovv", nact_, nact_, nvir_, nvir_};
        form_L(&L_oovv, &K_oovv);
        /* Trace Term */
        einsum(Indices{k, i}, &tmp_trace, Indices{k, l, a, b}, L_oovv, Indices{i, l, a, b}, *tau);
    }

    sort(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, i}, *f); // Populate beta with fock matrix
    einsum(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, a}, *f, Indices{i, a}, *t); // Add f^k t^i to beta (subscripts a on both)
    
    {
        Tensor<double, 2> tmp_sum {"Sum Term", dim1, dim1};
        Tensor K_ooov = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor<double, 4> L_ooov {"L_ooov", nact_, nact_, nact_, nvir_};
        form_L(&L_ooov, &K_ooov);
        einsum(1.0, Indices{k, i}, &tmp_sum, 1.0, Indices{l, a}, *t, Indices{l, k, i, a}, L_ooov);
        sort(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, i}, tmp_sum);
    }

    for (auto i = start; i < nocc_; i++) {
        (*beta)(i, i) += tmp_trace(i, i);
    }
}

void CCSDF12B::form_df_beta(einsums::Tensor<double, 2> *beta, einsums::Tensor<double, 2> *f, einsums::Tensor<double, 2> *t, 
                            einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto start = (use_frzn) ? 0 : nfrzn_;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    form_df_teints("K", K_pqrs.get(), J_inv_AB, {'O', 'O', 'O', 'O'});
    Tensor<double, 2> tmp_trace {"Trace Term", dim1, dim1};
    {
        Tensor K_oovv = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor<double, 4> L_oovv {"L_oovv", nact_, nact_, nvir_, nvir_};
        form_L(&L_oovv, &K_oovv);
        /* Trace Term */
        einsum(Indices{k, i}, &tmp_trace, Indices{k, l, a, b}, L_oovv, Indices{i, l, a, b}, *tau);
    }

    sort(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, i}, *f); // Populate beta with fock matrix
    einsum(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, a}, *f, Indices{i, a}, *t); // Add f^k t^i to beta (subscripts a on both)
    
    {
        Tensor<double, 2> tmp_sum {"Sum Term", dim1, dim1};
        Tensor K_ooov = (*K_pqrs)(Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor<double, 4> L_ooov {"L_ooov", nact_, nact_, nact_, nvir_};
        form_L(&L_ooov, &K_ooov);
        einsum(1.0, Indices{k, i}, &tmp_sum, 1.0, Indices{l, a}, *t, Indices{l, k, i, a}, L_ooov);
        sort(1.0, Indices{k, i}, &(*beta), 1.0, Indices{k, i}, tmp_sum);
    }

    for (auto i = start; i < nocc_; i++) {
        (*beta)(i, i) += tmp_trace(i, i);
    }
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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto h_xx = std::make_unique<Tensor<double, 2>>("h_xx", nri_, nri_); 
    form_oeints(h_xx.get());
    Tensor h_vv = (*h_xx)(Range{nocc_, nobs_}, Range{nocc_, nobs_});

    sort(1.0, Indices{i, a}, &(*s), 1.0, Indices{i, a}, *f);

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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto h_xx = std::make_unique<Tensor<double, 2>>("h_xx", nri_, nri_);
    form_oeints(h_xx.get());
    Tensor h_vv = (*h_xx)(Range{nocc_, nobs_}, Range{nocc_, nobs_});

    sort(1.0, Indices{i, a}, &(*s), 1.0, Indices{i, a}, *f);

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

    sort(1.0, Indices{i, a}, &(*r), 1.0, Indices{i, a}, *f);
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

    sort(1.0, Indices{i, a}, &(*r), 1.0, Indices{i, a}, *f);
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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    Tensor<double, 4> tmp {"Temp", dim1, dim1, nvir_, nvir_};
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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    sort(1.0, Indices{a, b}, &(*X), 1.0, Indices{a, b}, *f);
    sort(1.0, Indices{a, b}, &(*X), -1.0, Indices{a, b}, *A);
    einsum(1.0, Indices{a, b}, &(*X), -1.0, Indices{k, a}, *r, Indices{k, b}, *t);
    
    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", dim1, nobs_, nobs_, nobs_);
    auto K_opqr = std::make_unique<Tensor<double, 4>>("K_opqr", dim1, nobs_, nobs_, nobs_);
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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    sort(1.0, Indices{a, b}, &(*X), 1.0, Indices{a, b}, *f);
    sort(1.0, Indices{a, b}, &(*X), -1.0, Indices{a, b}, *A);
    einsum(1.0, Indices{a, b}, &(*X), -1.0, Indices{k, a}, *r, Indices{k, b}, *t);
    
    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", dim1, nobs_, nobs_, nobs_);
    auto K_opqr = std::make_unique<Tensor<double, 4>>("K_opqr", dim1, nobs_, nobs_, nobs_);
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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", dim1, nobs_, nobs_, nobs_);
    auto K_opqr = std::make_unique<Tensor<double, 4>>("K_opqr", dim1, nobs_, nobs_, nobs_);

    form_teints("J", J_opqr.get(), {'o', 'O', 'O', 'O'});
    form_teints("K", K_opqr.get(), {'o', 'O', 'O', 'O'});
    
    {
        Tensor J_oovv = (*J_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor K_oovv = (*K_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        sort(1.0, Indices{i, j, a, b}, &(*Y), 1.0, Indices{i, j, a, b}, K_oovv);
        sort(1.0, Indices{i, j, a, b}, &(*Y), -0.5, Indices{i, j, a, b}, J_oovv);

    }

    {
        Tensor J_ovvv = (*J_opqr)(All, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor K_ovvv = (*K_opqr)(All, Range{nocc_, nobs_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        einsum(1.0, Indices{i, j, a, b}, &(*Y), -0.5, Indices{a, b, i, c}, J_ovvv, Indices{j, c}, *t);
        einsum(1.0, Indices{i, j, a, b}, &(*Y), 1.0, Indices{a, i, b, c}, K_ovvv, Indices{j, c}, *t);
    }

    Tensor<double, 4> tmp {"Temp", dim1, dim1, nvir_, nvir_};
    sort(1.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, *taut);
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, b, a}, *taut);
    {
        Tensor K_oovv = (*K_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor<double, 4> L_oovv {"L_oovv", nact_, nvir_, nvir_, nvir_};
        form_L(&L_oovv, &K_oovv);
        einsum(1.0, Indices{i, j, a, b}, &(*Y), 0.5, Indices{i, j, a, b}, L_oovv, Indices{i, j, a, b}, tmp);
    }

    einsum(1.0, Indices{i, j, a, b}, &(*Y), 1.0, Indices{i, a}, *f, Indices{j, b}, *t);

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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", dim1, nobs_, nobs_, nobs_);
    auto K_opqr = std::make_unique<Tensor<double, 4>>("K_opqr", dim1, nobs_, nobs_, nobs_);

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
        einsum(1.0, Indices{i, j, a, b}, &(*Y), -0.5, Indices{a, b, i, c}, J_ovvv, Indices{j, c}, *t);
        einsum(1.0, Indices{i, j, a, b}, &(*Y), 1.0, Indices{a, i, b, c}, K_ovvv, Indices{j, c}, *t);
    }

    Tensor<double, 4> tmp {"Temp", dim1, dim1, nvir_, nvir_};
    sort(1.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, *taut);
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, b, a}, *taut);
    {
        Tensor K_oovv = (*K_opqr)(All, Range{nfrzn_, nocc_}, Range{nocc_, nobs_}, Range{nocc_, nobs_});
        Tensor<double, 4> L_oovv {"L_oovv", nact_, nvir_, nvir_, nvir_};
        form_L(&L_oovv, &K_oovv);
        einsum(1.0, Indices{i, j, a, b}, &(*Y), 0.5, Indices{i, j, a, b}, L_oovv, Indices{i, j, a, b}, tmp);
    }

    einsum(1.0, Indices{i, j, a, b}, &(*Y), 1.0, Indices{i, a}, *f, Indices{j, b}, *t);

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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", dim1, nobs_, nobs_, nobs_);
    auto K_oopq = std::make_unique<Tensor<double, 4>>("K_oopq", dim1, dim1, nobs_, nobs_);

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
        einsum(1.0, Indices{k, j, a, b}, &(*Z), -1.0, Indices{l, k, l, a}, K_ooov, Indices{l, b}, *t);
    }

}

void CCSDF12B::form_df_Z_Werner(einsums::Tensor<double, 4> *Z, einsums::Tensor<double, 4> *taut, einsums::Tensor<double, 2> *t, 
                                einsums::Tensor<double, 3> *J_inv_AB)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto J_opqr = std::make_unique<Tensor<double, 4>>("J_opqr", dim1, nobs_, nobs_, nobs_);
    auto K_oopq = std::make_unique<Tensor<double, 4>>("K_oopq", dim1, dim1, nobs_, nobs_);

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
        einsum(1.0, Indices{k, j, a, b}, &(*Z), -1.0, Indices{l, k, l, a}, K_ooov, Indices{l, b}, *t);
    }

}

void CCSDF12B::form_alpha(einsums::Tensor<double, 4> *alpha, einsums::Tensor<double, 4> *tau, einsums::Tensor<double, 2> *t)
{
    using namespace einsums;
    using namespace tensor_algebra;
    using namespace tensor_algebra::index;

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto K_oopq = std::make_unique<Tensor<double, 4>>("K_oopq", dim1, dim1, nobs_, nobs_);
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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto K_oopq = std::make_unique<Tensor<double, 4>>("K_oopq", dim1, dim1, nobs_, nobs_);
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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    form_teints("K", K_pqrs.get(), {'O', 'O', 'O', 'O'});

    Tensor<double, 4> tmp {"Temp", dim1, dim1, nvir_, nvir_};
    sort(1.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, *T_ijab);
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, b, a}, *T_ijab);

    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, a}, *s, Indices{j, b}, *t);
    einsum(1.0, Indices{i, j, a, b}, &(*G), -1.0, Indices{k, i}, *beta, Indices{k, j, a, b}, *tau);
    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, j, a, b}, *T_ijab, Indices{a, b}, *X);
    einsum(1.0, Indices{j, i, a, b}, &(*G), -1.0, Indices{k, i, a, b}, *T_ijab, Indices{k, j, a, b}, *Z);
    einsum(1.0, Indices{i, j, a, b}, &(*G), -0.5, Indices{k, i, a, b}, *T_ijab, Indices{k, j, a, b}, *Z);
    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, k, a, b}, tmp, Indices{k, j, a, b}, *Y);

    Tensor<double, 4> tmp2 {"Temp2", dim1, dim1, dim1, nvir_};
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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    form_df_teints("K", K_pqrs.get(), J_inv_AB, {'O', 'O', 'O', 'O'});

    Tensor<double, 4> tmp {"Temp", dim1, dim1, nvir_, nvir_};
    sort(1.0, Indices{i, j, a, b}, &tmp, 2.0, Indices{i, j, a, b}, *T_ijab);
    sort(1.0, Indices{i, j, a, b}, &tmp, -1.0, Indices{j, i, b, a}, *T_ijab);

    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, a}, *s, Indices{j, b}, *t);
    einsum(1.0, Indices{i, j, a, b}, &(*G), -1.0, Indices{k, i}, *beta, Indices{k, j, a, b}, *tau);
    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, j, a, b}, *T_ijab, Indices{a, b}, *X);
    einsum(1.0, Indices{j, i, a, b}, &(*G), -1.0, Indices{k, i, a, b}, *T_ijab, Indices{k, j, a, b}, *Z);
    einsum(1.0, Indices{i, j, a, b}, &(*G), -0.5, Indices{k, i, a, b}, *T_ijab, Indices{k, j, a, b}, *Z);
    einsum(1.0, Indices{i, j, a, b}, &(*G), 1.0, Indices{i, k, a, b}, tmp, Indices{k, j, a, b}, *Y);

    Tensor<double, 4> tmp2 {"Temp2", dim1, dim1, dim1, nvir_};
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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    auto W_oopq = std::make_unique<Tensor<double, 4>>("W_oopq", dim1, dim1, nobs_, nobs_);
    auto F_pqoo = std::make_unique<Tensor<double, 4>>("F_pqoo", nobs_, nobs_, dim1, dim1);
    form_teints("K", K_pqrs.get(), {'O', 'O', 'O', 'O'});
    form_teints("FG", W_oopq.get(), {'o', 'O', 'o', 'O'});
    form_teints("F", F_pqoo.get(), {'O', 'o', 'O', 'o'});

    Tensor<double, 4> tmp {"Temp", dim1, dim1, nvir_, nvir_};

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

    for (auto i = 0; i < dim1; i++)
    {
        for (auto j = 0; j < dim1; j++)
        {
            for (auto a = 0; a < nvir_; a++)
            {
                for (auto b = 0; b < nvir_; b++)
                {
                    (*V_ijab)(i, j, a, b) = T_ijkl(i, j, a, b) * tmp(i, j, a, b);
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

    const bool use_frzn = (nfrzn_ > 0) ? true : false;
    const auto dim1 = (use_frzn) ? nocc_ : nact_;

    auto K_pqrs = std::make_unique<Tensor<double, 4>>("K_pqrs", nobs_, nobs_, nobs_, nobs_);
    auto W_oopq = std::make_unique<Tensor<double, 4>>("W_oopq", dim1, dim1, nobs_, nobs_);
    auto F_pqoo = std::make_unique<Tensor<double, 4>>("F_pqoo", nobs_, nobs_, dim1, dim1);
    form_df_teints("K", K_pqrs.get(), J_inv_AB, {'O', 'O', 'O', 'O'});
    form_df_teints("FG", W_oopq.get(), J_inv_AB, {'o', 'O', 'o', 'O'});
    form_df_teints("F", F_pqoo.get(), J_inv_AB, {'O', 'o', 'O', 'o'});

    Tensor<double, 4> tmp {"Temp", dim1, dim1, nvir_, nvir_};

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

    for (auto i = 0; i < dim1; i++)
    {
        for (auto j = 0; j < dim1; j++)
        {
            for (auto a = 0; a < nvir_; a++)
            {
                for (auto b = 0; b < nvir_; b++)
                {
                    (*V_ijab)(i, j, a, b) = T_ijkl(i, j, a, b) * tmp(i, j, a, b);
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
    auto f_view = (*f)(All, All);

    {
        outfile->Printf("     Forming J\n");
        auto J = DiskTensor<double, 4>{state::data(), "Coulomb", nri_, nocc_, nri_, nocc_};
        if (!J.existed()) form_teints("J", &J);

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
        if (!K.existed()) form_teints("K", &K);

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
    auto f_view = (*f)(All, All);

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
                auto G_F_IJ_oc = (*G_F)(I, J, Range{0, nocc_}, Range{nobs_, nri_}); G_F_IJ_oc.set_read_only(true);
                auto G_F_JI_oc = (*G_F)(J, I, Range{0, nocc_}, Range{nobs_, nri_}); G_F_JI_oc.set_read_only(true);
                auto F_IJ_oc = (*F)(I, J, Range{0, nocc_}, Range{nobs_, nri_}); F_IJ_oc.set_read_only(true);
                auto F_JI_oc = (*F)(J, I, Range{0, nocc_}, Range{nobs_, nri_}); F_JI_oc.set_read_only(true);

                einsum(Indices{}, &tmp1, Indices{m, q}, G_F_IJ_oc.get(), Indices{m, q}, F_IJ_oc.get());
                einsum(1.0, Indices{}, &tmp1, 1.0, Indices{m, q}, G_F_JI_oc.get(), Indices{m, q}, F_JI_oc.get());

                if (I != J) {
                    einsum(Indices{}, &tmp2, Indices{m, q}, G_F_IJ_oc.get(), Indices{m, q}, F_JI_oc.get());
                    einsum(1.0, Indices{}, &tmp2, 1.0, Indices{m, q}, G_F_JI_oc.get(), Indices{m, q}, F_IJ_oc.get());
                }
            }

            // Term 4
            {
                auto G_F_IJ_pq = (*G_F)(I, J, Range{0, nobs_}, Range{0, nobs_}); G_F_IJ_pq.set_read_only(true);
                auto G_F_JI_pq = (*G_F)(J, I, Range{0, nobs_}, Range{0, nobs_}); G_F_JI_pq.set_read_only(true);
                auto F_IJ_pq = (*F)(I, J, Range{0, nobs_}, Range{0, nobs_}); F_IJ_pq.set_read_only(true);

                einsum(1.0, Indices{}, &tmp1, 1.0, Indices{p, q}, F_IJ_pq.get(), Indices{p, q}, G_F_IJ_pq.get());

                if (I != J) {
                    einsum(1.0, Indices{}, &tmp2, 1.0, Indices{p, q}, F_IJ_pq.get(), Indices{p, q}, G_F_JI_pq.get());
                }
            }

            int start = 0, stop = nact_;
            if ((*VX).name() == "X Intermediate Tensor") { start = nfrzn_, stop = nocc_; }
            auto FG_F2_IJ = (*FG_F2)(I, J, All, Range{start, stop}); FG_F2_IJ.set_read_only(true); // Term 1

            auto VX_IJ = (*VX)(I, J, All, All);
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

}} // end namespaces