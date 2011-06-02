// Copyright (C) 2003-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-05-02
// Last changed: 2011-03-17

#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/Lagrange.h>
#include <dolfin/quadrature/RadauQuadrature.h>
#include <dolfin/la/uBLASVector.h>
#include <dolfin/la/uBLASDenseMatrix.h>
#include <dolfin/la/HighPrecision.h>
#include <dolfin/ode/ODE.h>
#include "dGqMethod.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dGqMethod::dGqMethod(unsigned int q) : Method(q, q + 1, q + 1)
{
  log(TRACE, "Initializing discontinuous Galerkin method dG(%d).", q);

  init();

  _type = Method::dG;

  p = 2*q + 1;
}
//-----------------------------------------------------------------------------
real dGqMethod::ueval(real x0, real values[], real tau) const
{
  // Note: x0 is not used, maybe this can be done differently

  real sum = 0.0;
  for (unsigned int i = 0; i < nn; i++)
    sum += values[i] * trial->eval(i, tau);

  return sum;
}
//-----------------------------------------------------------------------------
real dGqMethod::residual(real x0, real values[], real f, real k) const
{
  // FIXME: Include jump term in residual
  real sum = 0.0;
  for (uint i = 0; i < nn; i++)
    sum += values[i] * derivatives[i];

  return sum / k - f;
}
//-----------------------------------------------------------------------------
real dGqMethod::timestep(real r, real tol, real k0, real kmax) const
{
  // FIXME: Missing stability factor and interpolation constant
  // FIXME: Missing jump term

  if ( real_abs(r) < real_epsilon() )
    return kmax;

  //return pow(tol / fabs(r), 1.0 / static_cast<real>(q+1));

  const real qq = static_cast<real>(q);
  return real_pow(tol*real_pow(k0, q)/real_abs(r), 1.0/(2.0*qq + 1.0));
}
//-----------------------------------------------------------------------------
real dGqMethod::error(real k, real r) const
{
  // FIXME: Missing jump term and interpolation constant
  return real_pow(k, static_cast<real>(q + 1))*real_abs(r);
}
//-----------------------------------------------------------------------------
void dGqMethod::get_nodal_values(const real& x0, const real* x,
                                 real* nodal_values) const
{
  real_set(nn, nodal_values, x);
}
//-----------------------------------------------------------------------------
std::string dGqMethod::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << std::setiosflags(std::ios::scientific) << std::setprecision(16);

    s << "  Radau quadrature points and weights on [0, 1]" << std::endl;
    s << "  ---------------------------------------------" << std::endl;
    s << "" << std::endl;
    s << "    i  points                  weights" << std::endl;
    s << "    ----------------------------------------------------" << std::endl;
    for (unsigned int i = 0; i < nq; i++)
      s << "    " << i << "  "
        << to_double(qpoints[i]) << "  "
        << to_double(qweights[i]) << std::endl;

    for (unsigned int i = 0; i < nn; i++)
    {
      s << std::endl;
      s << "  Weights for degree of freedom " << i << ": " << std::endl;
      s << "  --------------------------------";
      s << "" << std::endl;
      for (unsigned int j = 0; j < nq; j++)
        s << "  " << j << "  " << nweights[i][j];
      s << std::endl;
    }

    s << std::endl;
    s << "  Weights in matrix format" << std::endl;
    s << "  ------------------------" << std::endl;
    for (unsigned int i = 0; i < nn; i++)
    {
      s << "  ";
      for (unsigned int j = 0; j < nq; j++)
        s << nweights[i][j] << " ";
      s << std::endl;
    }
  }
  else
    s << "<dGqMethod for q = " << q << ">";

  return s.str();
}
//-----------------------------------------------------------------------------
void dGqMethod::compute_quadrature()
{
  // Use Radau quadrature
  RadauQuadrature quadrature(nq);

  // Get points, rescale from [-1,1] to [0,1], and reverse the points
  for (unsigned int i = 0; i < nq; i++)
  {
    qpoints[i] = 1.0 - (quadrature.point(nq - 1 - i) + 1.0)/2.0;
    npoints[i] = qpoints[i];
  }

  // Get points, rescale from [-1,1] to [0,1], and reverse the points
  for (unsigned int i = 0; i < nq; i++)
    qweights[i] = 0.5*quadrature.weight(nq - 1 - i);
}
//-----------------------------------------------------------------------------
void dGqMethod::compute_basis()
{
  assert(!trial);
  assert(!test);

  // Compute Lagrange basis for trial space
  trial = new Lagrange(q);
  for (unsigned int i = 0; i < nq; i++)
    trial->set(i, qpoints[i]);

  // Compute Lagrange basis for test space
  test = new Lagrange(q);
  for (unsigned int i = 0; i < nq; i++)
    test->set(i, qpoints[i]);
}
//-----------------------------------------------------------------------------
void dGqMethod::compute_weights()
{
  uBLASDenseMatrix A(nn, nn);
  ublas_dense_matrix& _A = A.mat();

  std::vector<real> A_real(nn*nn);
  real_zero(nn*nn, &A_real[0]);

  std::vector<real> trial_ddx(nn*nq);
  std::vector<real> test_eval(nn*nq);

  std::vector<real> trial_eval_0(nn);
  std::vector<real> test_eval_0(nn);

  for (uint a = 0; a < nn; ++a)
  {
    trial_eval_0[a] = trial->eval(a, 0.0);
    test_eval_0[a]  = test->eval(a, 0.0);

    for (uint b = 0; b < nq; ++b)
    {
      trial_ddx[a + nq*b] = trial->ddx(a, qpoints[b]);
      test_eval[a + nq*b] = test->eval(a, qpoints[b]);
    }
  }

  // Compute matrix coefficients
  for (unsigned int i = 0; i < nn; i++)
  {
    for (unsigned int j = 0; j < nn; j++)
    {
      // Use Radau quadrature which is exact for the order we need, 2q
      real integral = 0.0;
      for (unsigned int k = 0; k < nq; k++)
      {
        //real x = qpoints[k];
        //integral += qweights[k] * trial->ddx(j, x) * test->eval(i, x);
        integral += qweights[k] * trial_ddx[j + nq*k] * test_eval[i + nq*k];
      }

      A_real[i + nn*j] = integral + trial_eval_0[j] * test_eval_0[i];
      _A(i, j) = to_double(A_real[i + nn*j]);
    }
  }

  uBLASVector b(nn);
  ublas_vector& _b = b.vec();

  std::vector<real> b_real(nn);

  // Compute nodal weights for each degree of freedom (loop over points)
  for (unsigned int i = 0; i < nq; i++)
  {
    // Get nodal point
    //real x = qpoints[i];

    // Evaluate test functions at current nodal point
    for (unsigned int j = 0; j < nn; j++)
    {
      b_real[j] = test_eval[i + j*nq];
      _b[j] = to_double(b_real[j]);
    }

    #ifndef HAS_GMP
    uBLASVector w(nn);
    ublas_vector& _w = w.vec();

    // Solve for the weight functions at the nodal point
    A.solve(w, b);

    // Save weights including quadrature
    for (uint j = 0; j < nn; j++)
      nweights[j][i] = qweights[i] * _w[j];
    #else
    std::vector<real> w_real(nn);

    // Solve system using the double precision invert as preconditioner
    uBLASDenseMatrix A_inv(A);
    A_inv.invert();

    HighPrecision::real_solve_precond(nn, &A_real[0], &w_real[0], &b_real[0],
                                      A_inv, real_epsilon());

    for (uint j = 0; j < nn; ++j)
      nweights[j][i] = qweights[i] * w_real[j];
    #endif
  }
}
//-----------------------------------------------------------------------------
