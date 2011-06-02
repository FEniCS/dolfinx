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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Benjamin Kehlet 2009
//
// First added:  2005-05-02
// Last changed: 2011-03-17

#include <dolfin/common/constants.h>
#include <dolfin/log/log.h>
#include <dolfin/math/Lagrange.h>
#include <dolfin/quadrature/LobattoQuadrature.h>
#include <dolfin/la/uBLASVector.h>
#include <dolfin/la/uBLASDenseMatrix.h>
#include <dolfin/la/HighPrecision.h>
#include <dolfin/ode/ODE.h>
#include "cGqMethod.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
cGqMethod::cGqMethod(uint q) : Method(q, q + 1, q)
{
  log(TRACE, "Initializing continuous Galerkin method cG(%d).", q);

  init();

  _type = Method::cG;

  p = 2*q;
}
//-----------------------------------------------------------------------------
real cGqMethod::ueval(real x0, real values[], real tau) const
{
  real sum = x0*trial->eval(0, tau);
  for (uint i = 0; i < nn; i++)
    sum += values[i] * trial->eval(i + 1, tau);

  return sum;
}
//-----------------------------------------------------------------------------
real cGqMethod::residual(real x0, real values[], real f, real k) const
{
  real sum = x0*derivatives[0];
  for (uint i = 0; i < nn; i++)
    sum += values[i]*derivatives[i + 1];

  return sum/k - f;
}
//-----------------------------------------------------------------------------
real cGqMethod::timestep(real r, real tol, real k0, real kmax) const
{
  // FIXME: Missing stability factor and interpolation constant

  if ( real_abs(r) < real_epsilon() )
    return kmax;

  const real qq = static_cast<real>(q);
  return real_pow(tol*real_pow(k0, q)/real_abs(r), 0.5/qq);
}
//-----------------------------------------------------------------------------
real cGqMethod::error(real k, real r) const
{
  // FIXME: Missing interpolation constant
  return real_pow(k, static_cast<real>(q))*real_abs(r);
}
//-----------------------------------------------------------------------------
void cGqMethod::get_nodal_values(const real& u0, const real* x,
                                 real* nodal_values) const
{
  nodal_values[0] = u0;
  for (uint i = 0; i < nn; i++)
    nodal_values[i+1] = x[i];
}
//-----------------------------------------------------------------------------
std::string cGqMethod::str(bool verbose) const
{
  std::stringstream s;

  if (verbose)
  {
    s << str(false) << std::endl << std::endl;

    s << std::setiosflags(std::ios::scientific) << std::setprecision(16);

    s << "  Lobatto quadrature points and weights on [0, 1]" << std::endl;
    s << "  -----------------------------------------------" << std::endl;
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
  {
    s << "<cGqMethod for q = " << q << ">";
  }

  return s.str();
}
//-----------------------------------------------------------------------------
void cGqMethod::compute_quadrature()
{
  // Use Lobatto quadrature
  LobattoQuadrature quadrature(nq);

  // Get quadrature points and rescale from [-1,1] to [0,1]
  for (uint i = 0; i < nq; i++)
    qpoints[i] = (quadrature.point(i) + 1.0) / 2.0;

  // Get nodal points and rescale from [-1,1] to [0,1]
  for (uint i = 0; i < nn; i++)
    npoints[i] = (quadrature.point(i + 1) + 1.0) / 2.0;

  // Get quadrature weights and rescale from [-1,1] to [0,1]
  for (uint i = 0; i < nq; i++)
    qweights[i] = 0.5 * quadrature.weight(i);
}
//-----------------------------------------------------------------------------
void cGqMethod::compute_basis()
{
  assert(!trial);
  assert(!test);

  // Compute Lagrange basis for trial space
  trial = new Lagrange(q);
  for (uint i = 0; i < nq; i++)
    trial->set(i, qpoints[i]);

  // Compute Lagrange basis for test space using the Lobatto points for q - 1
  test = new Lagrange(q - 1);
  if ( q > 1 )
  {
    LobattoQuadrature lobatto(nq - 1);
    for (uint i = 0; i < (nq - 1); i++)
      test->set(i, (lobatto.point(i) + 1.0) / 2.0);
  }
  else
    test->set(0, 1.0);
}
//-----------------------------------------------------------------------------
void cGqMethod::compute_weights()
{
  uBLASDenseMatrix A(q, q);
  ublas_dense_matrix& _A = A.mat();

  std::vector<real> A_real(q*q);

  std::vector<real> trial_ddx(nn*nq);
  std::vector<real> test_eval(nn*nq);

  for (uint a = 0; a < nn; ++a)
  {
    for (uint b = 0; b < nq; ++b)
    {
      trial_ddx[a + nn*b] = trial->ddx(a + 1, qpoints[b]);
      test_eval[a + nn*b] = test->eval(a, qpoints[b]);
    }
  }

  // Compute matrix coefficients
  for (uint i = 0; i < nn; i++)
  {
    for (uint j = 0; j < nn; j++)
    {
      // Use Lobatto quadrature which is exact for the order we need, 2q-1
      real integral = 0.0;
      for (uint k = 0; k < nq; k++)
      {
        //real x = qpoints[k];
        //integral += qweights[k] * trial->ddx(j + 1, x) * test->eval(i, x);
        integral += qweights[k] * trial_ddx[j + nn*k] * test_eval[i + nn*k];
      }

      A_real[i + nn*j] = integral;
      _A(i, j) = to_double(integral);
    }
  }

  //A.disp();

  uBLASVector b(q);
  ublas_vector& _b = b.vec();
  std::vector<real> b_real(q);

  // Compute nodal weights for each degree of freedom (loop over points)
  for (uint i = 0; i < nq; i++)
  {
    // Get nodal point
    //real x = qpoints[i];

    // Evaluate test functions at current nodal point
    for (uint j = 0; j < nn; j++)
    {
      b_real[j] = test_eval[i*nn + j];
      _b[j] = to_double(b_real[j]);
    }

    //b.disp();

#ifndef HAS_GMP
    uBLASVector w(q);
    ublas_vector& _w = w.vec();

    // Solve for the weight functions at the nodal point
    A.solve(w, b);

    // Save weights including quadrature
    for (uint j = 0; j < nn; j++)
      nweights[j][i] = qweights[i] * _w[j];
#else
    std::vector<real> w_real(q);

    uBLASDenseMatrix A_inv(A);
    A_inv.invert();

    // Solve system using the double precision invert as preconditioner
    HighPrecision::real_solve_precond(q, &A_real[0], &w_real[0], &b_real[0], A_inv, real_epsilon());

    for (uint j = 0; j < nn; ++j)
      nweights[j][i] = qweights[i] * w_real[j];
#endif
  }
}
//-----------------------------------------------------------------------------
