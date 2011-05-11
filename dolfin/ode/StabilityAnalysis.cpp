// Copyright (C) 2009 Benjamin Kehlet
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
// First added:  2009-07-20
// Last changed: 2010-06-21

#include "StabilityAnalysis.h"
#include "ODESolution.h"
#include "ODE.h"
#include <dolfin/log/Logger.h>
#include <dolfin/log/Progress.h>
#include <dolfin/io/PythonFile.h>
#include <dolfin/common/real.h>
#include <dolfin/common/Array.h>
#include <dolfin/la/HighPrecision.h>
#include <vector>
#include <boost/scoped_array.hpp>

using namespace dolfin;

//-----------------------------------------------------------------------------
StabilityAnalysis::StabilityAnalysis(ODE& ode, ODESolution& u) :
  ode(ode), u(u), write_to_file(ode.parameters["save_solution"]), n(ode.size())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
StabilityAnalysis::~StabilityAnalysis()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void StabilityAnalysis::analyze_integral(uint q)
{
  begin("Computing stability factors");

  double progress_end = to_double(ode.endtime()*ode.endtime() + ode.endtime());

  // Collect
  std::vector< std::pair<real, real*> > s;

  Array<real> tmp(n);
  //boost::scoped_array<real> tmp_array(new real [n]); real* tmp = tmp_array.get();
  boost::scoped_array<real> A_array(new real[n*n]);  real* A = A_array.get();
  boost::scoped_array<real> B_array(new real[n*n]);  real* B = B_array.get();

  uint count = 0;

  PythonFile file("stability_factors.py");

  Progress p("Computing stability factors");

  // How should the length of the timestep be decided?
  for (  ODESolution::iterator it = u.begin(); it != u.end(); it++ )
  {
    // Get data for current time step
    ODESolutionData& timestep = *it;
    real& t = timestep.a;

    // Check if we have reached end time
    if (t > ode.endtime())
      break;

    // Allocate matrices to be pushed on s, will be deleted at end of function
    real* C = new real[n*n];

    // Get solution values at first nodal point on interval
    timestep.eval_a(tmp.data().get());

    // Get transpose of Jacobian
    get_JT(A, tmp, t);

    // Multiply by A^q to differentiate q times: C = JT^q
    HighPrecision::real_mat_pow(n, C, A, q);

    // Multiply A with length of timestep: A = k*JT
    real_mult(n*n, A, timestep.k);

    // Compute matrix exponential: B = e^(k*JT)
    HighPrecision::real_mat_exp(n, B, A, 10);

    // Multiply each matrix in s with B from right
    for (std::vector< std::pair<real, real*> >::iterator s_iterator = s.begin();
	        s_iterator != s.end(); ++s_iterator)
    {
      HighPrecision::real_mat_prod_inplace(n, (*s_iterator).second, B);
    }

    // Differentiate: C = JT^q * e^(k*JT)
    HighPrecision::real_mat_prod_inplace(n, C, B);

    // Store differentiated fundamental solution
    s.push_back(std::pair<real, real*>(t + timestep.k, C));

    // Now compute the stability factor for T = t by integrating
    Array<real> sample(n);
    sample.zero();

    real prev = 0.0;
    for (std::vector< std::pair<real, real*> >::iterator s_iterator = s.begin();
	 s_iterator != s.end(); ++s_iterator)
    {
      // Get time and fundamental solution (matrix Z)
      real t  = s_iterator->first;
      real* Z = s_iterator->second;

      // Initial data is unit vectors so we don't need to multiply, just pick columns in Z
      for (uint i = 0; i < n; ++i)
      {
        // Compute norm of (differentiated) dual solution
        real norm = real_norm(n, &Z[n*i]);

        // Add to integral (for computing L^1 norm in time)
        sample[i] += norm * real_abs(t - prev);

        // Add to integral (for computing L^2 norm in time)
        //sample[i] += norm * norm * real_abs(t-prev);
      }

      prev = t;
    }

    // Take square root (for computing L^2 norm in time)
    //for (uint i = 0; i < n; i++)
      //sample[i] = real_sqrt(sample[i]);

    // Store to file
    file << std::pair<real, RealArrayRef>(t, RealArrayRef(sample));

    // Update progress
    const double t_double = to_double(t);
    p = (t_double * t_double + t_double) / progress_end;
    count++;
  }

  // Delete the allocated C matrices
  for (std::vector< std::pair<real, real*> >::iterator s_iterator = s.begin();
       s_iterator != s.end(); ++s_iterator)
  {
    real* Z = s_iterator->second;
    delete [] Z;
  }

  end();
}
//-----------------------------------------------------------------------------
void StabilityAnalysis::analyze_endpoint()
{
  begin("Computing stability factor");

  Progress p("Computing stability factors");
  real endtime = ode.endtime();

  PythonFile file("stability_factors.py");

  uint n = ode.size();
  boost::scoped_array<real> s(new real[n*n]);
  real_identity(n, s.get());

  boost::scoped_array<real> A(new real[n*n]);
  boost::scoped_array<real> B(new real[n*n]);
  Array<real> tmp(n);

  // How should the length of the timestep be decided?
  for (ODESolution::iterator it = u.begin(); it != u.end(); it++)
  {
    ODESolutionData& timestep = *it;
    real& t = timestep.a;

    if (t > ode.endtime())
      break;

    //u.eval(t, tmp);
    timestep.eval_a(tmp.data().get());

    get_JT(A.get(), tmp, t);

    real_mult(n*n, A.get(), timestep.k);

    HighPrecision::real_mat_exp(n, B.get(), A.get(), 10);

    HighPrecision::real_mat_prod_inplace(n, s.get(), B.get());

    for (uint i = 0; i < n; ++i)
      tmp[i] = real_norm(n, &s[i*n]);

    file << std::pair<real, RealArrayRef>(t, RealArrayRef(tmp));

    p = to_double(t/endtime);
  }

  end();
}
//-----------------------------------------------------------------------------
void StabilityAnalysis::get_JT(real* JT, const Array<real>& u, real& t)
{
  uint n = u.size();

  // Note that matrices are stored column-oriented in the real functions

  Array<real> e(n);
  // Declare array to wrap the columns of JT.
  Array<real> JT_array(n, JT);
  for (uint i = 0; i < n; ++i)
  {
    // Fill out each column of A
    e.zero();
    e[i] = 1.0;
    JT_array.update(n, &JT[i*n]);
    ode.JT(e, JT_array, u, t);
  }
}
//-----------------------------------------------------------------------------
