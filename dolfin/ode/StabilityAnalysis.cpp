// Copyright (C) 2009 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
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

  boost::scoped_array<real> tmp_array(new real [n]); real* tmp = tmp_array.get();
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
    boost::scoped_array<real> _C(new real[n*n]);  real* C = _C.get();

    // Get solution values at first nodal point on interval
    timestep.eval_a(tmp);

    // Get transpose of Jacobian
    get_JT(A, tmp, t);

    // Multiply by A^q to differentiate q times: C = JT^q
    real_mat_pow(n, C, A, q);

    // Multiply A with length of timestep: A = k*JT
    real_mult(n*n, A, timestep.k);

    // Compute matrix exponential: B = e^(k*JT)
    real_mat_exp(n, B, A, 10);

    // Multiply each matrix in s with B from right
    for (std::vector< std::pair<real, real*> >::iterator s_iterator = s.begin();
	 s_iterator != s.end(); ++s_iterator)
    {
      real_mat_prod_inplace(n, (*s_iterator).second, B);
    }

    // Differentiate: C = JT^q * e^(k*JT)
    real_mat_prod_inplace(n, C, B);

    // Store differentiated fundamental solution
    s.push_back(std::pair<real, real*>(t + timestep.k, C));

    // Now compute the stability factor for T = t by integrating
    boost::scoped_array<real> sample(new real[n]);
    real_zero(n, sample.get());
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
    file << std::tr1::tuple<uint, real, real*>(n, t, sample.get());

    // Update progress
    double t_double = to_double(t);
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
  boost::scoped_array<real> _s(new real[n*n]); real* s =_s.get();
  real_identity(n,s);

  boost::scoped_array<real> _A(new real[n*n]);  real* A = _A.get();
  boost::scoped_array<real> _B(new real[n*n]);  real* B = _B.get();
  boost::scoped_array<real> _tmp(new real[n]);  real* tmp = _tmp.get();


  // How should the length of the timestep be decided?
  for (  ODESolution::iterator it = u.begin(); it != u.end(); it++ )
  {
    ODESolutionData& timestep = *it;
    real& t = timestep.a;

    if (t > ode.endtime())
      break;

    //u.eval(t, tmp);
    timestep.eval_a(tmp);

    get_JT(A, tmp, t);

    real_mult(n*n, A, timestep.k);

    real_mat_exp(n, B, A, 10);

    real_mat_prod_inplace(n, s, B);

    for (uint i=0; i<n; ++i)
    {
      tmp[i] = real_norm(n, &s[i*n]);
    }

    file << std::tr1::tuple<uint, real, real*>(n, t, tmp);

    p = to_double(t/endtime);

  }

  end();
}
//-----------------------------------------------------------------------------
void StabilityAnalysis::get_JT(real* JT, const real* u, real& t)
{
  // Note that matrices are stored column-oriented in the real functions

  boost::scoped_array<real> _e(new real[n]);  real* e = _e.get();
  for (uint i = 0; i < n; ++i)
  {
    // Fill out each column of A
    real_zero(n, e);
    e[i] = 1.0;
    ode.JT(e, &JT[i*n], u, t);
  }
}
//-----------------------------------------------------------------------------
