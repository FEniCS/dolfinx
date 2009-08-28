// Copyright (C) 2009 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-07-20
// Last changed: 2009-07-20

#include "StabilityAnalysis.h"
#include "ODESolution.h"
#include "ODE.h"
#include <dolfin/log/Logger.h>
#include <dolfin/log/Progress.h>
#include <dolfin/io/PythonFile.h>
#include <dolfin/common/real.h>
#include <vector>

using namespace dolfin;


StabilityAnalysis::StabilityAnalysis(ODE& ode, ODESolution& u) : 
  ode(ode), u(u), write_to_file(ode.parameters("save_solution")), n(ode.size()) {}
//-----------------------------------------------------------------------------
StabilityAnalysis::~StabilityAnalysis(){}
//-----------------------------------------------------------------------------
void StabilityAnalysis::analyze_integral(uint q)
{
  begin("Computing stability factors");

  real progress_end = ode.endtime()*ode.endtime() + ode.endtime();


  // Collect 
  std::vector< std::pair<real, real*> > s;

  real tmp[n];
  real A[n*n];
  real B[n*n];

  uint count = 0;
  
  PythonFile file("stability_factors.py");

  Progress p("Computing stability factors");

  // How should the length of the timestep be decided?
  for (  ODESolution::iterator it = u.begin(); it != u.end(); it++ )
  {
    ODESolutionData& timestep = *it;
    real& t = timestep.a;

    real* C = new real[n*n];

    u.eval(t, tmp);

    get_JT(A, tmp, t);

    real_mat_pow(n, C, A, q);
    
    // Multiply A with length of timestep
    // A = k*JT(U)
    real_mult(n*n, A, timestep.k);

    // B = e^(k*JT(U))
    real_mat_exp(n, B, A, 10);

    // multiply each matrix in s with B from right
    for (std::vector< std::pair<real, real*> >::iterator s_iterator = s.begin(); 
	 s_iterator != s.end(); ++s_iterator)
    {
      real_mat_prod_inplace(n, (*s_iterator).second, B);
    }

    real_mat_prod_inplace(n, C, B);

    s.push_back( std::pair<real, real*> (t+timestep.k, C) );

    // Now compute the stability factor for T=t
    real sample[n];
    real_zero(n, sample);

    real prev = 0.0;
    
    for (std::vector< std::pair<real, real*> >::iterator s_iterator = s.begin(); 
	 s_iterator != s.end(); ++s_iterator)
    {
      real t  = s_iterator->first;
      real* Z = s_iterator->second;

      // Since the initial data is the unity vectors, we don't have to multiply.
      // We can just pick outthe columns of Z

      for (uint i=0; i<n; ++i) 
      {
	sample[i] += real_norm(n, &Z[n*i]) * real_abs(t-prev);
      }
      prev = t;
    }

    file << std::tr1::tuple<uint, real, real*>(n, t, sample);

    // update progress
    p = to_double( (t*t+t)/progress_end );
    count++;
  }

  end();
}
//-----------------------------------------------------------------------------
// Compute z(0) (the endtime of the dual) as function of (primal) endtime T
void StabilityAnalysis::analyze_endpoint() {
  begin("Computing stability factor");

  Progress p("Computing stability factors");
  real endtime = ode.endtime();

  PythonFile file("stability_factors.py");

  uint n = ode.size();
  real s[n*n];
  real_identity(n,s);

  real A[n*n];
  real B[n*n];
  real tmp[n];


  // How should the length of the timestep be decided?
  for (  ODESolution::iterator it = u.begin(); it != u.end(); it++ )
  {
    ODESolutionData& timestep = *it;
    real& t = timestep.a;

    u.eval(t, tmp);

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
  real e[n];

  for (uint i=0; i<n; ++i) 
  {
    // fill out each column of A
    real_zero(n, e);
    e[i] = 1.0;
    ode.JT(e, &JT[i*n], u, t);
  }
}
