// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2004-04-04
// Last changed: 2005-12-19

/*

// FIXME: BROKEN

#include <cmath>
#include <dolfin/parameter/parameters.h>
#include "ReducedModel.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ReducedModel::ReducedModel(ODE& ode) 
  : ODE(ode.size(), ode.endtime()), ode(ode), g(ode.size()), reduced(false)
{
  warning("Automatic modeling is EXPERIMENTAL.");

  tau     = get("ODE average length");
  samples = get("ODE average samples");
  tol     = get("ODE average tolerance");

  // Copy the sparsity
  //sparsity = ode.sparsity;

  // Adjust the maximum allowed time step to the initial time step
  double kmax = get("ODE initial time step");
  set("ODE maximum time step", kmax);
}
//-----------------------------------------------------------------------------
ReducedModel::~ReducedModel()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
double ReducedModel::f(const Vector& u, double t, unsigned int i)
{
  if ( g[i].active() )
    return ode.f(u, t, i) + g[i]();
  else
    return 0.0;

  return 0.0;
}
//-----------------------------------------------------------------------------
double ReducedModel::u0(unsigned int i)
{
  return ode.u0(i);
}
//-----------------------------------------------------------------------------
void ReducedModel::update(RHS& f, Function& u, double t)
{
  // Update for the given model if used
  ode.update(f, u, t);

  // Create the reduced model only one time
  if ( reduced )
    return;

  // Check if we have reached the beyond twice the average length
  if ( t < (2.0*tau) )
    return;

  // Write a message
  message("Creating reduced model at t = %.1e.", 2*tau);

  // Compute averages of u and f
  computeAverages(f, u, fbar, ubar);

  // Compute reduced model
  for (unsigned int i = 0; i < ode.size(); i++)
    g[i].computeModel(ubar, fbar, i, tau, ode);
}
//-----------------------------------------------------------------------------
void ReducedModel::update(Solution& u, Adaptivity& adaptivity, double t)
{
  // Update for the given model if used
  ode.update(u, adaptivity, t);

  // Check if we have reached the beyond twice the average length
  if ( t < (2.0*tau) )
    return;

  // Create the reduced model only one time
  if ( reduced )
    return;  

  // FIXME: Choose better maximum time step
  // Adjust (increase) maximum allowed time step
  adaptivity.adjustMaximumTimeStep(0.1);

  // Adjust end values for inactive components
  for (unsigned int i = 0; i < ode.size(); i++)
    if ( !g[i].active() )
      u.setlast(i, ubar(i));

  // Clear averages
  ubar.clear();
  fbar.clear();

  // Remember that we have created the model
  reduced = true;
}
//-----------------------------------------------------------------------------
void ReducedModel::save(Sample& sample)
{
  ode.save(sample);
}
//-----------------------------------------------------------------------------
void ReducedModel::computeAverages(RHS& f, Function& u,
				   Vector& fbar, Vector& ubar)
{
error("This function needs to be updated to the new format.");

  // Sample length
  double k = tau / static_cast<double>(samples);

  // Weight for each sample
  double w = 1.0 / static_cast<double>(samples);

  // Initialize averages
  ubar.resize(ode.size()); // Average on first half (and then both...)
  fbar.resize(ode.size()); // Average on both intervals
  ubar = 0.0;
  fbar = 0.0;

  // Minimum and maximum values
  Vector umin(ode.size());
  Vector umax(ode.size());
  umin = 0.0;
  umax = 0.0;

  // Additional average on second interval for u
  Vector ubar2(ode.size());
  ubar2 = 0.0;

  Progress p("Computing averages", 2*samples);

  // Compute average values on first half of the averaging interval
  for (unsigned int j = 0; j < samples; j++)
  {
    for (unsigned int i = 0; i < ode.size(); i++)
    {
      // Sample time
      double t = 0.5*k + static_cast<double>(j)*k;
      
      // Compute u and f
      double uu = u(i, t);
      double ff = f(i, t);

      // Compute average values of u and f
      ubar(i) += w * uu;
      fbar(i) += 0.5 * w * ff;
      
      // Compute minimum and maximum values of u and f
      if ( j == 0 )
      {
	umin(i) = uu;
	umax(i) = uu;
      }
      else
      {
	umin(i) = std::min(umin(i), uu);
	umax(i) = std::max(umax(i), uu);
      }
    }
    
    ++p;
  }
    
  // Compute average values on second half of the averaging interval
  for (unsigned int j = 0; j < samples; j++)
  {
    for (unsigned int i = 0; i < ode.size(); i++)
    {
      // Sample time
      double t = tau + 0.5*k + static_cast<double>(j)*k;
      
      // Compute u and f
      double uu = u(i, t);
      double ff = f(i, t);
      
      // Compute average values of u and f
      ubar2(i) += w * uu;
      fbar(i)  += 0.5 * w * ff;
      
      // Compute minimum and maximum values of u and f
      umin(i) = std::min(umin(i), uu);
      umax(i) = std::max(umax(i), uu);
    }

    ++p;
  }

  // Compute relative changes in u
  for (unsigned int i = 0; i < ode.size(); i++)
  {
    double du = fabs(ubar(i) - ubar2(i)) / (umax(i) - umin(i) + DOLFIN_EPS);

    cout << "Component " << i << ":" << endl;
    cout << "  ubar1 = " << ubar(i) << endl;
    cout << "  ubar2 = " << ubar2(i) << endl;
    cout << "  fbar  = " << fbar(i) << endl;
    cout << "  du/u  = " << du << endl;

    // Inactivate components whose average is constant
    if ( du < tol )
    {
      cout << "Inactivating component " << i << endl;
      g[i].inactivate();
    }

    // Save average
    ubar(i) = 0.5*(ubar(i) + ubar2(i));
  }

}
//-----------------------------------------------------------------------------
// ReducedModel::Model
//-----------------------------------------------------------------------------
ReducedModel::Model::Model() : g(0), _active(true)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
ReducedModel::Model::~Model()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
double ReducedModel::Model::operator() () const
{
  return g;
}
//-----------------------------------------------------------------------------
bool ReducedModel::Model::active() const
{
  return _active;
}
//-----------------------------------------------------------------------------
void ReducedModel::Model::inactivate()
{
  _active = false;
}
//-----------------------------------------------------------------------------
void ReducedModel::Model::computeModel(Vector& ubar, Vector& fbar,
				       unsigned int i, double tau, ODE& ode)
{
  // FIXME: BROKEN
  //g = fbar(i) - ode.f(ubar, tau, i);

  cout << "Modeling term for component " << i << ": " << g << endl;
}
//-----------------------------------------------------------------------------
*/
