// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/ReducedModel.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ReducedModel::ReducedModel(ODE& ode) 
  : ODE(ode.size()), ode(ode), g(ode.size()), reduced(false)
{
  dolfin_warning("Automatic modeling is EXPERIMENTAL.");

  T = ode.endtime();
  
  tau     = dolfin_get("average length");
  samples = dolfin_get("average samples");
  tol     = dolfin_get("tolerance");

  // Adjust the maximum allowed time step to the initial time step
  real kmax = dolfin_get("initial time step");
  dolfin_set("maximum time step", kmax);
}
//-----------------------------------------------------------------------------
ReducedModel::~ReducedModel()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real ReducedModel::f(const Vector& u, real t, unsigned int i)
{
  if ( g[i].active() )
    return ode.f(u, t, i) + g[i]();
  else
    return 0.0;
}
//-----------------------------------------------------------------------------
real ReducedModel::u0(unsigned int i)
{
  return ode.u0(i);
}
//-----------------------------------------------------------------------------
Element::Type ReducedModel::method(unsigned int i)
{
  return ode.method(i);
}
//-----------------------------------------------------------------------------
unsigned int ReducedModel::order(unsigned int i)
{
  return ode.order(i);
}
//-----------------------------------------------------------------------------
real ReducedModel::timestep(unsigned int i)
{
  return ode.timestep(i);
}
//-----------------------------------------------------------------------------
void ReducedModel::update(RHS& f, Function& u, real t, Adaptivity& adaptivity)
{
  // Update for the given model if used
  ode.update(f, u, t, adaptivity);

  // Create the reduced model only one time
  if ( reduced )
    return;

  // Check if we have reached the beyond twice the average length
  if ( t < (2.0*tau) )
    return;

  // Write a message
  dolfin_info("Creating reduced model at t = %f.", 2*tau);

  // Average values of u and f
  Vector ubar(ode.size());
  Vector fbar(ode.size());

  // Compute average values
  for (unsigned int i = 0; i < ode.size(); i++)
    g[i].computeAverage(f, u, i, tau, samples, tol, ubar, fbar);

  // Compute reduced model
  for (unsigned int i = 0; i < ode.size(); i++)
    g[i].computeModel(ubar, fbar, i, tau, ode);

  // Adjust (increase) maximum allowed time step
  adaptivity.adjustMaximumTimeStep(tau);

  // Remember that we have created the model
  reduced = true;
}
//-----------------------------------------------------------------------------
void ReducedModel::save(Sample& sample)
{
  ode.save(sample);
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
real ReducedModel::Model::operator() () const
{
  return g;
}
//-----------------------------------------------------------------------------
bool ReducedModel::Model::active() const
{
  return _active;
}
//-----------------------------------------------------------------------------
void ReducedModel::Model::computeAverage(RHS& f, Function& u, unsigned int i,
					 real tau, unsigned int samples,
					 real tol, Vector& ubar, Vector& fbar)
{
  // Sample length
  real k = tau / static_cast<real>(samples);

  // Weight for each sample
  real w = 1.0 / static_cast<real>(samples);
  
  // Average values of u and f
  real ubar1 = 0.0;
  real ubar2 = 0.0;
  real fbar1 = 0.0;
  real fbar2 = 0.0;

  // Maximum values of u and f
  real umax = 0.0;
  real fmax = 0.0;

  // Compute average values on first half of the averaging interval
  for (unsigned int j = 0; j < samples; j++)
  {
    // Sample time
    real t = 0.5*k + static_cast<real>(j)*k;
    
    // Compute u and f
    real uu = u(i, t);
    real ff = f(i, t);
    
    // Compute average values of u and f
    ubar1 += w * uu;
    fbar1 += w * ff;
    
    // Compute maximum values of u and f
    umax = std::max(umax, fabs(uu));
    fmax = std::max(fmax, fabs(ff));
  }

  // Compute average values on second half of the averaging interval
  for (unsigned int j = 0; j < samples; j++)
  {
    // Sample time
    real t = tau + 0.5*k + static_cast<real>(j)*k;

    // Compute u and f
    real uu = u(i, t);
    real ff = f(i, t);
    
    // Compute average values of u and f
    ubar2 += w * uu;
    fbar2 += w * ff;
    
    // Compute maximum values of u and f
    umax = std::max(umax, fabs(uu));
    fmax = std::max(fmax, fabs(ff));
  }

  // Compute relative change in u
  real du = fabs(ubar2 - ubar1) / umax;

  // Save averages
  ubar(i) = 0.5*(ubar1 + ubar2);
  fbar(i) = 0.5*(fbar1 + fbar2);

  // Inactivate components whose average is constant
  if ( du < tol )
  {
    cout << "Inactivating component " << i << endl;
    _active = false;
  }
  
  /*
  cout << "Updating model for component " << i << ":" << endl;
  cout << "  ubar1 = " << ubar1 << endl;
  cout << "  ubar2 = " << ubar2 << endl;
  cout << "  fbar1 = " << fbar1 << endl;
  cout << "  fbar2 = " << fbar2 << endl;
  cout << "  umax  = " << umax << endl;
  cout << "  fmax  = " << fmax << endl;
  cout << "  du    = " << du << endl;
  cout << "  df    = " << df << endl;
  */
}
//-----------------------------------------------------------------------------
void ReducedModel::Model::computeModel(Vector& ubar, Vector& fbar,
				       unsigned int i, real tau, ODE& ode)
{
  g = fbar(i) - ode.f(ubar, tau, i);

  cout << "Modeling term for component " << i << ": " << g << endl;
}
//-----------------------------------------------------------------------------
