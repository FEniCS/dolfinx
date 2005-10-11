// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2004-01-19
// Last changed: 2005

#ifndef __PARTICLE_SYSTEM
#define __PARTICLE_SYSTEM

#include <dolfin/constants.h>
#include <dolfin/ODE.h>

namespace dolfin
{

  /// A ParticleSystem represents a system of particles in three
  /// dimensions, consisting of a given number of particles and
  /// obeying Newton's second law:
  ///
  ///     F = m * a
  ///
  /// where F is the force, m is the mass, and a is the acceleration.
  ///
  /// Positions and velocities are stored in the following order in
  /// the solution vector u(t):
  ///
  /// u = (x_0,  y_0,  z_0,  ... , x_n-1,  y_n-1,  z_n-1, 
  ///      x_0', y_0', z_0', ... , x_n-1', y_n-1', z_n-1').
  ///
  /// FIXME: Need to implement ODE::feval()

  class ParticleSystem : public ODE
  {
  public:

    /// Constructor
    ParticleSystem(unsigned int n, real T, unsigned int dim = 3);

    /// Destructor
    ~ParticleSystem();

    /// Return x-component of initial position for particle i
    virtual real x0(unsigned int i);

    /// Return y-component of initial position for particle i
    virtual real y0(unsigned int i);

    /// Return z-component of initial position for particle i
    virtual real z0(unsigned int i);

    /// Return x-component of initial velocity for particle i
    virtual real vx0(unsigned int i);

    /// Return x-component of initial velocity for particle i
    virtual real vy0(unsigned int i);

    /// Return x-component of initial velocity for particle i
    virtual real vz0(unsigned int i);
    
    /// Return x-component of the force on particle i at time t
    virtual real Fx(unsigned int i, real t);

    /// Return y-component of the force on particle i at time t
    virtual real Fy(unsigned int i, real t);

    /// Return z-component of the force on particle i at time t
    virtual real Fz(unsigned int i, real t);

    /// Return mass of particle i at time t
    virtual real mass(unsigned int i, real t);

    /// Return time step for particle i
    virtual real k(unsigned int i);

    /// Return initial value for ODE
    real u0(unsigned int i);

    /// Return right-hand side for ODE
    real f(const real u[], real t, unsigned int i);

    /// Return time step for ODE
    real timestep(unsigned int i);

  protected:

    // Return x-component of current position (inline optimized)
    real x(unsigned int i) const { return u[dim*i]; }
    // Return y-component of current position (inline optimized)
    real y(unsigned int i) const { return u[dim*i + 1]; }
    // Return z-component of current position (inline optimized)
    real z(unsigned int i) const { return u[dim*i + 2]; }
    // Return x-component of current velocity (inline optimized)
    real vx(unsigned int i) const { return u[offset + dim*i]; }
    // Return y-component of current velocity (inline optimized)
    real vy(unsigned int i) const { return u[offset + dim*i + 1]; }
    // Return z-component of current velocity (inline optimized)
    real vz(unsigned int i) const { return u[offset + dim*i + 2]; }

    // Return distance between to particles
    real dist(unsigned int i, unsigned int j) const;

    // Number of particles
    unsigned int n;

    // Number of dimensions
    unsigned int dim;

    // Half the number of components in the ODE system
    unsigned int offset;

    // Solution vector
    const real* u;

  };

}

#endif
