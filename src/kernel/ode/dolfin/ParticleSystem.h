// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

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
  /// u = (x_0,  y_0,  z_0,  ... , x_n,  y_n,  z_n, 
  ///      x_0', y_0', z_0', ... , x_n', y_n', z_n')

  class ParticleSystem : public ODE
  {
  public:

    /// Constructor
    ParticleSystem(unsigned int n, unsigned int dim = 3);

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

    /// Return initial value for ODE
    real u0(unsigned int i);

    /// Return right-hand side for ODE
    real f(const Vector& u, real t, unsigned int i);

  protected:

    // Return x-component of current position
    virtual real x(unsigned int i);

    // Return y-component of current position
    virtual real y(unsigned int i);

    // Return z-component of current position
    virtual real z(unsigned int i);

    // Return x-component of current velocity
    virtual real vx(unsigned int i);

    // Return y-component of current velocity
    virtual real vy(unsigned int i);

    // Return z-component of current velocity
    virtual real vz(unsigned int i);

    // Number of particles
    unsigned int n;

    // Number of dimensions
    unsigned int dim;

    // Half the number of components in the ODE system
    unsigned int offset;

    // Pointer to solution vector
    const Vector* u;

  };

}

#endif
