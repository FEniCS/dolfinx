// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#ifndef __REDUCED_MODEL_H
#define __REDUCED_MODEL_H

#include <dolfin/constants.h>
#include <dolfin/NewArray.h>
#include <dolfin/ODE.h>

namespace dolfin
{
  class Vector;
  class Function;

  /// ReducedModel represents an averaged ODE of the form
  ///
  ///     u'(t) = f(u(t),t) + g(u(t),t) on (0,T],
  ///         
  ///     u(0)  = u0,
  ///
  /// where u(t) is a vector of length N, and where g
  /// accounts for the effects of small scales on the
  /// averaged solution u.
  ///
  /// A ReducedModel is automatically created whenever the option
  /// "automatic modeling" is set. Note that this does not currently
  /// work in combination with the "step" interface of the
  /// multi-adaptive solver, i.e., the "solve" interface must be used.
  
  class ReducedModel : public ODE
  {
  public:
    
    /// Constructor
    ReducedModel(ODE& ode);
    
    /// Destructor
    ~ReducedModel();

    /// The right-hand side, including model
    real f(const Vector& u, real t, unsigned int i);

    /// Map initial data
    real u0(unsigned int i);

    /// Map the choice of method
    Element::Type method(unsigned int i);

    /// Map the choice of order
    unsigned int order(unsigned int i);

    /// Map the choice of time step
    real timestep(unsigned int i);
    
    /// Map the update function
    void update(RHS& f, Function& u, real t, Adaptivity& adaptivity);

    /// Map the save function
    void save(Sample& sample);

  private:

    // The component-specific model
    class Model
    {
    public:
      
      /// Constructor
      Model();

      /// Destructor
      ~Model();

      /// Evaluate model
      real operator() () const;

      /// Return state
      bool active() const;

      /// Compute average
      void computeAverage(RHS& f, Function& u, unsigned int i,
			  real tau, unsigned int samples, real tol,
			  Vector& ubar, Vector& fbar);

      /// Compute model
      void computeModel(Vector& ubar, Vector& fbar, unsigned int i, 
			real tau, ODE& ode);

    private:

      real g;
      bool _active;

    };
    
    // The given model
    ODE& ode;

    // The reduced model
    NewArray<Model> g;

    // True if model has been reduced
    bool reduced;

    // Length of running average
    real tau;

    // Number of samples for running average
    unsigned int samples;

    // Tolerance for active components
    real tol;
    
  };

}

#endif
