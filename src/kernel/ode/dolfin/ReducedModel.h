// Copyright (C) 2004-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2004-04-04
// Last changed: 2005-09-24

#ifndef __REDUCED_MODEL_H
#define __REDUCED_MODEL_H

/*

// FIXME: BROKEN

#include <dolfin/constants.h>
#include <dolfin/Array.h>
#include <dolfin/ODE.h>

namespace dolfin
{

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

  // FIXME: BROKEN
  
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

    /// Map update function
    void update(RHS& f, Function& u, real t);

    /// Map update function
    void update(Solution& u, Adaptivity& adaptivity, real t);

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

      /// Inactivate component
      void inactivate();

      /// Compute model
      void computeModel(Vector& ubar, Vector& fbar, unsigned int i, 
			real tau, ODE& ode);
    private:

      real g;
      bool _active;

    };

    // Compute averages
    void computeAverages(RHS& f, Function& u, Vector& fbar, Vector& ubar);

    // The given model
    ODE& ode;

    // The reduced model
    Array<Model> g;

    // True if model has been reduced
    bool reduced;

    // Length of running average
    real tau;

    // Number of samples for running average
    unsigned int samples;

    // Tolerance for active components
    real tol;
    
    // Averages of u and f
    //Vector ubar;
    //Vector fbar;

  };

}

*/

#endif
