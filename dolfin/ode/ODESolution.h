// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-06-11
// Last changed: 2008-06-11


#ifndef __ODESOLUTION_H
#define __ODESOLUTION_H

#include <iostream>
#include <fstream>
#include <dolfin/common/types.h>
#include <dolfin/la/uBLASVector.h>

namespace dolfin
{
  //forward declarations
  class ODE;
  class Sample;

  class ODESolution
  {
  public :
    /// Create solution data for given ODE
    ODESolution(ODE& ode);
    ~ODESolution();

    /// Evaluate (interpolate) value og solution at given time    
    void eval(const real t, uBLASVector& y);
    
    /// for testing
    void printVector(const uBLASVector& u);
    
  private :
    
    ODE& ode;
    char *filename;
    std::fstream file;
    std::pair<real, uBLASVector> *cache;
    uint count;
    
    uint cachesize;
    uint ringbufcounter;
    
    //binary tree with mapping from t value to index in file
    std::vector<real> bintree;
    
    uint step;
    
    void interpolate(const uBLASVector& v1,
                     const real t1, 
                     const uBLASVector& v2, 
                     const real t2, 
                     const real t, 
                     uBLASVector& result);
    
    void addSample(Sample& sample);
    
    void makeIndex(); //make this object ready for reading
    
    friend class TimeStepper;
    friend class ODESolver;
  };

}

#endif
