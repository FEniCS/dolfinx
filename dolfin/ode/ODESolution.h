// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-06-11
// Last changed: 2008-10-02


#ifndef __ODESOLUTION_H
#define __ODESOLUTION_H

#include <iostream>
#include <fstream>
#include <vector>
#include <dolfin/common/types.h>

#define ODESOLUTION_INITIAL_ALLOC 1000000
#define ODESOLUTION_MAX_ALLOC     16000000

namespace dolfin
{
  //forward declarations
  class ODE;
  class Sample;
  class uBLASVector;

  /// ODESolution stores the samples from the ODE solver, primarily to be able
  /// to solve the dual problem. To be ble to evaluate the solution in an
  /// arbitrary point ODESolution makes a simple linear interpolation between the
  /// the closest samples. A number of interpolated values is cached, since 
  /// the ODE solver repeatedly requests evaluation of the same t.
  /// 
  /// The samples is stored in memory if possible, otherwise stored in a
  /// temporary file and fetched from disk when needed.

  class ODESolution
  {
  public :
    /// Create solution data for given ODE
    ODESolution(ODE& ode);
    ~ODESolution();

    /// Evaluate (interpolate) value of solution at given time    
    void eval(const real t, uBLASVector& y);
    
  private :    
    ODE& ode;
    std::string filename;
    std::fstream file;
    std::pair<real, uBLASVector> *cache;
    
    uint cachesize;
    uint ringbufcounter;
    
    //sorted vector with pair. Each entry representents a
    // mapping from t value to index in file/buffer
    std::vector<real> bintree;
    uint step;
    
    real* buffer;
    uint buffersize;
    uint bufferoffset;
    uint buffercount; //actual number of entries in buffer
    bool dataondisk;

    void lerp(const uBLASVector& v1,
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
