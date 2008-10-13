// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg, 2008.
// 
// First added:  2008-06-11
// Last changed: 2008-10-06

#ifndef __ODESOLUTION_H
#define __ODESOLUTION_H

#include <iostream>
#include <fstream>
#include <vector>
#include <dolfin/common/types.h>
#include <dolfin/common/real.h>

#define ODESOLUTION_INITIAL_ALLOC 1000000
#define ODESOLUTION_MAX_ALLOC     16000000

namespace dolfin
{

  // Forward declarations
  class ODE;
  class Sample;

  /// ODESolution stores the samples from the ODE solver, primarily to
  /// be able to solve the dual problem. To be able to evaluate the
  /// solution in an arbitrary point, ODESolution makes a simple linear
  /// interpolation between the the closest samples. A number of
  /// interpolated values is cached, since the ODE solver repeatedly
  /// requests evaluation of the same t.
  /// 
  /// The samples are stored in memory if possible, otherwise stored
  /// in a temporary file and fetched from disk in blocks when needed.

  class ODESolution
  {
  public:

    /// Create solution data for given ODE
    ODESolution(ODE& ode);

    /// Destructor
    ~ODESolution();

    /// Evaluate (interpolate) value of solution at given time    
    void eval(const real t, real* y);

    // Add sample
    void add_sample(Sample& sample);

    // Flush values (make values available for interpolation)
    void flush();
    
  private:

    ODE& ode;
    std::string filename;
    std::fstream file;
    std::pair<real, real*>* cache;
    
    uint cache_size;
    uint ringbufcounter;
    
    // Sorted vector with pair. Each entry representents a mapping from t value to index in file/buffer
    std::vector<real> bintree;
    uint step;
    
    real* buffer;
    real* tmp;
    uint buffer_size;
    uint buffer_offset;
    uint buffer_count; //actual number of entries in buffer
    bool dataondisk;

    // Linear interpolation
    void interpolate(const real* v0, const real t0, 
                     const real* v1, const real t1,
                     const real t, real* result);

    // Read entry from buffer, fetch from disk if necessary
    void get_from_buffer(real* u, uint index);
    
  };

}

#endif
