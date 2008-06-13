#ifndef __ODESOLUTION_H
#define __ODESOLUTION_H

#include <iostream>
#include <fstream>
#include <dolfin/common/types.h>
#include <dolfin/la/uBlasVector.h>

namespace dolfin
{
  //forward declarations
  class ODE;
  class Sample;

  class ODESolution
  {
  public :
    
    ODESolution(ODE& ode);
    ~ODESolution();
    
    void eval(const real t, uBlasVector& y);
    
    //for testing
    void printVector(const uBlasVector& u);
    
  private :
    
    ODE& ode;
    char *filename;
    std::fstream file;
    std::pair<real, uBlasVector> *cache;
    uint count;
    
    uint cachesize;
    uint ringbufcounter;
    
    //binary tree with mapping from t value to index in file
    std::vector<real> bintree;
    
    uint step;
    
    void interpolate(const uBlasVector& v1,
                     const real t1, 
                     const uBlasVector& v2, 
                     const real t2, 
                     const real t, 
                     uBlasVector& result);
    
    void addSample(Sample& sample);
    
    void makeIndex(); //make this object ready for reading
    
    friend class TimeStepper;
    friend class ODESolver;
  };

}

#endif
