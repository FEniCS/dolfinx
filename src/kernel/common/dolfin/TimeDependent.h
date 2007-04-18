// Copyright (C) 2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-09-02
// Last changed: 2005

#ifndef __TIME_DEPENDENT_H
#define __TIME_DEPENDENT_H 
 

namespace dolfin
{
  
  /// Associates an object with time t 

  class TimeDependent
  {
  public:
    
    /// Constructors
    TimeDependent();
    TimeDependent(const real& t);
    
    /// Destructor
    ~TimeDependent();

    /// Associate object with time t
    void sync(const real& t);
    
    /// Return the current time t
    real time() const;

  private:
    
    // Pointer to the current time
    const real* t;

  };
  
}

#endif
