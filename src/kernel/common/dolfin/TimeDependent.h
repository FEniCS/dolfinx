// Copyright (C) 2005-2007 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-09-02
// Last changed: 2007-04-27

#ifndef __TIME_DEPENDENT_H
#define __TIME_DEPENDENT_H 
 
#include <dolfin/dolfin_log.h>

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
    real time() const
    {
	    if( !t )
        error("Time has not been associated with object.");		
	    return *t;
    };

  private:
    
    // Pointer to the current time
    const real* t;

  };
  
}

#endif
