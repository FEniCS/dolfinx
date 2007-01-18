#ifndef __ALE_FUNCTION_H
#define __ALE_FUNCTION_H


#include <dolfin/Function.h>


namespace dolfin
{

  /// Allows the user to define a function using both reference
  /// coordinates as well as spacial coordinates.
  
  class ALEFunction : public Function
  {
      
  public :
    
    /// eval allows one to use the reference points
    virtual real eval(const Point& p, const Point& r, unsigned int i) = 0;
    
    /// Original eval from Function class
    virtual real eval(const Point& p, unsigned int i) {
      
      return eval(p,p,i);
    }
  };
  
  
}  
  

#endif
