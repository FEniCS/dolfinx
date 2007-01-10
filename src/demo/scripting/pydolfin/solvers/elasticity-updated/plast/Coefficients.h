#include <iostream>
#include <dolfin.h>

namespace dolfin
{

  class MySource : public Function
  {
  public:
    MySource()
    {
    }

    real eval(const Point& p, unsigned int i)
    {
      if(time() > 0.1 && time() < 0.3)
      {
	if(i == 0 && p[1] > 1.0 && p[1] <= 2.0)
        {
	  return -200.0;
        }
        else
        {
          return 0.0;
        }
      }
      else
      {
	return 0.0;
      }
    }
  };

  class MyBC : public BoundaryCondition
  {
  public:
    MyBC()
    {
    }
    
    void eval(BoundaryValue& value, const Point& p, unsigned int i)
    {
      if(p[0] == 0.0)
	value = 0.0;
    }
  };


}
