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
        return 0.0;
    }
  };

  class VelocityBC : public BoundaryCondition
    {
    public:
      VelocityBC()
        {
        }
      
      void eval(BoundaryValue& value, const Point& p, unsigned int i)
        {
          if(p[0] < 0.01 && p[1] > 0.7)
	  {
	    if(i == 0)
	      value = 1.0;
	    else
	      value = 0.0;
	  }
          else if(p[0] > 0.99 && p[1] > 0.7 && i == 0)
	  {
	    if(i == 0)
	      value = 1.0;
	    else
	      value = 0.0;
	  }
	  else
	  {
	    value = 0.0;
	  }
        }
    };
}
