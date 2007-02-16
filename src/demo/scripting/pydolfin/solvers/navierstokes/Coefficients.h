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
	  if(p.x() == 0.0 && p.y() > 0.8)
	  {
	    if(i == 0)
	      value.set(1.0);
	    else
	      value.set(0.0);
	  }
	  else if(p.x() == 1.0 && p.y() < 0.2)
	  {
	    if(i == 0)
	      value.set(1.0);
	    else
	      value.set(0.0);
	  }
	  else
	  {
	    value.set(0.0);
	  }
	}
    };

  class PressureBC : public BoundaryCondition
    {
    public:
      PressureBC()
        {
        }
      
      void eval(BoundaryValue& value, const Point& p, unsigned int i)
        {
	  if(p.x() == 1.0 && p.y() < 0.2)
	  {
	    value.set(0.0);
	  }
	}
    };
}
