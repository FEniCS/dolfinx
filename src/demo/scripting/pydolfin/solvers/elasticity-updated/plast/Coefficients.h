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
      if(time() > 1.0 && time() < 1.2)
      {
	if(i == 0 && p.y > 1.0 && p.y <= 2.0)
	  return -800.0;
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
      if(p.x == 0.0)
	value = 0.0;
    }
  };


}
