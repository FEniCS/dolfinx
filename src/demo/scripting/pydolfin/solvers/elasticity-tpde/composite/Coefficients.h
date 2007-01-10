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
      if(i == 1)
        return -2.0;
      else
        return 0.0;
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
