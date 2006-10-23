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
      if(time() > 2.0 && time() < 2.2)
      {
	if(i == 0)
	  return -200.0;
	if(i == 1)
	  return -10.0;
      }
      else
      {
	return 0.0;
      }
    }
  };

}
