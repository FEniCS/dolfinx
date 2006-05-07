#include <dolfin.h>

using namespace dolfin;

int main()
{
  DenseMatrix A(3, 3);
  A(0, 0) = A(1, 1) = A(2, 2) = 1.0;
  A.invert();
  
  return 0;
}
