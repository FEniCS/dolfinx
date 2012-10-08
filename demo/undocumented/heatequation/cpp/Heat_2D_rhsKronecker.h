#ifndef __HEAT_2D_RHS_KRONECKER_H
#define __HEAT_2D_RHS_KRONECKER_H

// Include dolfin libraries
#include <dolfin/tensorproduct/TensorProductFunctionSpace.h>

// Include the generated code (includes a lot of other stuff)
#include "Heat_2D_rhs.h"

namespace Heat_2D_rhsKronecker
{

class KroneckerForm_0 : public dolfin::TensorProductForm
{
  public:

    // Constructor
    KroneckerForm_0(boost::shared_ptr<const dolfin::TensorProductFunctionSpace> V0)
     : dolfin::TensorProductForm(1, 1, 2, 1)
    {
      // Store function spaces
      _function_spaces[0] = V0;

      // Create forms in flattened structure (a0, a1, b0, b1, c0, c1) etc.
      _forms[0] = boost::shared_ptr<dolfin::Form>(new Heat_2D_rhs::Form_0(V0->extract_factor_space(0)));
      _forms[1] = boost::shared_ptr<dolfin::Form>(new Heat_2D_rhs::Form_1(V0->extract_factor_space(1)));

    }

    // Destructor
    ~KroneckerForm_0()
    {}

};

}
#endif
