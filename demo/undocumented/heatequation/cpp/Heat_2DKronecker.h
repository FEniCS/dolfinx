#ifndef __HEAT_2D_KRONECKER_H
#define __HEAT_2D_KRONECKER_H

// Include dolfin libraries
#include <dolfin/tensorproduct/TensorProductFunctionSpace.h>

// Include the generated code (includes a lot of other stuff)
#include "Heat_2D.h"

namespace Heat_2DKronecker
{

class KroneckerForm_0 : public dolfin::TensorProductForm
{
  public:

    // Constructor
    KroneckerForm_0(boost::shared_ptr<const dolfin::TensorProductFunctionSpace> V1, boost::shared_ptr<const dolfin::TensorProductFunctionSpace> V0)
     : dolfin::TensorProductForm(2, 1, 2, 3)
    {
      // Store function spaces
      _function_spaces[0] = V0;
      _function_spaces[1] = V1;

      // Create forms in flattened structure (a0, a1, b0, b1, c0, c1) etc.
      _forms[0] = boost::shared_ptr<dolfin::Form>(new Heat_2D::Form_0(V1->extract_factor_space(0),V0->extract_factor_space(0)));
      _forms[2] = boost::shared_ptr<dolfin::Form>(new Heat_2D::Form_2(V1->extract_factor_space(0),V0->extract_factor_space(0)));
      _forms[4] = boost::shared_ptr<dolfin::Form>(new Heat_2D::Form_4(V1->extract_factor_space(0),V0->extract_factor_space(0)));
      _forms[1] = boost::shared_ptr<dolfin::Form>(new Heat_2D::Form_1(V1->extract_factor_space(1),V0->extract_factor_space(1)));
      _forms[3] = boost::shared_ptr<dolfin::Form>(new Heat_2D::Form_3(V1->extract_factor_space(1),V0->extract_factor_space(1)));
      _forms[5] = boost::shared_ptr<dolfin::Form>(new Heat_2D::Form_5(V1->extract_factor_space(1),V0->extract_factor_space(1)));

    }

    // Destructor
    ~KroneckerForm_0()
    {}

};

}
#endif
