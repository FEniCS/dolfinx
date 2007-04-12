// Automatically generated code mapping element and dof map signatures
// to the corresponding ufc::finite_element and ufc::dof_map classes

#include <cstring>

#include "Lagrange_triangle_1.h"
#include "Lagrange_triangle_2.h"
#include "Lagrange_triangle_3.h"
#include "Lagrange_tetrahedron_1.h"
#include "Lagrange_tetrahedron_2.h"
#include "Lagrange_tetrahedron_3.h"
#include "Discontinuous_Lagrange_triangle_0.h"
#include "Discontinuous_Lagrange_triangle_1.h"
#include "Discontinuous_Lagrange_triangle_2.h"
#include "Discontinuous_Lagrange_triangle_3.h"
#include "Discontinuous_Lagrange_tetrahedron_0.h"
#include "Discontinuous_Lagrange_tetrahedron_1.h"
#include "Discontinuous_Lagrange_tetrahedron_2.h"
#include "Discontinuous_Lagrange_tetrahedron_3.h"

#include <dolfin/ElementLibrary.h>

ufc::finite_element* dolfin::ElementLibrary::create_finite_element(const char* signature)
{
  if (strcmp("signature", "Lagrange finite element of degree 1 on a triangle") == 0)
    return new Lagrange_triangle_1_finite_element_0();
  if (strcmp("signature", "Lagrange finite element of degree 2 on a triangle") == 0)
    return new Lagrange_triangle_2_finite_element_0();
  if (strcmp("signature", "Lagrange finite element of degree 3 on a triangle") == 0)
    return new Lagrange_triangle_3_finite_element_0();
  if (strcmp("signature", "Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new Lagrange_tetrahedron_1_finite_element_0();
  if (strcmp("signature", "Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new Lagrange_tetrahedron_2_finite_element_0();
  if (strcmp("signature", "Lagrange finite element of degree 3 on a tetrahedron") == 0)
    return new Lagrange_tetrahedron_3_finite_element_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 0 on a triangle") == 0)
    return new Discontinuous_Lagrange_triangle_0_finite_element_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 1 on a triangle") == 0)
    return new Discontinuous_Lagrange_triangle_1_finite_element_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 2 on a triangle") == 0)
    return new Discontinuous_Lagrange_triangle_2_finite_element_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 3 on a triangle") == 0)
    return new Discontinuous_Lagrange_triangle_3_finite_element_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 0 on a tetrahedron") == 0)
    return new Discontinuous_Lagrange_tetrahedron_0_finite_element_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new Discontinuous_Lagrange_tetrahedron_1_finite_element_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new Discontinuous_Lagrange_tetrahedron_2_finite_element_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 3 on a tetrahedron") == 0)
    return new Discontinuous_Lagrange_tetrahedron_3_finite_element_0();
  return 0;
}

ufc::dof_map* dolfin::ElementLibrary::create_dof_map(const char* signature)
{
  if (strcmp("signature", "Lagrange finite element of degree 1 on a triangle") == 0)
    return new Lagrange_triangle_1_dof_map_0();
  if (strcmp("signature", "Lagrange finite element of degree 2 on a triangle") == 0)
    return new Lagrange_triangle_2_dof_map_0();
  if (strcmp("signature", "Lagrange finite element of degree 3 on a triangle") == 0)
    return new Lagrange_triangle_3_dof_map_0();
  if (strcmp("signature", "Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new Lagrange_tetrahedron_1_dof_map_0();
  if (strcmp("signature", "Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new Lagrange_tetrahedron_2_dof_map_0();
  if (strcmp("signature", "Lagrange finite element of degree 3 on a tetrahedron") == 0)
    return new Lagrange_tetrahedron_3_dof_map_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 0 on a triangle") == 0)
    return new Discontinuous_Lagrange_triangle_0_dof_map_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 1 on a triangle") == 0)
    return new Discontinuous_Lagrange_triangle_1_dof_map_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 2 on a triangle") == 0)
    return new Discontinuous_Lagrange_triangle_2_dof_map_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 3 on a triangle") == 0)
    return new Discontinuous_Lagrange_triangle_3_dof_map_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 0 on a tetrahedron") == 0)
    return new Discontinuous_Lagrange_tetrahedron_0_dof_map_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new Discontinuous_Lagrange_tetrahedron_1_dof_map_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new Discontinuous_Lagrange_tetrahedron_2_dof_map_0();
  if (strcmp("signature", "Discontinuous Lagrange finite element of degree 3 on a tetrahedron") == 0)
    return new Discontinuous_Lagrange_tetrahedron_3_dof_map_0();
  return 0;
}
