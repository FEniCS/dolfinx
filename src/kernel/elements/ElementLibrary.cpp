// Automatically generated code mapping element and dof map signatures
// to the corresponding ufc::finite_element and ufc::dof_map classes

#include <cstring>

#include "ffc_00.h"
#include "ffc_01.h"
#include "ffc_02.h"
#include "ffc_03.h"
#include "ffc_04.h"
#include "ffc_05.h"
#include "ffc_06.h"
#include "ffc_07.h"
#include "ffc_08.h"
#include "ffc_09.h"
#include "ffc_10.h"
#include "ffc_11.h"
#include "ffc_12.h"
#include "ffc_13.h"
#include "ffc_14.h"
#include "ffc_15.h"
#include "ffc_16.h"
#include "ffc_17.h"
#include "ffc_18.h"
#include "ffc_19.h"
#include "ffc_20.h"
#include "ffc_21.h"
#include "ffc_22.h"
#include "ffc_23.h"
#include "ffc_24.h"
#include "ffc_25.h"
#include "ffc_26.h"

#include <dolfin/ElementLibrary.h>

ufc::finite_element* dolfin::ElementLibrary::create_finite_element(const char* signature)
{
  if (strcmp(signature, "Lagrange finite element of degree 1 on a triangle") == 0)
    return new ffc_00_finite_element_0();
  if (strcmp(signature, "Lagrange finite element of degree 2 on a triangle") == 0)
    return new ffc_01_finite_element_0();
  if (strcmp(signature, "Lagrange finite element of degree 3 on a triangle") == 0)
    return new ffc_02_finite_element_0();
  if (strcmp(signature, "Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new ffc_03_finite_element_0();
  if (strcmp(signature, "Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new ffc_04_finite_element_0();
  if (strcmp(signature, "Lagrange finite element of degree 3 on a tetrahedron") == 0)
    return new ffc_05_finite_element_0();
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 0 on a triangle") == 0)
    return new ffc_06_finite_element_0();
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 1 on a triangle") == 0)
    return new ffc_07_finite_element_0();
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 2 on a triangle") == 0)
    return new ffc_08_finite_element_0();
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 0 on a tetrahedron") == 0)
    return new ffc_09_finite_element_0();
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new ffc_10_finite_element_0();
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new ffc_11_finite_element_0();
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 3 on a tetrahedron") == 0)
    return new ffc_12_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 1 on a triangle, Lagrange finite element of degree 1 on a triangle]") == 0)
    return new ffc_13_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 2 on a triangle, Lagrange finite element of degree 2 on a triangle]") == 0)
    return new ffc_14_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 3 on a triangle, Lagrange finite element of degree 3 on a triangle]") == 0)
    return new ffc_15_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 1 on a tetrahedron, Lagrange finite element of degree 1 on a tetrahedron, Lagrange finite element of degree 1 on a tetrahedron]") == 0)
    return new ffc_16_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 2 on a tetrahedron, Lagrange finite element of degree 2 on a tetrahedron, Lagrange finite element of degree 2 on a tetrahedron]") == 0)
    return new ffc_17_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 3 on a tetrahedron, Lagrange finite element of degree 3 on a tetrahedron, Lagrange finite element of degree 3 on a tetrahedron]") == 0)
    return new ffc_18_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 0 on a triangle, Discontinuous Lagrange finite element of degree 0 on a triangle]") == 0)
    return new ffc_19_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 1 on a triangle, Discontinuous Lagrange finite element of degree 1 on a triangle]") == 0)
    return new ffc_20_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 2 on a triangle, Discontinuous Lagrange finite element of degree 2 on a triangle]") == 0)
    return new ffc_21_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 0 on a tetrahedron, Discontinuous Lagrange finite element of degree 0 on a tetrahedron, Discontinuous Lagrange finite element of degree 0 on a tetrahedron]") == 0)
    return new ffc_22_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 1 on a tetrahedron, Discontinuous Lagrange finite element of degree 1 on a tetrahedron, Discontinuous Lagrange finite element of degree 1 on a tetrahedron]") == 0)
    return new ffc_23_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 2 on a tetrahedron, Discontinuous Lagrange finite element of degree 2 on a tetrahedron, Discontinuous Lagrange finite element of degree 2 on a tetrahedron]") == 0)
    return new ffc_24_finite_element_0();
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 3 on a tetrahedron, Discontinuous Lagrange finite element of degree 3 on a tetrahedron, Discontinuous Lagrange finite element of degree 3 on a tetrahedron]") == 0)
    return new ffc_25_finite_element_0();
  if (strcmp(signature, "Brezzi-Douglas-Marini finite element of degree 1 on a triangle") == 0)
    return new ffc_26_finite_element_0();
  return 0;
}

ufc::dof_map* dolfin::ElementLibrary::create_dof_map(const char* signature)
{
  if (strcmp(signature, "FFC dof map for Lagrange finite element of degree 1 on a triangle") == 0)
    return new ffc_00_dof_map_0();
  if (strcmp(signature, "FFC dof map for Lagrange finite element of degree 2 on a triangle") == 0)
    return new ffc_01_dof_map_0();
  if (strcmp(signature, "FFC dof map for Lagrange finite element of degree 3 on a triangle") == 0)
    return new ffc_02_dof_map_0();
  if (strcmp(signature, "FFC dof map for Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new ffc_03_dof_map_0();
  if (strcmp(signature, "FFC dof map for Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new ffc_04_dof_map_0();
  if (strcmp(signature, "FFC dof map for Lagrange finite element of degree 3 on a tetrahedron") == 0)
    return new ffc_05_dof_map_0();
  if (strcmp(signature, "FFC dof map for Discontinuous Lagrange finite element of degree 0 on a triangle") == 0)
    return new ffc_06_dof_map_0();
  if (strcmp(signature, "FFC dof map for Discontinuous Lagrange finite element of degree 1 on a triangle") == 0)
    return new ffc_07_dof_map_0();
  if (strcmp(signature, "FFC dof map for Discontinuous Lagrange finite element of degree 2 on a triangle") == 0)
    return new ffc_08_dof_map_0();
  if (strcmp(signature, "FFC dof map for Discontinuous Lagrange finite element of degree 0 on a tetrahedron") == 0)
    return new ffc_09_dof_map_0();
  if (strcmp(signature, "FFC dof map for Discontinuous Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new ffc_10_dof_map_0();
  if (strcmp(signature, "FFC dof map for Discontinuous Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new ffc_11_dof_map_0();
  if (strcmp(signature, "FFC dof map for Discontinuous Lagrange finite element of degree 3 on a tetrahedron") == 0)
    return new ffc_12_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Lagrange finite element of degree 1 on a triangle, Lagrange finite element of degree 1 on a triangle]") == 0)
    return new ffc_13_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Lagrange finite element of degree 2 on a triangle, Lagrange finite element of degree 2 on a triangle]") == 0)
    return new ffc_14_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Lagrange finite element of degree 3 on a triangle, Lagrange finite element of degree 3 on a triangle]") == 0)
    return new ffc_15_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Lagrange finite element of degree 1 on a tetrahedron, Lagrange finite element of degree 1 on a tetrahedron, Lagrange finite element of degree 1 on a tetrahedron]") == 0)
    return new ffc_16_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Lagrange finite element of degree 2 on a tetrahedron, Lagrange finite element of degree 2 on a tetrahedron, Lagrange finite element of degree 2 on a tetrahedron]") == 0)
    return new ffc_17_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Lagrange finite element of degree 3 on a tetrahedron, Lagrange finite element of degree 3 on a tetrahedron, Lagrange finite element of degree 3 on a tetrahedron]") == 0)
    return new ffc_18_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Discontinuous Lagrange finite element of degree 0 on a triangle, Discontinuous Lagrange finite element of degree 0 on a triangle]") == 0)
    return new ffc_19_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Discontinuous Lagrange finite element of degree 1 on a triangle, Discontinuous Lagrange finite element of degree 1 on a triangle]") == 0)
    return new ffc_20_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Discontinuous Lagrange finite element of degree 2 on a triangle, Discontinuous Lagrange finite element of degree 2 on a triangle]") == 0)
    return new ffc_21_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Discontinuous Lagrange finite element of degree 0 on a tetrahedron, Discontinuous Lagrange finite element of degree 0 on a tetrahedron, Discontinuous Lagrange finite element of degree 0 on a tetrahedron]") == 0)
    return new ffc_22_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Discontinuous Lagrange finite element of degree 1 on a tetrahedron, Discontinuous Lagrange finite element of degree 1 on a tetrahedron, Discontinuous Lagrange finite element of degree 1 on a tetrahedron]") == 0)
    return new ffc_23_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Discontinuous Lagrange finite element of degree 2 on a tetrahedron, Discontinuous Lagrange finite element of degree 2 on a tetrahedron, Discontinuous Lagrange finite element of degree 2 on a tetrahedron]") == 0)
    return new ffc_24_dof_map_0();
  if (strcmp(signature, "FFC dof map for Mixed finite element: [Discontinuous Lagrange finite element of degree 3 on a tetrahedron, Discontinuous Lagrange finite element of degree 3 on a tetrahedron, Discontinuous Lagrange finite element of degree 3 on a tetrahedron]") == 0)
    return new ffc_25_dof_map_0();
  if (strcmp(signature, "FFC dof map for Brezzi-Douglas-Marini finite element of degree 1 on a triangle") == 0)
    return new ffc_26_dof_map_0();
  return 0;
}
