// Automatically generated code mapping element signatures
// to the corresponding Form classes representing projection

#include <cstring>

#include "ffc_L2proj_00.h"
#include "ffc_L2proj_01.h"
#include "ffc_L2proj_02.h"
#include "ffc_L2proj_03.h"
#include "ffc_L2proj_04.h"
#include "ffc_L2proj_05.h"
#include "ffc_L2proj_06.h"
#include "ffc_L2proj_07.h"
#include "ffc_L2proj_08.h"
#include "ffc_L2proj_09.h"
#include "ffc_L2proj_10.h"
#include "ffc_L2proj_11.h"
#include "ffc_L2proj_12.h"
#include "ffc_L2proj_13.h"
#include "ffc_L2proj_14.h"
#include "ffc_L2proj_15.h"
#include "ffc_L2proj_16.h"
#include "ffc_L2proj_17.h"
#include "ffc_L2proj_18.h"
#include "ffc_L2proj_19.h"

#include "ProjectionLibrary.h"

dolfin::Form* dolfin::ProjectionLibrary::create_projection_a(const char* signature)
{
  if (strcmp(signature, "Lagrange finite element of degree 1 on a triangle") == 0)
    return new ffc_L2proj_00BilinearForm;
  if (strcmp(signature, "Lagrange finite element of degree 2 on a triangle") == 0)
    return new ffc_L2proj_01BilinearForm;
  if (strcmp(signature, "Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new ffc_L2proj_02BilinearForm;
  if (strcmp(signature, "Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new ffc_L2proj_03BilinearForm;
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 0 on a triangle") == 0)
    return new ffc_L2proj_04BilinearForm;
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 1 on a triangle") == 0)
    return new ffc_L2proj_05BilinearForm;
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 2 on a triangle") == 0)
    return new ffc_L2proj_06BilinearForm;
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 0 on a tetrahedron") == 0)
    return new ffc_L2proj_07BilinearForm;
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new ffc_L2proj_08BilinearForm;
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new ffc_L2proj_09BilinearForm;
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 1 on a triangle, Lagrange finite element of degree 1 on a triangle]") == 0)
    return new ffc_L2proj_10BilinearForm;
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 2 on a triangle, Lagrange finite element of degree 2 on a triangle]") == 0)
    return new ffc_L2proj_11BilinearForm;
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 1 on a tetrahedron, Lagrange finite element of degree 1 on a tetrahedron, Lagrange finite element of degree 1 on a tetrahedron]") == 0)
    return new ffc_L2proj_12BilinearForm;
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 2 on a tetrahedron, Lagrange finite element of degree 2 on a tetrahedron, Lagrange finite element of degree 2 on a tetrahedron]") == 0)
    return new ffc_L2proj_13BilinearForm;
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 0 on a triangle, Discontinuous Lagrange finite element of degree 0 on a triangle]") == 0)
    return new ffc_L2proj_14BilinearForm;
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 1 on a triangle, Discontinuous Lagrange finite element of degree 1 on a triangle]") == 0)
    return new ffc_L2proj_15BilinearForm;
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 0 on a tetrahedron, Discontinuous Lagrange finite element of degree 0 on a tetrahedron, Discontinuous Lagrange finite element of degree 0 on a tetrahedron]") == 0)
    return new ffc_L2proj_16BilinearForm;
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 1 on a tetrahedron, Discontinuous Lagrange finite element of degree 1 on a tetrahedron, Discontinuous Lagrange finite element of degree 1 on a tetrahedron]") == 0)
    return new ffc_L2proj_17BilinearForm;
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 2 on a tetrahedron, Discontinuous Lagrange finite element of degree 2 on a tetrahedron, Discontinuous Lagrange finite element of degree 2 on a tetrahedron]") == 0)
    return new ffc_L2proj_18BilinearForm;
  if (strcmp(signature, "Brezzi-Douglas-Marini finite element of degree 1 on a triangle") == 0)
    return new ffc_L2proj_19BilinearForm;
  return 0;
}

dolfin::Form* dolfin::ProjectionLibrary::create_projection_L(const char* signature, Function& f)
{
  if (strcmp(signature, "Lagrange finite element of degree 1 on a triangle") == 0)
    return new ffc_L2proj_00LinearForm(f);
  if (strcmp(signature, "Lagrange finite element of degree 2 on a triangle") == 0)
    return new ffc_L2proj_01LinearForm(f);
  if (strcmp(signature, "Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new ffc_L2proj_02LinearForm(f);
  if (strcmp(signature, "Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new ffc_L2proj_03LinearForm(f);
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 0 on a triangle") == 0)
    return new ffc_L2proj_04LinearForm(f);
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 1 on a triangle") == 0)
    return new ffc_L2proj_05LinearForm(f);
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 2 on a triangle") == 0)
    return new ffc_L2proj_06LinearForm(f);
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 0 on a tetrahedron") == 0)
    return new ffc_L2proj_07LinearForm(f);
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 1 on a tetrahedron") == 0)
    return new ffc_L2proj_08LinearForm(f);
  if (strcmp(signature, "Discontinuous Lagrange finite element of degree 2 on a tetrahedron") == 0)
    return new ffc_L2proj_09LinearForm(f);
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 1 on a triangle, Lagrange finite element of degree 1 on a triangle]") == 0)
    return new ffc_L2proj_10LinearForm(f);
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 2 on a triangle, Lagrange finite element of degree 2 on a triangle]") == 0)
    return new ffc_L2proj_11LinearForm(f);
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 1 on a tetrahedron, Lagrange finite element of degree 1 on a tetrahedron, Lagrange finite element of degree 1 on a tetrahedron]") == 0)
    return new ffc_L2proj_12LinearForm(f);
  if (strcmp(signature, "Mixed finite element: [Lagrange finite element of degree 2 on a tetrahedron, Lagrange finite element of degree 2 on a tetrahedron, Lagrange finite element of degree 2 on a tetrahedron]") == 0)
    return new ffc_L2proj_13LinearForm(f);
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 0 on a triangle, Discontinuous Lagrange finite element of degree 0 on a triangle]") == 0)
    return new ffc_L2proj_14LinearForm(f);
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 1 on a triangle, Discontinuous Lagrange finite element of degree 1 on a triangle]") == 0)
    return new ffc_L2proj_15LinearForm(f);
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 0 on a tetrahedron, Discontinuous Lagrange finite element of degree 0 on a tetrahedron, Discontinuous Lagrange finite element of degree 0 on a tetrahedron]") == 0)
    return new ffc_L2proj_16LinearForm(f);
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 1 on a tetrahedron, Discontinuous Lagrange finite element of degree 1 on a tetrahedron, Discontinuous Lagrange finite element of degree 1 on a tetrahedron]") == 0)
    return new ffc_L2proj_17LinearForm(f);
  if (strcmp(signature, "Mixed finite element: [Discontinuous Lagrange finite element of degree 2 on a tetrahedron, Discontinuous Lagrange finite element of degree 2 on a tetrahedron, Discontinuous Lagrange finite element of degree 2 on a tetrahedron]") == 0)
    return new ffc_L2proj_18LinearForm(f);
  if (strcmp(signature, "Brezzi-Douglas-Marini finite element of degree 1 on a triangle") == 0)
    return new ffc_L2proj_19LinearForm(f);
  return 0;
}
