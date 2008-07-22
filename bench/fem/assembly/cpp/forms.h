// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-22
// Last changed: 2008-07-22

#include <dolfin.h>

#include "forms/Poisson2DP1.h"
#include "forms/Poisson2DP2.h"
#include "forms/Poisson2DP3.h"
#include "forms/THStokes2D.h"
#include "forms/StabStokes2D.h"
#include "forms/Elasticity3D.h"
#include "forms/NSEMomentum3D.h"

#define N_2D 256
#define N_3D 32

using namespace dolfin;

double bench_form(std::string form_name, double (*bench_form)(Form&, Mesh&))
{
  if (form_name == "Poisson2DP1")
  {
    UnitSquare mesh(N_2D, N_2D);
    Poisson2DP1BilinearForm form;
    return bench_form(form, mesh);
  }
  else if (form_name == "Poisson2DP2")
  {
    UnitSquare mesh(N_2D, N_2D);
    Poisson2DP2BilinearForm form;
    return bench_form(form, mesh);
  }
  else if (form_name == "Poisson2DP3")
  {
    UnitSquare mesh(N_2D, N_2D);
    Poisson2DP2BilinearForm form;
    return bench_form(form, mesh);
  }
  else if (form_name == "THStokes2D")
  {
    UnitSquare mesh(N_2D, N_2D);
    THStokes2DBilinearForm form;
    return bench_form(form, mesh);
  }
  else if (form_name == "StabStokes2D")
  {
    UnitSquare mesh(N_2D, N_2D);
    Function h(mesh, 1.0);
    StabStokes2DBilinearForm form(h);
    return bench_form(form, mesh);
  }
  else if (form_name == "Elasticity3D")
  {
    UnitCube mesh(N_3D, N_3D, N_3D);
    Elasticity3DBilinearForm form;
    return bench_form(form, mesh);
  }
  else if (form_name == "NSEMomentum3D")
  {
    UnitCube mesh(N_3D, N_3D, N_3D);
    Function w(mesh, 3, 1.0);
    Function d1(mesh, 1.0);
    Function d2(mesh, 1.0);
    Function k(mesh, 1.0);
    Function nu(mesh, 1.0);
    NSEMomentum3DBilinearForm form(w, d1, d2, k, nu);
    return bench_form(form, mesh);
  }
  else
  {
    error("Unknown form: %s.", form_name.c_str());
  }
  
  return 0.0;
}
