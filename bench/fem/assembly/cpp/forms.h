// Copyright (C) 2008-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-07-22
// Last changed: 2010-05-03

#include <dolfin.h>

#include "forms/Poisson2DP1.h"
#include "forms/Poisson2DP2.h"
#include "forms/Poisson2DP3.h"
#include "forms/THStokes2D.h"
#include "forms/StabStokes2D.h"
#include "forms/Elasticity3D.h"
#include "forms/NSEMomentum3D.h"

#define N_2D 512
#define N_3D 48

using namespace dolfin;

double bench_form(std::string form_name, double (*foo)(Form&))
{
  if (form_name == "LAP1")
  {
    UnitSquare mesh(N_2D, N_2D);
    Poisson2DP1::FunctionSpace V(mesh);
    Poisson2DP1::BilinearForm form(V, V);
    return foo(form);
  }
  else if (form_name == "LAP2")
  {
    UnitSquare mesh(N_2D, N_2D);
    Poisson2DP2::FunctionSpace V(mesh);
    Poisson2DP2::BilinearForm form(V, V);
    return foo(form);
  }
  else if (form_name == "LAP3")
  {
    UnitSquare mesh(N_2D, N_2D);
    Poisson2DP3::FunctionSpace V(mesh);
    Poisson2DP3::BilinearForm form(V, V);
    return foo(form);
  }
  else if (form_name == "TH")
  {
    UnitSquare mesh(N_2D, N_2D);
    THStokes2D::FunctionSpace V(mesh);
    THStokes2D::BilinearForm form(V, V);
    return foo(form);
  }
  else if (form_name == "STAB")
  {
    UnitSquare mesh(N_2D, N_2D);
    StabStokes2D::FunctionSpace V(mesh);
    Constant h(1.0);
    StabStokes2D::BilinearForm form(V, V, h);
    return foo(form);
  }
  else if (form_name == "LE")
  {
    UnitCube mesh(N_3D, N_3D, N_3D);
    Elasticity3D::FunctionSpace V(mesh);
    Elasticity3D::BilinearForm form(V, V);
    return foo(form);
  }
  else if (form_name == "NSE")
  {
    UnitCube mesh(N_3D, N_3D, N_3D);
    NSEMomentum3D::FunctionSpace V(mesh);
    Constant  w(1.0, 1.0, 1.0);
    Constant d1(1.0);
    Constant d2(1.0);
    Constant k(1.0);
    Constant nu(1.0);
    NSEMomentum3D::BilinearForm form(V, V, w, d1, d2, k, nu);
    return foo(form);
  }
  else
  {
    error("Unknown form: %s.", form_name.c_str());
  }

  return 0.0;
}
