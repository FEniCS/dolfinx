// Copyright (C) 2008-2009 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-07-22
// Last changed: 2010-05-03

#include <memory>
#include <dolfin.h>

#include "forms/Poisson2DP1.h"
#include "forms/Poisson2DP2.h"
#include "forms/Poisson2DP3.h"
#include "forms/THStokes2D.h"
#include "forms/StabStokes2D.h"
#include "forms/Elasticity3D.h"
#include "forms/NSEMomentum3D.h"

#define SIZE_2D 256
#define SIZE_3D 32

using namespace dolfin;

double bench_form(std::string form_name, double (*foo)(Form&))
{
  if (form_name == "poisson1")
  {
    auto mesh = std::make_shared<UnitSquareMesh>(SIZE_2D, SIZE_2D);
    auto V = std::make_shared<Poisson2DP1::FunctionSpace>(mesh);
    Poisson2DP1::BilinearForm form(V, V);
    return foo(form);
  }
  else if (form_name == "poisson2")
  {
    auto mesh = std::make_shared<UnitSquareMesh>(SIZE_2D, SIZE_2D);
    auto V = std::make_shared<Poisson2DP2::FunctionSpace>(mesh);
    Poisson2DP2::BilinearForm form(V, V);
    return foo(form);
  }
  else if (form_name == "poisson3")
  {
    auto mesh = std::make_shared<UnitSquareMesh>(SIZE_2D, SIZE_2D);
    auto V = std::make_shared<Poisson2DP3::FunctionSpace>(mesh);
    Poisson2DP3::BilinearForm form(V, V);
    return foo(form);
  }
  else if (form_name == "stokes")
  {
    auto mesh = std::make_shared<UnitSquareMesh>(SIZE_2D, SIZE_2D);
    auto V = std::make_shared<THStokes2D::FunctionSpace>(mesh);
    THStokes2D::BilinearForm form(V, V);
    return foo(form);
  }
  else if (form_name == "stabilization")
  {
    auto mesh = std::make_shared<UnitSquareMesh>(SIZE_2D, SIZE_2D);
    auto V = std::make_shared<StabStokes2D::FunctionSpace>(mesh);
    auto h = std::make_shared<Constant>(1.0);
    StabStokes2D::BilinearForm form(V, V, h);
    return foo(form);
  }
  else if (form_name == "elasticity")
  {
    auto mesh = std::make_shared<UnitCubeMesh>(SIZE_3D, SIZE_3D, SIZE_3D);
    auto V = std::make_shared<Elasticity3D::FunctionSpace>(mesh);
    Elasticity3D::BilinearForm form(V, V);
    return foo(form);
  }
  else if (form_name == "navierstokes")
  {
    auto mesh = std::make_shared<UnitCubeMesh>(SIZE_3D, SIZE_3D, SIZE_3D);
    auto V = std::make_shared<NSEMomentum3D::FunctionSpace>(mesh);
    auto w = std::make_shared<Constant>(1.0, 1.0, 1.0);
    auto d1 = std::make_shared<Constant>(1.0);
    auto d2 = std::make_shared<Constant>(1.0);
    auto k = std::make_shared<Constant>(1.0);
    auto nu = std::make_shared<Constant>(1.0);
    NSEMomentum3D::BilinearForm form(V, V, w, d1, d2, k, nu);
    return foo(form);
  }
  else
  {
    error("Unknown form: %s.", form_name.c_str());
  }

  return 0.0;
}
