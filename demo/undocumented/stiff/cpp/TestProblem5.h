// Copyright (C) 2004-2006 Anders Logg
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
// First added:  2004
// Last changed: 2006-08-21

#include <cmath>
#include <dolfin.h>

using namespace dolfin;

class TestProblem5 : public ODE
{
public:

  TestProblem5() : ODE(6, 180.0)
  {
    info("The Chemical Akzo-Nobel problem.");

    k1  = 18.7;
    k2  = 0.58;
    k3  = 0.09;
    k4  = 0.42;
    K   = 34.4;
    klA = 3.3;
    Ks  = 115.83;
    p   = 0.9;
    H   = 737.0;
  }

  void u0(Array<real>& u)
  {
    u[0] = 0.444;
    u[1] = 0.00123;
    u[2] = 0.0;
    u[3] = 0.007;
    u[4] = 0.0;
    u[5] = 0.36;
  }

  void f(const Array<real>& u, real t, Array<real>& y)
  {
    y[0] = -2.0*r1(u) + r2(u) - r3(u) - r4(u);
    y[1] = -0.5*r1(u) - r4(u) - 0.5*r5(u) + F(u);
    y[2] = r1(u) - r2(u) + r3(u);
    y[3] = -r2(u) + r3(u) - 2.0*r4(u);
    y[4] = r2(u) - r3(u) + r5(u);
    y[5] = Ks*u[0]*u[3] - u[5];
  }

private:

  real r1(const Array<real>& u)
  {
    return k1*real_pow(u[0], 4.0)*real_sqrt(u[1]);
  }

  real r2(const Array<real>& u)
  {
    return k2*u[2]*u[3];
  }

  real r3(const Array<real>& u)
  {
    return (k2/K)*u[0]*u[4];
  }

  real r4(const Array<real>& u)
  {
    return k3*u[0]*real_pow(u[3], 2.0);
  }

  real r5(const Array<real>& u)
  {
    return k4*real_pow(u[5], 2.0)*real_sqrt(u[1]);
  }

  real F(const Array<real>& u)
  {
    return klA * (p/H - u[1]);
  }

  real k1, k2, k3, k4, K, klA, Ks, p, H;

};
