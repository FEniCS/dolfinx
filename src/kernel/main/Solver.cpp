// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Solver.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Solver::Solver() : mesh(dummy_mesh), ode(dummy_ode)
{

}
//-----------------------------------------------------------------------------
Solver::Solver(Mesh& mesh_) : mesh(mesh_), ode(dummy_ode)
{

}
//-----------------------------------------------------------------------------
Solver::Solver(ODE& ode_) : mesh(dummy_mesh), ode(ode_)
{

}
//-----------------------------------------------------------------------------
Solver::~Solver()
{

}
//-----------------------------------------------------------------------------
