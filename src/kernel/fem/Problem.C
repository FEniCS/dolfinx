// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include "Problem.hh"
#include <Settings.hh>

//-----------------------------------------------------------------------------
Problem::Problem(Grid *grid)
{
  this->grid = grid;
  settings->Get("space dimension",&space_dimension);
  no_nodes = grid->GetNoNodes();
}
//-----------------------------------------------------------------------------
Problem::~Problem()
{

}
//-----------------------------------------------------------------------------
