// Copyright (C) 2011 Garth N. Wells
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
// First added:  2002-12-06
// Last changed: 2011-06-30

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <boost/lexical_cast.hpp>
#include "pugixml.hpp"

#include "dolfin/common/Array.h"
#include "dolfin/common/MPI.h"
#include "dolfin/fem/GenericDofMap.h"
#include "dolfin/function/Function.h"
#include "dolfin/function/FunctionSpace.h"
#include "dolfin/la/GenericVector.h"
#include "dolfin/log/log.h"
#include "dolfin/mesh/Mesh.h"
#include "dolfin/mesh/MeshTopology.h"
#include "dolfin/mesh/ParallelData.h"
#include "XMLFunctionData.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void XMLFunctionData::read(GenericVector& vector, const FunctionSpace& V,
                           const pugi::xml_node xml_dolfin)
{
  error("XMLFunctionData::read not implemented.");
}
//-----------------------------------------------------------------------------
void XMLFunctionData::write(const Function& u,
                      pugi::xml_node xml_node, bool write_to_stream)
{
  Array<double> x;
  if (MPI::num_processes() > 1)
    u.vector().gather_on_zero(x);
  else
    u.vector().get_local(x);

  // Get function space
  const FunctionSpace& V = u.function_space();

  // Get mesh and dofmap
  assert(V.mesh());
  const Mesh& mesh = *V.mesh();
  const GenericDofMap& dofmap = V.dofmap();

  std::vector<std::vector<std::vector<uint > > > gathered_dofmap;
  std::vector<std::vector<uint > > local_dofmap(mesh.num_cells());

  // Get local-to-global cell numbering
  if (MPI::num_processes() > 1)
  {
    assert(mesh.parallel_data().have_global_entity_indices(mesh.topology().dim()));
    const MeshFunction<uint>& global_cell_indices
      = mesh.parallel_data().global_entity_indices(mesh.topology().dim());

    // Build dof map data with global cell indices
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const uint local_cell_index = cell->index();
      const uint global_cell_index = global_cell_indices[*cell];
      local_dofmap[local_cell_index] = dofmap.cell_dofs(local_cell_index);
      local_dofmap[local_cell_index].push_back(global_cell_index);
    }
  }
  else
  {
    // Build dof map data
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      const uint local_cell_index = cell->index();
      local_dofmap[local_cell_index] = dofmap.cell_dofs(local_cell_index);
      local_dofmap[local_cell_index].push_back(local_cell_index);
    }
  }

  // Gather dof map data on root process
  MPI::gather(local_dofmap, gathered_dofmap);

  std::vector<std::pair<uint, uint> > global_dof_to_cell_dof;

  // Build global dof - (global cell, local dof) map on root process
  if (MPI::process_number() == 0)
  {
    global_dof_to_cell_dof.resize(x.size());
    assert(x.size() == dofmap.global_dimension());

    std::vector<std::vector<std::vector<uint > > > ::const_iterator proc_dofmap;
    for (proc_dofmap = gathered_dofmap.begin(); proc_dofmap != gathered_dofmap.end(); ++proc_dofmap)
    {
      std::vector<std::vector<uint> >::const_iterator cell_dofmap;
      for (cell_dofmap = proc_dofmap->begin(); cell_dofmap != proc_dofmap->end(); ++cell_dofmap)
      {
        const std::vector<uint>& cell_dofs = *cell_dofmap;
        const uint global_cell_index = cell_dofs.back();
        for (uint i = 0; i < cell_dofs.size() - 1; ++i)
          global_dof_to_cell_dof[cell_dofs[i]] = std::make_pair(global_cell_index, i);
      }
    }

    // Add vector node
    pugi::xml_node function_node = xml_node.append_child("function_data");
    function_node.append_attribute("size") = x.size();

    // Add data
    for (uint i = 0; i < x.size(); ++i)
    {
      assert(i < global_dof_to_cell_dof.size());

      pugi::xml_node dof_node = function_node.append_child("dof");
      dof_node.append_attribute("global_index") = i;
      dof_node.append_attribute("value") = boost::lexical_cast<std::string>(x[i]).c_str();
      dof_node.append_attribute("cell_index") = global_dof_to_cell_dof[i].first;
      dof_node.append_attribute("local_dof_index") = global_dof_to_cell_dof[i].second;
    }
  }
}
//-----------------------------------------------------------------------------

