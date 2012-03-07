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
// First added:  2011-09-27
// Last changed: 2011-11-14

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
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
void XMLFunctionData::read(Function& u, const pugi::xml_node xml_dolfin)
{
  dolfin_assert(u.vector());
  GenericVector& vector = *u.vector();
  dolfin_assert(u.function_space());
  const FunctionSpace& V = *u.function_space();

  std::vector<std::pair<uint, uint> > global_to_cell_dof;
  Array<double> x;
  Array<uint> indices;

  const uint num_dofs = V.dim();

  if (MPI::process_number() == 0)
  {
    // Check that we have a XML function data
    const pugi::xml_node xml_function_data_node = xml_dolfin.child("function_data");
    if (!xml_function_data_node)
    {
      dolfin_error("XMLFunctionData.cpp",
                   "read function from XML file",
                   "Not a DOLFIN Function XML file");
    }

    const uint size = xml_function_data_node.attribute("size").as_uint();
    dolfin_assert(vector.size() == size);
    dolfin_assert(vector.size() == num_dofs);

    global_to_cell_dof.resize(num_dofs);
    x.resize(num_dofs);
    indices.resize(num_dofs);

    // Iterate over each cell entry
    for (pugi::xml_node_iterator it = xml_function_data_node.begin(); it != xml_function_data_node.end(); ++it)
    {
      const std::string name = it->name();
      dolfin_assert(name == "dof");

      const uint global_index = it->attribute("index").as_uint();
      const double value = it->attribute("value").as_double();
      const uint cell_index = it->attribute("cell_index").as_uint();
      const uint local_dof_index = it->attribute("cell_dof_index").as_uint();

      global_to_cell_dof[global_index].first = cell_index;
      global_to_cell_dof[global_index].second = local_dof_index;
      x[global_index] = value;
    }
  }

  // Build current dof map based on function space V
  std::vector<std::vector<uint> > dof_map;
  build_dof_map(dof_map, V);

  // Map old-to-current vector positions
  if (MPI::process_number() == 0)
  {
    for (uint i = 0; i < num_dofs; ++i)
    {
      // Indices for data from file
      dolfin_assert(i < global_to_cell_dof.size());
      const uint global_cell_index = global_to_cell_dof[i].first;
      const uint local_dof_index   = global_to_cell_dof[i].second;

      // Local dof vector for V
      const std::vector<uint>& dofs = dof_map[global_cell_index];
      dolfin_assert(local_dof_index < dofs.size());
      const uint new_index = dofs[local_dof_index];

      // File to new
      indices[i] = new_index;
    }

    vector.set(x.data().get(), x.size(), indices.data().get());
  }

  // Finalise vector
  vector.apply("insert");
}
//-----------------------------------------------------------------------------
void XMLFunctionData::write(const Function& u, pugi::xml_node xml_node)
{
  Array<double> x;
  dolfin_assert(u.vector());
  if (MPI::num_processes() > 1)
    u.vector()->gather_on_zero(x);
  else
    u.vector()->get_local(x);

  // Get function space
  dolfin_assert(u.function_space());
  const FunctionSpace& V = *u.function_space();

  // Build map
  std::vector<std::vector<std::pair<uint, uint> > > global_dof_to_cell_dof;
  build_global_to_cell_dof(global_dof_to_cell_dof, V);

  if (MPI::process_number() == 0)
  {
    // Add vector node
    pugi::xml_node function_node = xml_node.append_child("function_data");
    function_node.append_attribute("size") = x.size();

    // Add data
    for (uint i = 0; i < x.size(); ++i)
    {
      dolfin_assert(i < global_dof_to_cell_dof.size());

      pugi::xml_node dof_node = function_node.append_child("dof");
      dof_node.append_attribute("index") = i;
      dof_node.append_attribute("value") = boost::lexical_cast<std::string>(x[i]).c_str();
      dof_node.append_attribute("cell_index") = global_dof_to_cell_dof[i][0].first;
      dof_node.append_attribute("cell_dof_index") = global_dof_to_cell_dof[i][0].second;
    }
  }
}
//-----------------------------------------------------------------------------
void XMLFunctionData::build_global_to_cell_dof(
  std::vector<std::vector<std::pair<uint, uint> > >& global_dof_to_cell_dof,
  const FunctionSpace& V)
{
  // Get mesh and dofmap
  dolfin_assert(V.mesh());
  dolfin_assert(V.dofmap());
  const Mesh& mesh = *V.mesh();
  const GenericDofMap& dofmap = *V.dofmap();

  std::vector<std::vector<std::vector<uint > > > gathered_dofmap;
  std::vector<std::vector<uint > > local_dofmap(mesh.num_cells());

  if (MPI::num_processes() > 1)
  {
    // Get local-to-global cell numbering
    dolfin_assert(mesh.parallel_data().have_global_entity_indices(mesh.topology().dim()));
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

  // Build global dof - (global cell, local dof) map on root process
  if (MPI::process_number() == 0)
  {
    global_dof_to_cell_dof.resize(dofmap.global_dimension());

    std::vector<std::vector<std::vector<uint > > > ::const_iterator proc_dofmap;
    for (proc_dofmap = gathered_dofmap.begin(); proc_dofmap != gathered_dofmap.end(); ++proc_dofmap)
    {
      std::vector<std::vector<uint> >::const_iterator cell_dofmap;
      for (cell_dofmap = proc_dofmap->begin(); cell_dofmap != proc_dofmap->end(); ++cell_dofmap)
      {
        const std::vector<uint>& cell_dofs = *cell_dofmap;
        const uint global_cell_index = cell_dofs.back();
        for (uint i = 0; i < cell_dofs.size() - 1; ++i)
          global_dof_to_cell_dof[cell_dofs[i]].push_back(std::make_pair(global_cell_index, i));
      }
    }
  }
}
//-----------------------------------------------------------------------------
void XMLFunctionData::build_dof_map(std::vector<std::vector<uint> >& dof_map,
                                    const FunctionSpace& V)
{
  // Get mesh and dofmap
  dolfin_assert(V.mesh());
  dolfin_assert(V.dofmap());
  const Mesh& mesh = *V.mesh();
  const GenericDofMap& dofmap = *V.dofmap();

  const uint num_cells = MPI::sum(mesh.num_cells());

  std::vector<std::vector<std::vector<uint > > > gathered_dofmap;
  std::vector<std::vector<uint > > local_dofmap(mesh.num_cells());

  if (MPI::num_processes() > 1)
  {
    // Get local-to-global cell numbering
    dolfin_assert(mesh.parallel_data().have_global_entity_indices(mesh.topology().dim()));
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


  // Build global dofmap on root process
  if (MPI::process_number() == 0)
  {
    dof_map.resize(num_cells);

    // Loop of dof map from each process
    std::vector<std::vector<std::vector<uint > > > ::const_iterator proc_dofmap;
    for (proc_dofmap = gathered_dofmap.begin(); proc_dofmap != gathered_dofmap.end(); ++proc_dofmap)
    {
      std::vector<std::vector<uint> >::const_iterator cell_dofmap;
      for (cell_dofmap = proc_dofmap->begin(); cell_dofmap != proc_dofmap->end(); ++cell_dofmap)
      {
        const std::vector<uint>& cell_dofs = *cell_dofmap;
        const uint global_cell_index = cell_dofs.back();
        dolfin_assert(global_cell_index < dof_map.size());
        dof_map[global_cell_index] = *cell_dofmap;
        dof_map[global_cell_index].pop_back();
      }
    }
  }
}
//-----------------------------------------------------------------------------
