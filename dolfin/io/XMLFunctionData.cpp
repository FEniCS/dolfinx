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
void XMLFunctionData::read(GenericVector& vector, const FunctionSpace& V,
                           const pugi::xml_node xml_dolfin)
{
    std::vector<std::pair<uint, uint> > global_to_cell_dof;
    Array<double> x;
    Array<uint> indices;

  if (MPI::process_number() == 0)
  {
    // Check that we have a XML function data
    const pugi::xml_node xml_function_data_node = xml_dolfin.child("function_data");
    if (!xml_function_data_node)
      error("XMLFunctionData::read: not a DOLFIN funcation data file.");

    const uint size = xml_function_data_node.attribute("size").as_uint();
    assert(vector.size() == size);

    global_to_cell_dof.resize(size);
    x.resize(size);
    indices.resize(size);

    // Iterate over each cell entry
    for (pugi::xml_node_iterator it = xml_function_data_node.begin(); it != xml_function_data_node.end(); ++it)
    {
      const std::string name = it->name();
      assert(name == "dof");

      const uint global_index = it->attribute("global_index").as_uint();
      const double value = it->attribute("value").as_double();
      const uint cell_index = it->attribute("cell_index").as_uint();
      const uint local_dof_index = it->attribute("local_dof_index").as_uint();

      global_to_cell_dof[global_index].first = cell_index;
      global_to_cell_dof[global_index].second = local_dof_index;
      x[global_index] = value;
    }
  }

  // Build map new
  std::vector<std::vector<std::pair<uint, uint> > > new_global_dof_to_cell_dof;
  build_global_to_cell_dof(new_global_dof_to_cell_dof, V);

  if (MPI::process_number() == 0)
  {
    assert(global_to_cell_dof.size() == new_global_dof_to_cell_dof.size());
    for (uint new_global_index = 0; new_global_index < new_global_dof_to_cell_dof.size(); ++new_global_index)
    {
      const std::vector<std::pair<uint, uint> >& pairs = new_global_dof_to_cell_dof[new_global_index];
      std::vector<std::pair<uint, uint> >::const_iterator it;

      //for (uint i = 0; i < pairs.size(); ++i)
      //  std::cout << "New: " << new_global_index << ", " << pairs[i].first << ", " << pairs[i].second << std::endl;

      for (uint old_global_index = 0; old_global_index < global_to_cell_dof.size(); ++old_global_index)
      {
        //std::cout << "   Old: " << old_global_index << ", " << global_to_cell_dof[old_global_index].first << ", " << global_to_cell_dof[old_global_index].second << std::endl;

        it = std::find(pairs.begin(), pairs.end(), global_to_cell_dof[old_global_index]);
        if (it != pairs.end())
        {
          indices[old_global_index] = new_global_index;
          break;
        }
      }
      assert(it != pairs.end());
    }

    vector.set(x.data().get(), x.size(), indices.data().get());
  }

  // Finalise vector
  vector.apply("insert");
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
      assert(i < global_dof_to_cell_dof.size());

      pugi::xml_node dof_node = function_node.append_child("dof");
      dof_node.append_attribute("global_index") = i;
      dof_node.append_attribute("value") = boost::lexical_cast<std::string>(x[i]).c_str();
      dof_node.append_attribute("cell_index") = global_dof_to_cell_dof[i][0].first;
      dof_node.append_attribute("local_dof_index") = global_dof_to_cell_dof[i][0].second;
    }
  }
}
//-----------------------------------------------------------------------------
void XMLFunctionData::build_global_to_cell_dof(
  std::vector<std::vector<std::pair<uint, uint> > >& global_dof_to_cell_dof,
  const FunctionSpace& V)
{
  // Get mesh and dofmap
  assert(V.mesh());
  const Mesh& mesh = *V.mesh();
  const GenericDofMap& dofmap = V.dofmap();

  std::vector<std::vector<std::vector<uint > > > gathered_dofmap;
  std::vector<std::vector<uint > > local_dofmap(mesh.num_cells());

  if (MPI::num_processes() > 1)
  {
    // Get local-to-global cell numbering
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

