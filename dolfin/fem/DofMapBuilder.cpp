// Copyright (C) 2008-2011 Anders Logg and Ola Skavhaug
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
// Modified by Niclas Jansson 2009.
// Modified by Garth N. Wells 2010.
// Modified by Joachim B Haga, 2012.
//
// First added:  2008-08-12
// Last changed: 2012-02-29

#include <ufc.h>
#include <boost/random.hpp>
#include <boost/unordered_map.hpp>

#include <dolfin/common/Timer.h>
#include <dolfin/graph/BoostGraphRenumbering.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/BoundaryMesh.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include "DofMap.h"
#include "UFCCell.h"
#include "UFCMesh.h"
#include "DofMapBuilder.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
void DofMapBuilder::build(DofMap& dofmap, const Mesh& dolfin_mesh,
                          const UFCMesh& ufc_mesh, bool distributed)
{
  // Start timer for dofmap initialization
  Timer t0("Init dofmap");

  // Create space for dof map
  dofmap._dofmap.resize(dolfin_mesh.num_cells());

  dofmap._off_process_owner.clear();

  dolfin_assert(dofmap._ufc_dofmap);

  // Build dofmap from ufc::dofmap
  dolfin::UFCCell ufc_cell(dolfin_mesh);
  for (dolfin::CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
  {
    // Update UFC cell
    ufc_cell.update(*cell);

    // Get standard local dimension
    const unsigned int local_dim = dofmap._ufc_dofmap->local_dimension(ufc_cell);
    dofmap._dofmap[cell->index()].resize(local_dim);

    // Tabulate standard UFC dof map
    dofmap._ufc_dofmap->tabulate_dofs(&dofmap._dofmap[cell->index()][0],
                                      ufc_mesh, ufc_cell);
  }

  // Build (renumber) dofmap when running in parallel
  if (distributed)
  {
    // Build set of global dofs
    const set global_dofs = compute_global_dofs(dofmap, dolfin_mesh);

    // Build distributed dof map
    build_distributed(dofmap, global_dofs, dolfin_mesh);
  }
  else
    dofmap._ownership_range = std::make_pair(0, dofmap.global_dimension());
}
//-----------------------------------------------------------------------------
void DofMapBuilder::build_distributed(DofMap& dofmap,
                                      const DofMapBuilder::set& global_dofs,
                                      const Mesh& mesh)
{
  // Create data structures
  DofMapBuilder::set owned_dofs, shared_owned_dofs, shared_unowned_dofs;
  DofMapBuilder::vec_map shared_dof_processes;

  // Computed owned and shared dofs (and owned and un-owned)
  compute_ownership(owned_dofs, shared_owned_dofs, shared_unowned_dofs,
                    shared_dof_processes, dofmap, global_dofs, mesh);

  // Renumber owned dofs and receive new numbering for unowned shared dofs
  parallel_renumber(owned_dofs, shared_owned_dofs, shared_unowned_dofs,
                    shared_dof_processes, dofmap, mesh);
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_ownership(set& owned_dofs, set& shared_owned_dofs,
                                      set& shared_unowned_dofs,
                                      vec_map& shared_dof_processes,
                                      const DofMap& dofmap,
                                      const DofMapBuilder::set& global_dofs,
                                      const Mesh& mesh)
{
  log(TRACE, "Determining dof ownership for parallel dof map");

  // Create a radom number generator for ownership 'voting'
  boost::mt19937 engine(MPI::process_number());
  boost::uniform_int<> distribution(0, 100000000);
  boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rng(engine, distribution);

  // Clear data structures
  owned_dofs.clear();
  shared_owned_dofs.clear();
  shared_unowned_dofs.clear();

  // Data structures for computing ownership
  boost::unordered_map<uint, uint> dof_vote;
  std::vector<uint> facet_dofs(dofmap.num_facet_dofs());

  // Communication buffer
  std::vector<uint> send_buffer;

  // Extract the interior boundary
  BoundaryMesh interior_boundary;
  interior_boundary.init_interior_boundary(mesh);

  // Build set of dofs on process boundary (assume all are owned by this process)
  const MeshFunction<unsigned int>& cell_map = interior_boundary.cell_map();
  if (!cell_map.empty())
  {
    for (CellIterator bc(interior_boundary); !bc.end(); ++bc)
    {
      // Get boundary facet
      Facet f(mesh, cell_map[*bc]);

      // Get cell to which facet belongs (pick first)
      Cell c(mesh, f.entities(mesh.topology().dim())[0]);

      // Tabulate dofs on cell
      const std::vector<uint>& cell_dofs = dofmap.cell_dofs(c.index());

      // Tabulate which dofs are on the facet
      dofmap.tabulate_facet_dofs(&facet_dofs[0], c.index(f));

      // Insert shared dofs into set and assign a 'vote'
      for (uint i = 0; i < dofmap.num_facet_dofs(); i++)
      {
        if (shared_owned_dofs.find(cell_dofs[facet_dofs[i]]) == shared_owned_dofs.end())
        {
          shared_owned_dofs.insert(cell_dofs[facet_dofs[i]]);
          dof_vote[cell_dofs[facet_dofs[i]]] = rng();

          send_buffer.push_back(cell_dofs[facet_dofs[i]]);
          send_buffer.push_back(dof_vote[cell_dofs[facet_dofs[i]]]);
        }
      }
    }
  }

  // Decide ownership of shared dofs
  const uint num_proc = MPI::num_processes();
  const uint proc_num = MPI::process_number();
  std::vector<uint> recv_buffer;
  for (uint k = 1; k < MPI::num_processes(); ++k)
  {
    const uint src  = (proc_num - k + num_proc) % num_proc;
    const uint dest = (proc_num + k) % num_proc;
    MPI::send_recv(send_buffer, dest, recv_buffer, src);

    for (uint i = 0; i < recv_buffer.size(); i += 2)
    {
      const uint received_dof  = recv_buffer[i];
      const uint received_vote = recv_buffer[i + 1];

      if (shared_owned_dofs.find(received_dof) != shared_owned_dofs.end())
      {
        // Move dofs with higher ownership votes from shared to shared but not owned
        if (received_vote < dof_vote[received_dof])
        {
          shared_unowned_dofs.insert(received_dof);
          shared_owned_dofs.erase(received_dof);
        }
        else if (received_vote == dof_vote[received_dof])
        {
          // FIXME: Eventually replace this with a more robust condition. It's
          // good for testing that ownership of shared dofs is spread roughly
          // equally
          dolfin_error("DofMapBuilder.cpp",
                       "compute mapping of degrees of freedom",
                       "Cannot decide on dof ownership; votes are equal");
        }

        // Remember the sharing of the dof
        shared_dof_processes[received_dof].push_back(src);
      }
      else if (shared_unowned_dofs.find(received_dof) != shared_unowned_dofs.end())
      {
        // Remember the sharing of the dof
        shared_dof_processes[received_dof].push_back(src);
      }
    }
  }

  // Add/remove global dofs to relevant sets (process 0 owns local dofs)
  if (MPI::process_number() == 0)
  {
    shared_owned_dofs.insert(global_dofs.begin(), global_dofs.end());
    for (set::const_iterator dof = global_dofs.begin(); dof != global_dofs.begin(); ++dof)
    {
      set::const_iterator _dof = shared_unowned_dofs.find(*dof);
      if (_dof != shared_unowned_dofs.end())
        shared_unowned_dofs.erase(_dof);
    }
  }
  else
  {
    shared_unowned_dofs.insert(global_dofs.begin(), global_dofs.end());
    for (set::const_iterator dof = global_dofs.begin(); dof != global_dofs.begin(); ++dof)
    {
      set::const_iterator _dof = shared_owned_dofs.find(*dof);
      if (_dof != shared_owned_dofs.end())
        shared_owned_dofs.erase(_dof);
    }
  }

  // Mark all shared and owned dofs as owned by the processes
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const std::vector<uint>& cell_dofs = dofmap.cell_dofs(cell->index());
    const uint cell_dimension = dofmap.cell_dimension(cell->index());
    for (uint i = 0; i < cell_dimension; ++i)
    {
      // Mark dof as owned if in unowned set
      if (shared_unowned_dofs.find(cell_dofs[i]) == shared_unowned_dofs.end())
        owned_dofs.insert(cell_dofs[i]);
    }
  }

  // Check that sum of locally owned dofs is equal to global dimension
  const uint _owned_dim = owned_dofs.size();
  dolfin_assert(MPI::sum(_owned_dim) == dofmap.global_dimension());

  log(TRACE, "Finished determining dof ownership for parallel dof map");
}
//-----------------------------------------------------------------------------
void DofMapBuilder::parallel_renumber(const set& owned_dofs,
                             const set& shared_owned_dofs,
                             const set& shared_unowned_dofs,
                             const vec_map& shared_dof_processes,
                             DofMap& dofmap, const Mesh& mesh)
{
  log(TRACE, "Renumber dofs for parallel dof map");

  // FIXME: Handle double-renumbered dof map
  if (!dofmap.ufc_map_to_dofmap.empty())
  {
    dolfin_error("DofMapBuilder.cpp",
                 "compute parallel renumbering of degrees of freedom",
                 "The degree of freedom mapping cannot (yet) be renumbered twice");
  }

  const std::vector<std::vector<uint> >& old_dofmap = dofmap._dofmap;
  std::vector<std::vector<uint> > new_dofmap(old_dofmap.size());
  dolfin_assert(old_dofmap.size() == mesh.num_cells());

  // Compute offset for owned and non-shared dofs
  const uint process_offset = MPI::global_offset(owned_dofs.size(), true);

  // Clear some data
  dofmap._off_process_owner.clear();

  // Build vector of owned dofs
  const std::vector<uint> my_dofs(owned_dofs.begin(), owned_dofs.end());

  cout << "Num cells: " << mesh.num_cells() << endl;
  cout << "Local of map dim: " << old_dofmap.size() << endl;
  cout << "my dofs dim: " << my_dofs.size() << endl;
  cout << "Number of owned dofs: " << owned_dofs.size() << endl;


  // Create contiguous local numbering for locally owned dofs
  uint my_counter = 0;
  boost::unordered_map<uint, uint> my_old_to_new_dof_index;
  for (set_iterator owned_dof = owned_dofs.begin(); owned_dof != owned_dofs.end(); ++owned_dof, my_counter++)
    my_old_to_new_dof_index[*owned_dof] = my_counter;

  // Build local graph based on old dof map with contiguous numbering
  //std::vector<std::vector<uint> > graph(old_dofmap.size());
  Graph graph(owned_dofs.size());
  for (uint cell = 0; cell < old_dofmap.size(); ++cell)
  {
    //cout << "Cell index: " << cell << endl;
    const std::vector<uint>& dofs0 = dofmap.cell_dofs(cell);
    const std::vector<uint>& dofs1 = dofmap.cell_dofs(cell);

    std::vector<uint>::const_iterator node0, node1;
    for (node0 = dofs0.begin(); node0 != dofs0.end(); ++node0)
    {
      //cout << "Loop 0: " << *node0 << endl;
      boost::unordered_map<uint, uint>::const_iterator _node0 = my_old_to_new_dof_index.find(*node0);
      if (_node0 != my_old_to_new_dof_index.end())
      {
        const uint local_node0 = _node0->second;
        //cout << "if 0: " << local_node0 << endl;
        dolfin_assert(local_node0 < graph.size());
        for (node1 = dofs1.begin(); node1 != dofs1.end(); ++node1)
        {
          //cout << "Loop 1: " << *node1 << endl;
          boost::unordered_map<uint, uint>::const_iterator _node1 = my_old_to_new_dof_index.find(*node1);
          if (_node1 != my_old_to_new_dof_index.end())
          {
            const uint local_node1 = _node1->second;
            //cout << "Insert: " << local_node0 << ", " << local_node1 << endl;
            graph[local_node0].insert(local_node1);
          }
        }
      }
    }
  }

  /*
  for (uint i = 0; i < graph.size(); ++i)
  {
    cout << "Row: " << i << endl;
    for (uint j = 0; j < graph[i].size(); ++j)
      cout << "  Col: " << j << ", " << graph[i][j] << endl;
  }
  */


  //cout << "(0) Min, max, size: " << *std::min_element(dof_remap.begin(), dof_remap.end())
  //  << ", " <<  *std::min_element(dof_remap.begin(), dof_remap.end()) << ", " << dof_remap.size() << endl;

  // Reorder dofs
  const std::vector<uint> dof_remap = BoostGraphRenumbering::compute_king(graph);
  //cout << "Remap: " << endl;
  //for (uint i = 0; i < dof_remap.size(); ++i)
  //  cout << dof_remap[i] << endl;

  cout << "(1) Min, max, size: " << *std::min_element(dof_remap.begin(), dof_remap.end())
    << ", " <<  *std::max_element(dof_remap.begin(), dof_remap.end()) << ", " << dof_remap.size() << endl;
  std::set<uint> tmp(dof_remap.begin(), dof_remap.end());
  cout << "Duplicate test: " << tmp.size() << ", " <<  dof_remap.size() << endl;

  /*
  // Build local sparsity graph
  cout << "Build graph: " << MPI::process_number() << endl;
  const Graph graph = GraphBuilder::local_graph(mesh, dofmap, dofmap);
  cout << "Build reordering map: " << MPI::process_number() << endl;
  const std::vector<uint> dof_remap = BoostGraphRenumbering::compute_king(graph);
  cout << "End build reordering map: " << MPI::process_number() << endl;

  cout << "Owned dof min/max: " << *(std::min_element(owned_dofs.begin(), owned_dofs.end()))
     << ", " << *(std::max_element(owned_dofs.begin(), owned_dofs.end()))  << endl;
  */

  // Map from old to new index for dofs
  boost::unordered_map<uint, uint> old_to_new_dof_index;

  // Renumber owned dofs and buffer dofs that are owned but shared with another
  // process
  uint counter = 0;
  std::vector<uint> send_buffer;
  for (set_iterator owned_dof = owned_dofs.begin(); owned_dof != owned_dofs.end(); ++owned_dof, counter++)
  {
    // Set new dof number
    //old_to_new_dof_index[*owned_dof] = process_offset + counter;
    old_to_new_dof_index[*owned_dof] = process_offset + dof_remap[counter];

    // Update UFC-to-renumbered map for new number
    //dofmap.ufc_map_to_dofmap[*owned_dof] = process_offset + counter;
    dofmap.ufc_map_to_dofmap[*owned_dof] = process_offset + dof_remap[counter];

    // If this dof is shared and owned, buffer old and new index for sending
    if (shared_owned_dofs.find(*owned_dof) != shared_owned_dofs.end())
    {
      send_buffer.push_back(*owned_dof);
      send_buffer.push_back(process_offset + dof_remap[counter]);
    }
  }

  // Exchange new dof numbers for dofs that are shared
  const uint num_proc = MPI::num_processes();
  const uint proc_num = MPI::process_number();
  std::vector<uint> recv_buffer;
  for (uint k = 1; k < MPI::num_processes(); ++k)
  {
    const uint src  = (proc_num - k + num_proc) % num_proc;
    const uint dest = (proc_num + k) % num_proc;
    MPI::send_recv(send_buffer, dest, recv_buffer, src);

    // Add dofs renumbered by another process to the old-to-new map
    for (uint i = 0; i < recv_buffer.size(); i += 2)
    {
      const uint received_old_dof_index = recv_buffer[i];
      const uint received_new_dof_index = recv_buffer[i + 1];

      // Check if this process has shared dof (and is not the owner)
      if (shared_unowned_dofs.find(received_old_dof_index) != shared_unowned_dofs.end())
      {
        // Add to old-to-new dof map
        old_to_new_dof_index[received_old_dof_index] = received_new_dof_index;

        // Store map from off-process dof to owner
        dofmap._off_process_owner[received_new_dof_index] = src;

        // Update UFC-to-renumbered map
        dofmap.ufc_map_to_dofmap[received_old_dof_index] = received_new_dof_index;
      }
    }
  }

  // Insert the shared-dof-to-process mapping into the dofmap, renumbering
  // as necessary
  for (vec_map::const_iterator it = shared_dof_processes.begin();
            it != shared_dof_processes.end(); ++it)
  {
    boost::unordered_map<uint, uint>::const_iterator new_index = old_to_new_dof_index.find(it->first);
    if (new_index == old_to_new_dof_index.end())
      dofmap._shared_dofs.insert(*it);
    else
      dofmap._shared_dofs.insert(std::make_pair(new_index->second, it->second));
    dofmap._neighbours.insert(it->second.begin(), it->second.end());
  }

  // Build new dof map
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    const uint cell_index = cell->index();
    const uint cell_dimension = dofmap.cell_dimension(cell_index);

    // Resize cell map and insert dofs
    new_dofmap[cell_index].resize(cell_dimension);
    for (uint i = 0; i < cell_dimension; ++i)
    {
      const uint old_index = old_dofmap[cell_index][i];
      new_dofmap[cell_index][i] = old_to_new_dof_index[old_index];
    }
  }

  // Set new dof map
  dofmap._dofmap = new_dofmap;

  // Set ownership range
  dofmap._ownership_range = std::make_pair<uint, uint>(process_offset, process_offset + owned_dofs.size());

  log(TRACE, "Finished renumbering dofs for parallel dof map");
}
//-----------------------------------------------------------------------------
DofMapBuilder::set DofMapBuilder::compute_global_dofs(const DofMap& dofmap,
                                                       const Mesh& dolfin_mesh)
{
  // Wrap UFC dof map
  boost::shared_ptr<const ufc::dofmap> _dofmap(dofmap._ufc_dofmap.get(),
                                               NoDeleter());

  // Create UFCMesh
  const UFCMesh ufc_mesh(dolfin_mesh);

  // Compute global dof indices
  uint offset = 0;
  set global_dof_indices;
  compute_global_dofs(global_dof_indices, offset, _dofmap, dolfin_mesh, ufc_mesh);

  return global_dof_indices;
}
//-----------------------------------------------------------------------------
void DofMapBuilder::compute_global_dofs(DofMapBuilder::set& global_dofs,
                            uint& offset,
                            boost::shared_ptr<const ufc::dofmap> dofmap,
                            const Mesh& dolfin_mesh, const UFCMesh& ufc_mesh)
{
  dolfin_assert(dofmap);
  const uint D = dolfin_mesh.topology().dim();

  if (dofmap->num_sub_dofmaps() == 0)
  {
    // Check if dofmap is for global dofs
    bool global_dof = true;
    for (uint d = 0; d <= D; ++d)
    {
      if (dofmap->needs_mesh_entities(d))
      {
        global_dof = false;
        break;
      }
    }

    if (global_dof)
    {
      // Check that we have just one dof
      if (dofmap->global_dimension() != 1)
      {
        dolfin_error("DofMapBuilder.cpp",
                     "compute global degrees of freedom",
                     "Global degree of freedom has dimension != 1");
      }

      boost::scoped_ptr<ufc::mesh> ufc_mesh(new ufc::mesh);
      boost::scoped_ptr<ufc::cell> ufc_cell(new ufc::cell);
      uint dof = 0;
      dofmap->tabulate_dofs(&dof, *ufc_mesh, *ufc_cell);

      // Insert global dof index
      std::pair<DofMapBuilder::set::iterator, bool> ret = global_dofs.insert(dof + offset);
      if (!ret.second)
      {
        dolfin_error("DofMapBuilder.cpp",
                     "compute global degrees of freedom",
                     "Global degree of freedom already exists");
      }
    }
  }
  else
  {
    for (uint i = 0; i < dofmap->num_sub_dofmaps(); ++i)
    {
      // Extract sub-dofmap and intialise
      boost::shared_ptr<ufc::dofmap> sub_dofmap(dofmap->create_sub_dofmap(i));
      DofMap::init_ufc_dofmap(*sub_dofmap, ufc_mesh, dolfin_mesh);

      compute_global_dofs(global_dofs, offset, sub_dofmap, dolfin_mesh,
                          ufc_mesh);

      // Get offset
      if (sub_dofmap->num_sub_dofmaps() == 0)
        offset += sub_dofmap->global_dimension();
    }
  }
}
//-----------------------------------------------------------------------------
