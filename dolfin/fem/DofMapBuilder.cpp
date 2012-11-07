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
#include <dolfin/graph/BoostGraphOrdering.h>
#include <dolfin/graph/GraphBuilder.h>
#include <dolfin/graph/SCOTCH.h>
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
                          const UFCMesh& ufc_mesh,
                          bool reorder, bool distributed)
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

  // Build (re-order) dofmap when running in parallel
  if (distributed)
  {
    // Build set of global dofs
    const set global_dofs = compute_global_dofs(dofmap, dolfin_mesh);

    // Build distributed dof map
    build_distributed(dofmap, global_dofs, dolfin_mesh);
  }
  else
  {
    if (reorder)
    {
      // Build graph
      Graph graph(dofmap.global_dimension());
      for (CellIterator cell(dolfin_mesh); !cell.end(); ++cell)
      {
        const std::vector<uint>& dofs0 = dofmap.cell_dofs(cell->index());
        const std::vector<uint>& dofs1 = dofmap.cell_dofs(cell->index());
        std::vector<uint>::const_iterator node;
        for (node = dofs0.begin(); node != dofs0.end(); ++node)
          graph[*node].insert(dofs1.begin(), dofs1.end());
      }

      // Reorder graph (reverse Cuthill-McKee)
      const std::vector<std::size_t> dof_remap
          = BoostGraphOrdering::compute_cuthill_mckee(graph, true);

      // Reorder dof map
      dolfin_assert(dofmap.ufc_map_to_dofmap.empty());
      for (uint i = 0; i < dofmap.global_dimension(); ++i)
        dofmap.ufc_map_to_dofmap[i] = dof_remap[i];

      // Re-number dofs for cell
      std::vector<std::vector<uint> >::iterator cell_map;
      std::vector<uint>::iterator dof;
      for (cell_map = dofmap._dofmap.begin(); cell_map != dofmap._dofmap.end(); ++cell_map)
        for (dof = cell_map->begin(); dof != cell_map->end(); ++dof)
          *dof = dof_remap[*dof];
    }
    dofmap._ownership_range = std::make_pair(0, dofmap.global_dimension());
  }
  
  // Periodic modification. Compute master-slave pairs and eliminate slaves 
  if (!dolfin_mesh.facet_pairs.empty())
  {  
    periodic_modification(dofmap, dolfin_mesh);
    dofmap._global_dim = dofmap._ufc_dofmap->global_dimension() - dofmap._num_slaves;
  }
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
        // Move dofs with higher ownership votes from shared to shared
        // but not owned
        if (received_vote < dof_vote[received_dof])
        {
          shared_unowned_dofs.insert(received_dof);
          shared_owned_dofs.erase(received_dof);
        }
        else if (received_vote == dof_vote[received_dof] && proc_num > src)
        {
          // If votes are equal, let lower rank process take ownership
          shared_unowned_dofs.insert(received_dof);
          shared_owned_dofs.erase(received_dof);
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
  dolfin_assert(MPI::sum(_owned_dim) == dofmap._ufc_dofmap->global_dimension());

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

  // Create contiguous local numbering for locally owned dofs
  uint my_counter = 0;
  boost::unordered_map<uint, uint> my_old_to_new_dof_index;
  for (set_iterator owned_dof = owned_dofs.begin(); owned_dof != owned_dofs.end(); ++owned_dof, my_counter++)
    my_old_to_new_dof_index[*owned_dof] = my_counter;

  // Build local graph based on old dof map with contiguous numbering
  Graph graph(owned_dofs.size());
  for (uint cell = 0; cell < old_dofmap.size(); ++cell)
  {
    const std::vector<uint>& dofs0 = dofmap.cell_dofs(cell);
    const std::vector<uint>& dofs1 = dofmap.cell_dofs(cell);
    std::vector<uint>::const_iterator node0, node1;
    for (node0 = dofs0.begin(); node0 != dofs0.end(); ++node0)
    {
      boost::unordered_map<uint, uint>::const_iterator _node0
          = my_old_to_new_dof_index.find(*node0);
      if (_node0 != my_old_to_new_dof_index.end())
      {
        const uint local_node0 = _node0->second;
        dolfin_assert(local_node0 < graph.size());
        for (node1 = dofs1.begin(); node1 != dofs1.end(); ++node1)
        {
          boost::unordered_map<uint, uint>::const_iterator
                _node1 = my_old_to_new_dof_index.find(*node1);
          if (_node1 != my_old_to_new_dof_index.end())
          {
            const uint local_node1 = _node1->second;
            graph[local_node0].insert(local_node1);
          }
        }
      }
    }
  }

  // Reorder dofs locally
  const std::vector<std::size_t> dof_remap
      = BoostGraphOrdering::compute_cuthill_mckee(graph, true);

  // Map from old to new index for dofs
  boost::unordered_map<uint, uint> old_to_new_dof_index;

  // Renumber owned dofs and buffer dofs that are owned but shared with
  // another process
  uint counter = 0;
  std::vector<uint> send_buffer;
  for (set_iterator owned_dof = owned_dofs.begin(); owned_dof != owned_dofs.end(); ++owned_dof, counter++)
  {
    // Set new dof number
    old_to_new_dof_index[*owned_dof] = process_offset + dof_remap[counter];

    // Update UFC-to-renumbered map for new number
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
  dofmap._ownership_range = std::make_pair<uint, uint>(process_offset,
                                          process_offset + owned_dofs.size());

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
typedef std::pair<uint, uint> dof_pair;
typedef std::map<uint, uint> uint_map_type;
typedef std::map<uint, dof_pair> dof_pair_map;
typedef dof_pair_map::iterator dof_pair_map_iterator;

void DofMapBuilder::extract_dof_pairs(const DofMap& dofmap, const Mesh& mesh, 
                                      dof_pair_map& _slave_master_map,
                                      std::pair<uint, uint> ownership_range)
{
  Timer t0("Extracting dof pairs");
  const uint num_sub_dofmaps = dofmap._ufc_dofmap->num_sub_dofmaps();
  if (num_sub_dofmaps > 0)
  {
    // Call recursively for all sub_dofmaps
    std::vector<uint> component(1);
    for (uint i=0; i<num_sub_dofmaps; i++)
    {
      component[0] = i;
      DofMap* sub_dofmap = dofmap.extract_sub_dofmap(component, mesh);
      extract_dof_pairs(*sub_dofmap, mesh, _slave_master_map, ownership_range);
    }
  }
  else
  {    
    // Get facet-to-facet map from mesh class
    std::vector<std::pair< std::pair<int, int>, std::pair<int, int> > > facet_pairs = mesh.facet_pairs;
    const uint num_periodic_faces = facet_pairs.size();
    
    // Get dimensions
    const uint tdim = mesh.topology().dim();
    const uint gdim = mesh.geometry().dim();
    
    // Arrays used for mapping coordinates
    std::vector<double> x(gdim);
    std::vector<double> y(gdim);
    std::vector<double> dx(gdim);    
    
    // Declare some variables used to map dofs
    std::vector<uint> local_master_dofs(dofmap.num_facet_dofs());
    std::vector<uint> local_slave_dofs(dofmap.num_facet_dofs());
    typedef boost::multi_array<double, 2> coors;
    coors master_coor(boost::extents[dofmap.max_cell_dimension()][gdim]);
    coors slave_coor(boost::extents[dofmap.max_cell_dimension()][gdim]);      
        
    std::set<uint> communicating_processors;
    std::map<uint, std::vector<double> > x_map;
    std::map<uint, std::vector<double> > received_y;
    std::map<uint, std::vector<uint> > glob_map;
    std::map<uint, std::vector<uint> > slave_map;
    std::vector<uint> glob_vec(1);
    std::vector<uint> global_master_dofs;
    std::vector<uint> global_slave_dofs;

    const uint process_number = MPI::process_number();
        
    // Vectors to hold global dofs on the master/slave facets
    std::vector<uint> global_master_dofs_on_facet;
    std::vector<uint> global_slave_dofs_on_facet;
    
    // Loop over periodic face maps and extract dof pairs
    for (uint i=0; i<num_periodic_faces; i++)
    {   
      // There are two connected periodic facets that could live on different processes
      const uint master_process = facet_pairs[i].first.second;
      const uint slave_process = facet_pairs[i].second.second;
      bool parallel_process = (master_process != slave_process);
      
      global_master_dofs_on_facet.clear();
      global_slave_dofs_on_facet.clear();
      global_master_dofs.clear();
      global_slave_dofs.clear();
      received_y.clear();
      x_map.clear();      
      communicating_processors.clear();
      
      if (master_process == process_number)
      {
        // Get info from master facet: cell, dofs, coordinates
        const Facet master_facet(mesh, facet_pairs[i].first.first);
        const Cell master_cell(mesh, master_facet.entities(tdim)[0]);        
        global_master_dofs = dofmap.cell_dofs(master_cell.index());
        dofmap.tabulate_coordinates(master_coor, master_cell);
        const uint local_master_facet = master_cell.index(master_facet);
        dofmap.tabulate_facet_dofs(&local_master_dofs[0], local_master_facet); 
        const Point midpoint = master_facet.midpoint();   
        for (uint i=0; i<gdim; i++)
          x[i] = midpoint[i];
        communicating_processors.insert(slave_process);
      }
      if (slave_process == process_number)
      {
        // Get info from slave facet: cell, dofs, coordinates
        const Facet slave_facet(mesh, facet_pairs[i].second.first);      
        const Cell slave_cell(mesh, slave_facet.entities(tdim)[0]);
        global_slave_dofs = dofmap.cell_dofs(slave_cell.index());
        dofmap.tabulate_coordinates(slave_coor, slave_cell);
        const uint local_slave_facet = slave_cell.index(slave_facet);
        dofmap.tabulate_facet_dofs(&local_slave_dofs[0], local_slave_facet); 
        const Point midpoint = slave_facet.midpoint();
        for (uint i=0; i<gdim; i++)
          y[i] = midpoint[i];
        communicating_processors.insert(master_process);
        x_map[master_process] = y;
      }
      
      // Send slave y to master if parallel
      if (parallel_process)
      {
        MPI::distribute(communicating_processors, x_map, received_y);
        if (process_number == master_process)
        {
          y.assign(received_y[slave_process].begin(), received_y[slave_process].end());
        }
      }
      communicating_processors.clear();
      received_y.clear();
      x_map.clear();
      glob_map.clear();
      slave_map.clear();
      
      // The distance dx is used to match master and slave dofs
      if (master_process == process_number)
      {
        for (uint j=0; j<gdim; j++)
          dx[j] = y[j] - x[j];
      }

      // Match master and slave dofs and put pair in map
      for (uint j=0; j<dofmap.num_facet_dofs(); j++)
      {
        uint global_master_dof;
        uint global_slave_dof;
        if (master_process == process_number)
        {
          // Get global master dof and coordinates
          global_master_dof = global_master_dofs[local_master_dofs[j]];
          std::copy(master_coor[local_master_dofs[j]].begin(),
                    master_coor[local_master_dofs[j]].end(), x.begin());
        
          communicating_processors.insert(slave_process);            
        } 
        bool found_match;
        for (uint k=0; k<dofmap.num_facet_dofs(); k++)
        {
          found_match = false;
          if (slave_process == process_number)
          {
            // Get global slave dof and coordinates
            global_slave_dof = global_slave_dofs[local_slave_dofs[k]];
            std::copy(slave_coor[local_slave_dofs[k]].begin(),
                      slave_coor[local_slave_dofs[k]].end(), y.begin());
            communicating_processors.insert(master_process);
            x_map[master_process] = y;
            glob_vec[0] = global_slave_dof;
            glob_map[master_process] = glob_vec;
          }
          // Exchange info with master_process
          if (parallel_process)
          {
            MPI::distribute(communicating_processors, x_map, received_y);
            MPI::distribute(communicating_processors, glob_map, slave_map);
            if (process_number == master_process)
            {
              y.assign(received_y[slave_process].begin(), received_y[slave_process].end());
              global_slave_dof = slave_map[slave_process][0];
            }
          }
          
          // Look for a match in coordinates on the master process
          if (master_process == process_number)
          {              
            for(uint l=0; l<gdim; l++) 
              y[l] = y[l] - dx[l];   
            
            // y should now be equal to x for a periodic match
            double error = 0.;
            for(uint l=0; l<gdim; l++) 
              error += pow(y[l] - x[l], 2);
            
            if (error < 1e-12) // Match! Store master and slave in vectors
            {  
              global_master_dofs_on_facet.push_back(global_master_dof);
              global_slave_dofs_on_facet.push_back(global_slave_dof);
              found_match = true;
            }
          }
          MPI::broadcast(found_match, master_process);
          if (found_match) break;
        }        
      }  // Finished on facet. Move to next periodic facet pair
            
      // At this point there should be a match between dofs on master facet
      // Put this information on all processes
      MPI::broadcast(global_master_dofs_on_facet, master_process);
      MPI::broadcast(global_slave_dofs_on_facet, master_process);
      
      // Add dof pair on facet to the global _slave_master_map
      for (uint j=0; j<global_master_dofs_on_facet.size(); j++)
      {
        uint master_dof = global_master_dofs_on_facet[j]; 
        uint slave_dof  = global_slave_dofs_on_facet[j];
        uint dof_owner = 0;
        if (master_dof >= ownership_range.first && master_dof < ownership_range.second)
          dof_owner = process_number;        
        dof_pair pair(master_dof, MPI::max(dof_owner));
        
        // At this point we need to do something clever in case of more than one periodic direction
        // For example, a rectangle mesh with two periodic directions will have four corners that 
        // should be equal. In that case we will here end up with one master and three slaves of the 
        // same master. A 3D Cube with 8 corners should similarily have 7 slaves of the same master. 
        
        // If the slave does not exist, then add to slave_master map. But check also
        // if the slave has been used as a master before.
        if (_slave_master_map.find(slave_dof) == _slave_master_map.end())
        {
          _slave_master_map[slave_dof] = pair;
          for (dof_pair_map_iterator it = _slave_master_map.begin();
                                     it != _slave_master_map.end(); ++it)
          {
            if (it->second.first == slave_dof) // Has been previously used as master
            {
              _slave_master_map[it->first] = pair; // Use latest master value for previous as well
              break;
            }
          }
        }
        else // The slave_dof exists as slave from before
        {
          // Check if the master_dof has been used previously as a slave
          for (dof_pair_map_iterator it = _slave_master_map.begin();
                                     it != _slave_master_map.end(); ++it)
          {
            if (it->first == master_dof)
            {
              _slave_master_map[slave_dof] = it->second; // In that case use previous master for the current slave as well
              break;
            }
          }
        }
      }
    }    
  }
}

void DofMapBuilder::periodic_modification(DofMap& dofmap, const Mesh& mesh)
{
  Timer t0("Periodic dofmap modification");
  // Periodic modification. Replace all slaves with masters and then delete slave dofs
  typedef std::map<uint, std::pair<uint, uint> > dof_pairs;
  typedef dof_pairs::iterator dof_pairs_iterator;
  dof_pairs _slave_master_map;
  
  // Extracting dof pairs for dofmap
  extract_dof_pairs(dofmap, mesh, _slave_master_map, dofmap._ownership_range);
  
  // Modify the dofmap by putting the master in all locations where a slave is found
  for (uint i=0; i<dofmap._dofmap.size(); i++)
  {
    const std::vector<uint>& global_dofs = dofmap.cell_dofs(i); 
    for (uint j=0; j<dofmap.max_cell_dimension(); j++)
    {
      const uint dof = global_dofs[j];
      for (dof_pairs_iterator it = _slave_master_map.begin();
                            it != _slave_master_map.end(); ++it)
      {
        if (dof == it->first) // it->first is slave dof
        {
          dofmap._dofmap[i][j] = it->second.first;
          // Put master in _off_process_owner
          if (it->second.second != MPI::process_number())
            dofmap._off_process_owner[it->second.first] = it->second.second; 
          // Remove slave from _off_process_owner
          if (dofmap._off_process_owner.find(it->first) != dofmap._off_process_owner.end())
            dofmap._off_process_owner.erase(it->first);
          for (boost::unordered_map<uint, uint>::iterator op = dofmap.ufc_map_to_dofmap.begin();
            op != dofmap.ufc_map_to_dofmap.end(); ++op)
          {
            if (op->second == dof)
              dofmap.ufc_map_to_dofmap[op->first] = it->second.first;
          }
          break;
        }
      }
    }
  }
  
  // Eliminate the slaves entirely
  std::vector<uint> _slaves_on_process;
  std::vector<uint> _all_slaves;
  for (dof_pairs_iterator it = _slave_master_map.begin();
                          it != _slave_master_map.end(); ++it)
  {
    if (it->first >= dofmap._ownership_range.first && it->first < dofmap._ownership_range.second)
      _slaves_on_process.push_back(it->first);
    
    _all_slaves.push_back(it->first);
  }
  
  uint slaves_on_this_process = _slaves_on_process.size();
  std::vector<uint> slaves_on_all_processes(MPI::num_processes()); 
  MPI::all_gather(slaves_on_this_process, slaves_on_all_processes);
  uint accumulated_slaves = 0;
  for (uint i=0; i<MPI::process_number(); i++)
    accumulated_slaves += slaves_on_all_processes[i];
  
  // Modify ownership_range due to deleted slave dofs. Only for parent dofmap.
  dofmap._ownership_range.first -= accumulated_slaves;
  dofmap._ownership_range.second -= (accumulated_slaves + slaves_on_this_process);
  
  // Renumber all dofs due to deleted slave dofs
  std::vector<uint>::iterator it;
  std::sort(_all_slaves.begin(), _all_slaves.end());
  dofmap._num_slaves = _all_slaves.size();    
  
  for (uint i=0; i<dofmap._dofmap.size(); i++)
  {
    const std::vector<uint>& global_dofs = dofmap.cell_dofs(i); 
    for (uint j=0; j<dofmap.max_cell_dimension(); j++)
    {
      const uint dof = global_dofs[j];
      it = std::lower_bound(_all_slaves.begin(), _all_slaves.end(), dof);
      uint new_dof = dof - uint(it - _all_slaves.begin());
      dofmap._dofmap[i][j] = new_dof;
    }
  }
  
  // Modify _off_process_owner due to deleted slave dofs
  boost::unordered_map<uint, uint> new_off_process_owner;
  for (boost::unordered_map<uint, uint>::iterator op_dof = dofmap._off_process_owner.begin();
      op_dof != dofmap._off_process_owner.end(); ++op_dof)
  {
    uint old_dof = op_dof->first;
    it = std::lower_bound(_all_slaves.begin(), _all_slaves.end(), old_dof);
    uint new_dof = old_dof - uint(it - _all_slaves.begin());
    new_off_process_owner.insert(std::make_pair<uint, uint> (new_dof, op_dof->second));
  }
  dofmap._off_process_owner = new_off_process_owner;    
  
  // Modify ufc_map_to_dofmap due to deleted slave dofs
  boost::unordered_map<uint, uint> new_ufc_map_to_dofmap;
  for (boost::unordered_map<uint, uint>::iterator op = dofmap.ufc_map_to_dofmap.begin();
      op != dofmap.ufc_map_to_dofmap.end(); ++op)
  {
    uint old_dof = op->second;
    it = std::lower_bound(_all_slaves.begin(), _all_slaves.end(), old_dof);
    uint new_dof = old_dof - uint(it - _all_slaves.begin());
    new_ufc_map_to_dofmap.insert(std::make_pair<uint, uint> (op->first, new_dof));
  }
  dofmap.ufc_map_to_dofmap = new_ufc_map_to_dofmap;  
  dofmap._global_dim = dofmap._ufc_dofmap->global_dimension() - dofmap._num_slaves;
}
