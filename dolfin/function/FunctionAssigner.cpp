// Copyright (C) 2013 Johan Hake
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
// First added:  2013-09-20
// Last changed: 2013-09-24

#include <utility>

#include <dolfin/common/types.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/FunctionSpace.h>

#include "FunctionAssigner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionAssigner::FunctionAssigner(boost::shared_ptr<const FunctionSpace> assigning_space, 
				   boost::shared_ptr<const FunctionSpace> receiving_space) :
  _assigning_spaces(1, assigning_space), _receiving_spaces(1, receiving_space),
  _assigning_indices(0), _receiving_indices(0), _transfer(0)
{
  _check_compatability();
  _init_indices();
}
//-----------------------------------------------------------------------------
FunctionAssigner::FunctionAssigner(boost::shared_ptr<const FunctionSpace> assigning_space, 
	           std::vector<boost::shared_ptr<const FunctionSpace> > receiving_spaces) :
  _assigning_spaces(1, assigning_space), _receiving_spaces(receiving_spaces),
  _assigning_indices(0), _receiving_indices(0), _transfer(0)
{
  _check_compatability();
  _init_indices();
}
//-----------------------------------------------------------------------------
FunctionAssigner::FunctionAssigner(std::vector<boost::shared_ptr<const FunctionSpace> > assigning_spaces, 
				   boost::shared_ptr<const FunctionSpace> receiving_space) :
  _assigning_spaces(assigning_spaces), _receiving_spaces(1, receiving_space),
  _assigning_indices(0), _receiving_indices(0), _transfer(0)
{
  _check_compatability();
  _init_indices();
}
//-----------------------------------------------------------------------------
FunctionAssigner::~FunctionAssigner()
{
  
}
//-----------------------------------------------------------------------------
void FunctionAssigner::assign(boost::shared_ptr<const Function> assigning_func, 
			      boost::shared_ptr<Function> receiving_func) const
{
  
}
//-----------------------------------------------------------------------------
void FunctionAssigner::assign(std::vector<boost::shared_ptr<const Function> > assigning_funcs, 
			      boost::shared_ptr<Function> receiving_func) const
{
}
//-----------------------------------------------------------------------------
void FunctionAssigner::assign(boost::shared_ptr<const Function> assigning_funcs, 
			      std::vector<boost::shared_ptr<Function> > receiving_func) const
{
}
//-----------------------------------------------------------------------------
void FunctionAssigner::_check_compatability()
{

  // Check for empty space vectors
  if (_assigning_spaces.size()==0)
    dolfin_error("FunctionAssigner.cpp",
                 "create function assigner",
                 "Expected at least one FunctionSpace for the assigning spaces");
    
  if (_receiving_spaces.size()==0)
    dolfin_error("FunctionAssigner.cpp",
                 "create function assigner",
                 "Expected at least one FunctionSpace for the receiving spaces");

  // Get mesh
  const Mesh& mesh = *_assigning_spaces[0]->mesh();

  // Check function spaces uses the same mesh.
  for (std::size_t i=0; i<num_assigning_spaces(); i++)

    // Compare pointers
    if (&mesh != _assigning_spaces[i]->mesh().get())
      dolfin_error("FunctionAssigner.cpp",
		   "create function assigner",
		   "Expected all FunctionSpaces to bed defined over the same Mesh");
      
  for (std::size_t i=0; i<num_receiving_spaces(); i++)

    // Compare pointers
    if (&mesh != _receiving_spaces[i]->mesh().get())
      dolfin_error("FunctionAssigner.cpp",
		   "create function assigner",
		   "Expected all FunctionSpaces to bed defined over the same Mesh");
  

  // If we have an 1 -> N assignment
  if (num_assigning_spaces() < num_receiving_spaces())
  {
    if (num_assigning_spaces()!=1)
      dolfin_error("FunctionAssigner.cpp",
		   "create function assigner",
		   "Expected only 1 assigning FunctionSpace");
    
    // Check that the number of assigning subspaces are the same as number 
    // of receiving spaces
    const std::size_t N = num_receiving_spaces();
    if (_assigning_spaces[0]->element()->num_sub_elements()==N)
      dolfin_error("FunctionAssigner.cpp",
		   "create function assigner",
		   "Expected the same number of sub spaces in the assigning "\
		   "FunctionSpace as the number of receiving FunctionSpaces");

    // Check that the number of entity dofs corresponds
    for (std::size_t entity_dim=0; entity_dim < mesh.topology().dim(); entity_dim++)
    {
      // Collect num receiving entity dofs
      const std::size_t num_receiving_entity_dofs = _receiving_spaces[0]->dofmap()->
	num_entity_dofs(entity_dim);
      
      // Check num entity dofs for the rest of the receiving FunctionSpaces
      for (std::size_t receiving_ind=1; receiving_ind < num_receiving_spaces(); 
	   receiving_ind++)
      {
	if (_receiving_spaces[receiving_ind]->dofmap()->num_entity_dofs(entity_dim) != \
	    num_receiving_entity_dofs)
	  dolfin_error("FunctionAssigner.cpp",
		       "create function assigner",
		       "The receiving FunctionSpaces have incompatible number of "\
		       "entity dofs for entity %d", entity_dim);
      }

      // Check num entity dofs for the assigning subspaces
      for (std::size_t sub_space_ind=0; sub_space_ind<N; sub_space_ind++)
      {
	// Get subspace
	const FunctionSpace& sub_space = *(*_assigning_spaces[0])[sub_space_ind];
	
	// Check num entity dofs
	if (sub_space.dofmap()->num_entity_dofs(entity_dim)==num_receiving_entity_dofs)
	  dolfin_error("FunctionAssigner.cpp",
		       "create function assigner",
		       "A subspace of the assigning FunctionSpace has incompatible "\
		       "number of entity dofs for entity %d", entity_dim);
      }

    }

  }
  
  // If we have an N -> 1 assignment
  else if (num_assigning_spaces() > num_receiving_spaces())
  {
    if (num_receiving_spaces()!=1)
      dolfin_error("FunctionAssigner.cpp",
		   "create function assigner",
		   "Expected only 1 receiving FunctionSpace");
    
    // Check that the number of receiving subspaces are the same as number 
    // of assigning spaces
    const std::size_t N = num_assigning_spaces();
    if (_receiving_spaces[0]->element()->num_sub_elements()==N)
      dolfin_error("FunctionAssigner.cpp",
		   "create function assigner",
		   "Expected the same number of sub spaces in the receiving "\
		   "FunctionSpace as the number of assigning FunctionSpaces");

    // Check that the number of entity dofs corresponds
    for (std::size_t entity_dim=0; entity_dim < mesh.topology().dim(); entity_dim++)
    {
      // Collect num assigning entity dofs
      const std::size_t num_assigning_entity_dofs = _assigning_spaces[0]->dofmap()->
	num_entity_dofs(entity_dim);
      
      // Check num entity dofs for the rest of the receiving FunctionSpaces
      for (std::size_t assigning_ind=1; assigning_ind < num_assigning_spaces(); 
	   assigning_ind++)
      {
	if (_assigning_spaces[assigning_ind]->dofmap()->num_entity_dofs(entity_dim) != \
	    num_assigning_entity_dofs)
	  dolfin_error("FunctionAssigner.cpp",
		       "create function assigner",
		       "The assigning FunctionSpaces have incompatible number of "\
		       "entity dofs for entity %d", entity_dim);
      }

      // Check num entity dofs for the receiving subspaces
      for (std::size_t sub_space_ind=0; sub_space_ind<N; sub_space_ind++)
      {
	// Get subspace
	const FunctionSpace& sub_space = *(*_receiving_spaces[0])[sub_space_ind];
	
	// Check num entity dofs
	if (sub_space.dofmap()->num_entity_dofs(entity_dim)==num_assigning_entity_dofs)
	  dolfin_error("FunctionAssigner.cpp",
		       "create function assigner",
		       "A subspace of the receiving FunctionSpace has incompatible "\
		       "number of entity dofs for entity %d", entity_dim);
      }
    }
  }
  
  // If we have an 1 -> 1 assignment
  else 
  {
    if (num_receiving_spaces()!=1 || num_assigning_spaces()!=1)
      dolfin_error("FunctionAssigner.cpp",
		   "create function assigner",
		   "Expected only 1 receiving and 1 assigning FunctionSpace");
    
    for (std::size_t entity_dim=0; entity_dim < mesh.topology().dim(); entity_dim++)
    {
      
      // Collect num assigning entity dofs
      const std::size_t num_assigning_entity_dofs = _assigning_spaces[0]->dofmap()->
	num_entity_dofs(entity_dim);
      
      const std::size_t num_receiving_entity_dofs = _receiving_spaces[0]->dofmap()->
	num_entity_dofs(entity_dim);
      
      if (num_receiving_entity_dofs != num_assigning_entity_dofs)
	dolfin_error("FunctionAssigner.cpp",
		     "create function assigner",
		     "The assigning and receiving FunctionSpaces have "	\
		     "incompatible number of entity dofs for entity %d", entity_dim);
    }
  }

}
//-----------------------------------------------------------------------------
void FunctionAssigner::_init_indices()
{

  std::vector<dolfin::la_index>::const_iterator index_it;

  // A vector of sets for collecting the assigning and receiving dofs
  std::vector<std::set<dolfin::la_index> > assigning_dofs;
  std::vector<std::set<dolfin::la_index> > receiving_dofs;
  
  // Get mesh
  const Mesh& mesh = *_assigning_spaces[0]->mesh();

  // Build indices for 1-1 assignment
  if (num_receiving_spaces()==1 && num_assigning_spaces()==1)
  {
    
    // Get dof maps
    const GenericDofMap& assigning_dofmap = *_assigning_spaces[0]->dofmap();
    const GenericDofMap& receiving_dofmap = *_receiving_spaces[0]->dofmap();
    
    std::set<dolfin::la_index> assigning_dofs;
    std::set<dolfin::la_index> receiving_dofs;
    
    const std::size_t assigning_n0 = assigning_dofmap.ownership_range().first;
    const std::size_t assigning_n1 = assigning_dofmap.ownership_range().second;
    const std::size_t receiving_n0 = receiving_dofmap.ownership_range().first;
    const std::size_t receiving_n1 = receiving_dofmap.ownership_range().second;

    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      
      const std::vector<dolfin::la_index>& assigning_cell_dofs = \
	assigning_dofmap.cell_dofs(cell->index());
      const std::vector<dolfin::la_index>& receiving_cell_dofs = \
	receiving_dofmap.cell_dofs(cell->index());
      
      dolfin_assert(assigning_cell_dofs.size()==receiving_cell_dofs.size());

      // Iterate over the local dofs and collect on process dofs
      for (std::size_t i=0; i<assigning_cell_dofs.size(); i++)
      {
	const std::size_t assigning_dof = assigning_cell_dofs[i];
	if (assigning_dof >= assigning_n0 && assigning_dof < assigning_n1)
	  assigning_dofs.insert(assigning_dof);

	const std::size_t receiving_dof = receiving_cell_dofs[i];
	if (receiving_dof >= receiving_n0 && receiving_dof < receiving_n1)
	  receiving_dofs.insert(receiving_dof);
	
      }

    }

  }
  
}
//-----------------------------------------------------------------------------
