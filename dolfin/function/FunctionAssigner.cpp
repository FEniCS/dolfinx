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
// Last changed: 2014-02-28

#include <map>
#include <utility>
#include <dolfin/common/types.h>
#include <dolfin/fem/GenericDofMap.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include "FunctionAssigner.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
FunctionAssigner::FunctionAssigner(
 std::shared_ptr<const FunctionSpace> receiving_space,
 std::shared_ptr<const FunctionSpace> assigning_space)
  : _receiving_spaces(1, receiving_space),_assigning_spaces(1, assigning_space),
    _receiving_indices(1), _assigning_indices(1), _transfer(1)
{
  // Get mesh
  const Mesh& mesh = _get_mesh();

  // Build vectors of indices
  _check_and_build_indices(mesh, _receiving_spaces, _assigning_spaces);
}
//-----------------------------------------------------------------------------
FunctionAssigner::FunctionAssigner(
  std::vector<std::shared_ptr<const FunctionSpace>> receiving_spaces,
  std::shared_ptr<const FunctionSpace> assigning_space)
  : _receiving_spaces(receiving_spaces), _assigning_spaces(1, assigning_space),
    _receiving_indices(receiving_spaces.size()),
    _assigning_indices(receiving_spaces.size()),
    _transfer(receiving_spaces.size())
{
  // Get mesh
  const Mesh& mesh = _get_mesh();

  // Check that the number of assigning subspaces are the same as number
  // of receiving spaces
  const std::size_t N = _receiving_spaces.size();
  dolfin_assert(_assigning_spaces[0]);
  if (_assigning_spaces[0]->element()->num_sub_elements() != N)
  {
    dolfin_error("FunctionAssigner.cpp",
                 "create function assigner",
                 "Expected the same number of sub spaces in the assigning "
                 "FunctionSpace as the number of receiving FunctionSpaces");
  }

  // Collect assigning sub spaces
  std::vector<std::shared_ptr<const FunctionSpace>> assigning_sub_spaces;
  for (std::size_t sub_space_ind = 0; sub_space_ind < N; sub_space_ind++)
    assigning_sub_spaces.push_back((*_assigning_spaces[0])[sub_space_ind]);

  // Build vectors of indices
  _check_and_build_indices(mesh, _receiving_spaces, assigning_sub_spaces);
}
//-----------------------------------------------------------------------------
FunctionAssigner::FunctionAssigner(
  std::shared_ptr<const FunctionSpace> receiving_space,
  std::vector<std::shared_ptr<const FunctionSpace>> assigning_spaces)
  :_receiving_spaces(1, receiving_space), _assigning_spaces(assigning_spaces),
   _receiving_indices(assigning_spaces.size()),
   _assigning_indices(assigning_spaces.size()),
   _transfer(assigning_spaces.size())
{
  // Get mesh
  const Mesh& mesh = _get_mesh();

  // Check that the number of receiving subspaces are the same as number
  // of assigning spaces
  const std::size_t N = assigning_spaces.size();
  dolfin_assert(_receiving_spaces[0]);
  if (_receiving_spaces[0]->element()->num_sub_elements()!=N)
  {
    dolfin_error("FunctionAssigner.cpp",
                 "create function assigner",
                 "Expected the same number of sub spaces in the receiving "
                 "FunctionSpace as the number of assigning FunctionSpaces");
  }

  // Collect receiving sub spaces
  std::vector<std::shared_ptr<const FunctionSpace>> receiving_sub_spaces;
  for (std::size_t sub_space_ind = 0; sub_space_ind < N; sub_space_ind++)
    receiving_sub_spaces.push_back((*_receiving_spaces[0])[sub_space_ind]);

  // Build vectors of indices
  _check_and_build_indices(mesh, receiving_sub_spaces, _assigning_spaces);
}
//-----------------------------------------------------------------------------
FunctionAssigner::~FunctionAssigner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void
FunctionAssigner::assign(std::shared_ptr<Function> receiving_func,
                         std::shared_ptr<const Function> assigning_func) const
{
  // Wrap functions
  std::vector<std::shared_ptr<Function>> receiving_funcs(1, receiving_func);
  std::vector<std::shared_ptr<const Function>>
    assigning_funcs(1, assigning_func);

  // Do the assignment
  _assign(receiving_funcs, assigning_funcs);
}
//-----------------------------------------------------------------------------
void FunctionAssigner::assign(
  std::shared_ptr<Function> receiving_func,
  std::vector<std::shared_ptr<const Function>> assigning_funcs) const
{
  // Num assigning functions
  const std::size_t N = assigning_funcs.size();
  if (receiving_func->function_space()->element()->num_sub_elements() != N)
  {
    dolfin_error("FunctionAssigner.cpp",
                 "assigning functions",
                 "Expected the same number of sub functions in the receiving "
                 "Function as the number of assigning Functions");
  }

  // Collect receiving sub functions
  std::vector<std::shared_ptr<Function>> receiving_funcs(0);
  for (std::size_t i = 0; i < N; i++)
  {
    std::shared_ptr<Function>
      func(reference_to_no_delete_pointer((*receiving_func)[i]));
    receiving_funcs.push_back(func);
  }

  // Do the assignment
  _assign(receiving_funcs, assigning_funcs);
}
//-----------------------------------------------------------------------------
void
FunctionAssigner::assign(std::vector<std::shared_ptr<Function>> receiving_funcs,
                         std::shared_ptr<const Function> assigning_func) const
{
  // Num receiving functions
  const std::size_t N = receiving_funcs.size();
  if (assigning_func->function_space()->element()->num_sub_elements() != N)
  {
    dolfin_error("FunctionAssigner.cpp",
                 "assigning functions",
                 "Expected the same number of sub functions in the assigning "
                 "Function as the number of receiving Functions");
  }

  // Collect receiving sub functions
  std::vector<std::shared_ptr<const Function>> assigning_funcs(0);
  for (std::size_t i = 0; i < N; i++)
  {
    std::shared_ptr<const Function>
      func(reference_to_no_delete_pointer((*assigning_func)[i]));
    assigning_funcs.push_back(func);
  }

  // Do the assignment
  _assign(receiving_funcs, assigning_funcs);
}
//-----------------------------------------------------------------------------
void FunctionAssigner::_assign(
  std::vector<std::shared_ptr<Function>> receiving_funcs,
  std::vector<std::shared_ptr<const Function>> assigning_funcs) const
{
  // Num spaces
  const std::size_t N = std::max(_assigning_spaces.size(),
                                 _receiving_spaces.size());

  if (assigning_funcs.size() != N)
  {
    dolfin_error("FunctionAssigner.cpp",
                 "assign functions",
                 "Expected the same number of assigning (sub)functions as "
                 "the number of assigning (sub)spaces.");
  }

  if (receiving_funcs.size() != N)
  {
    dolfin_error("FunctionAssigner.cpp",
                 "assign functions",
                 "Expected the same number of receiving (sub)functions as "
                 "the number of receiving (sub)spaces.");
  }

  // Flag to determine if the receiving vector is the same
  bool same_receiving_vector = true;
  const GenericVector* receiving_vector = receiving_funcs[0]->_vector.get();

  // Iterate over the spaces and do the assignments
  for (std::size_t i = 0; i < N; i++)
  {
    // Check that the functions are in the FunctionAssigner spaces
    if (_receiving_spaces.size() == 1)
    {
      dolfin_assert(_receiving_spaces[0]);

      // Check 1-1 assignment
      if (_assigning_spaces.size() == 1)
      {
        dolfin_assert(receiving_funcs[0]);
        if (!receiving_funcs[0]->in(*_receiving_spaces[0]))
        {
          dolfin_error("FunctionAssigner.cpp",
                       "assign functions",
                       "The receiving Function is not in the receiving "
                       "FunctionSpaces");
        }
      }

      // Check N-1 assignment
      else
      {
        dolfin_assert(receiving_funcs[i]);
        dolfin_assert((*_receiving_spaces[0])[i]);
        if (!receiving_funcs[i]->in(*(*_receiving_spaces[0])[i]))
        {
          dolfin_error("FunctionAssigner.cpp",
                       "assign functions",
                       "The receiving sub Functions are not in the receiving "
                       "sub FunctionSpaces");
        }
      }
    }
    else
    {
      // Check 1-N assignment
      if (!receiving_funcs[i]->in(*_receiving_spaces[i]))
      {
        dolfin_error("FunctionAssigner.cpp",
                     "assign functions",
                     "The receiving Functions are not in the receiving "
                     "FunctionSpaces");
      }
    }

    if (_assigning_spaces.size() == 1)
    {
      dolfin_assert(_assigning_spaces[0]);

      // Check 1-1 assignment
      if (_receiving_spaces.size() == 1)
      {
        dolfin_assert(assigning_funcs[0]);
        if (!assigning_funcs[0]->in(*_assigning_spaces[0]))
        {
          dolfin_error("FunctionAssigner.cpp",
                       "assign functions",
                       "The assigning Function is not in the assigning "
                       "FunctionSpaces");
        }
      }

      // Check 1-N assignment
      else
      {
        dolfin_assert(assigning_funcs[i]);
        dolfin_assert((*_assigning_spaces[0])[i]);
        if (!assigning_funcs[i]->in(*(*_assigning_spaces[0])[i]))
        {
          dolfin_error("FunctionAssigner.cpp",
                       "assign functions",
                       "The assigning sub Functions are not in the assigning "
                       "sub FunctionSpaces");
        }
      }
    }
    else
    {
      dolfin_assert(assigning_funcs[i]);
      dolfin_assert(_assigning_spaces[i]);

      // Check N-1 assignment
      if (!assigning_funcs[i]->in(*_assigning_spaces[i]))
      {
        dolfin_error("FunctionAssigner.cpp",
                     "assign function",
                     "The assigning Functions are not in the assigning "
                     "FunctionSpaces");
      }
    }

    // Check if the receiving vector is the same
    if (i != 0)
    {
      dolfin_assert(receiving_funcs[i]->_vector);
      same_receiving_vector = same_receiving_vector
        && (receiving_vector == receiving_funcs[i]->_vector.get());
    }

    // Get assigning values
    assigning_funcs[i]->_vector->get_local(&_transfer[i][0],
                                           _transfer[i].size(),
             &_assigning_indices[i][0]);

    // Set receiving values
    receiving_funcs[i]->_vector->set_local(&_transfer[i][0],
                                           _transfer[i].size(),
                                           &_receiving_indices[i][0]);

  }

  // Apply to common same vector or all of them
  if (same_receiving_vector)
  {
    receiving_funcs[0]->_vector->apply("insert");
  }
  else
  {
    for (std::size_t i = 0; i < N; i++)
      receiving_funcs[i]->_vector->apply("insert");
  }
}
//-----------------------------------------------------------------------------
const Mesh& FunctionAssigner::_get_mesh() const
{
  // Check for empty space vectors
  if (_assigning_spaces.size() == 0)
  {
    dolfin_error("FunctionAssigner.cpp",
                 "create function assigner",
                 "Expected at least one FunctionSpace "
                 "for the assigning spaces");
  }

  if (_receiving_spaces.size()==0)
  {
    dolfin_error("FunctionAssigner.cpp",
                 "create function assigner",
                 "Expected at least one FunctionSpace "
                 "for the receiving spaces");
  }

  // Get mesh
  dolfin_assert(_assigning_spaces[0]);
  dolfin_assert(_assigning_spaces[0]->mesh());
  const Mesh& mesh = *_assigning_spaces[0]->mesh();

  // Check that function spaces uses the same mesh.
  for (std::size_t i = 1; i < _assigning_spaces.size(); i++)
  {
    // Compare pointers
    dolfin_assert(_assigning_spaces[i]);
    dolfin_assert(_assigning_spaces[i]->mesh());
    if (&mesh != _assigning_spaces[i]->mesh().get())
    {
      dolfin_error("FunctionAssigner.cpp",
                   "create function assigner",
                   "Expected all FunctionSpaces to be defined "
                   "over the same Mesh");
    }
  }

  for (std::size_t i = 0; i < _receiving_spaces.size(); i++)
  {
    // Compare pointers
    dolfin_assert(_receiving_spaces[i]);
    dolfin_assert(_receiving_spaces[i]->mesh());
    if (&mesh != _receiving_spaces[i]->mesh().get())
    {
      dolfin_error("FunctionAssigner.cpp",
                   "create function assigner",
                   "Expected all FunctionSpaces to be defined "
                   "over the same Mesh");
    }
  }

  // Return checked mesh
  return mesh;
}
//-----------------------------------------------------------------------------
void FunctionAssigner::_check_and_build_indices(
  const Mesh& mesh,
  const std::vector<std::shared_ptr<const FunctionSpace>>& receiving_spaces,
  const std::vector<std::shared_ptr<const FunctionSpace>>& assigning_spaces)
{

  // Num spaces
  const std::size_t N = receiving_spaces.size();

  // Check num entity dofs for the receiving and assigning spaces
  // corresponds
  for (std::size_t i = 0; i < N; i++)
  {
    // Iterate over all entity dimensions
    for (std::size_t entity_dim=0; entity_dim < mesh.topology().dim();
         entity_dim++)
    {
      // Check num entity dofs for assigning spaces
      if (assigning_spaces[i]->dofmap()->num_entity_dofs(entity_dim)
          != receiving_spaces[i]->dofmap()->num_entity_dofs(entity_dim))
      {
        dolfin_error("FunctionAssigner.cpp",
                     "create function assigner",
                     "The assigning and receiving FunctionSpaces have "
                     "incompatible number of entity dofs for entity %d "
                     "and space no: %d",
                     entity_dim, i);
      }
    }
  }

  dolfin_assert(_receiving_indices.size()==N);
  dolfin_assert(_assigning_indices.size()==N);

  // Iterate over all spaces and collect dofs
  for (std::size_t i = 0; i < N; i++)
  {
    // Get dofmaps
    dolfin_assert(assigning_spaces[i]);
    dolfin_assert(assigning_spaces[i]->dofmap());
    const GenericDofMap& assigning_dofmap = *assigning_spaces[i]->dofmap();

    dolfin_assert(receiving_spaces[i]);
    dolfin_assert(receiving_spaces[i]->dofmap());
    const GenericDofMap& receiving_dofmap = *receiving_spaces[i]->dofmap();

    // Get on-process dof ranges
    const std::size_t bs_assigning = assigning_dofmap.block_size();
    const std::size_t assigning_range
      = assigning_dofmap.ownership_range().second
      - assigning_dofmap.ownership_range().first
      + bs_assigning*assigning_dofmap.index_map()->local_to_global_unowned().size();

    const std::size_t bs_receiving = receiving_dofmap.block_size();
    const std::size_t receiving_range
      = (receiving_dofmap.ownership_range().second
         - receiving_dofmap.ownership_range().first)
      + bs_receiving*receiving_dofmap.index_map()->local_to_global_unowned().size();

    // Create a map between receiving and assigning dofs
    std::map<std::size_t, std::size_t> receiving_assigning_map;

    // Iterate over cells and collect cell dofs
    for (CellIterator cell(mesh); !cell.end(); ++cell)
    {
      // Get local cell dofs
      const ArrayView<const dolfin::la_index> assigning_cell_dofs
        = assigning_dofmap.cell_dofs(cell->index());
      const ArrayView<const dolfin::la_index> receiving_cell_dofs
        = receiving_dofmap.cell_dofs(cell->index());

      // Check that both spaces have the same number of dofs
      if (assigning_cell_dofs.size() != receiving_cell_dofs.size())
      {
        dolfin_error("FunctionAssigner.cpp",
                     "create function assigner",
                     "The receiving and assigning spaces do not have the same "
                     "number of dofs per cell");
      }

      // Iterate over the local dofs and collect on-process dofs
      for (std::size_t j = 0; j < assigning_cell_dofs.size(); j++)
      {
        const std::size_t assigning_dof = assigning_cell_dofs[j];
        const std::size_t receiving_dof = receiving_cell_dofs[j];
        if (assigning_dof < assigning_range && receiving_dof < receiving_range)
          receiving_assigning_map[receiving_dof] = assigning_dof;
      }
    }

    // Transfer dofs to contiguous vectors
    _assigning_indices[i].reserve(receiving_assigning_map.size());
    _receiving_indices[i].reserve(receiving_assigning_map.size());

    std::map<std::size_t, std::size_t>::const_iterator it;
    for (it = receiving_assigning_map.begin();
         it != receiving_assigning_map.end(); ++it)
    {
      _receiving_indices[i].push_back(it->first);
      _assigning_indices[i].push_back(it->second);
    }

    // Resize transfer vector
    _transfer[i].resize(_receiving_indices[i].size());
  }
}
//-----------------------------------------------------------------------------
