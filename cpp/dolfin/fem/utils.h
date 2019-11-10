// Copyright (C) 2013, 2015, 2016 Johan Hake, Jan Blechta
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateElement.h"
#include "DofMap.h"
#include "ElementDofLayout.h"
#include <dolfin/common/types.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/cell_types.h>
#include <memory>
#include <vector>

struct ufc_dofmap;
struct ufc_form;
struct ufc_coordinate_mapping;
struct ufc_function_space;

namespace dolfin
{
namespace common
{
class IndexMap;
}

namespace la
{
class PETScMatrix;
class PETScVector;
} // namespace la
namespace function
{
class Constant;
class Function;
class FunctionSpace;
} // namespace function

namespace mesh
{
class Mesh;
} // namespace mesh

namespace fem
{
class Form;

/// Extract FunctionSpaces for (0) rows blocks and (1) columns blocks
/// from a rectangular array of bilinear forms. Raises an exception if
/// there is an inconsistency. e.g. if each form in row i does not have
/// the same test space then an exception is raised.
/// @param[in] a A rectangular block on bilinear forms
/// @return Function spaces for each row blocks (0) and for each column
/// blocks (1).
std::array<std::vector<std::shared_ptr<const function::FunctionSpace>>, 2>
block_function_spaces(const std::vector<std::vector<const fem::Form*>>& a);

/// Create matrix. Matrix is not zeroed.
la::PETScMatrix create_matrix(const Form& a);

/// Initialise monolithic matrix for an array for bilinear forms. Matrix
/// is not zeroed.
la::PETScMatrix
create_matrix_block(const std::vector<std::vector<const fem::Form*>>& a);

/// Create nested (MatNest) matrix. Matrix is not zeroed.
la::PETScMatrix
create_matrix_nest(const std::vector<std::vector<const fem::Form*>>& a);

/// Initialise monolithic vector. Vector is not zeroed.
la::PETScVector
create_vector_block(const std::vector<const common::IndexMap*>& maps);

/// Create nested (VecNest) vector. Vector is not zeroed.
la::PETScVector
create_vector_nest(const std::vector<const common::IndexMap*>& maps);

/// Get new global index in 'spliced' indices
std::size_t get_global_index(const std::vector<const common::IndexMap*>& maps,
                             const int field, const int n);

/// Create an ElementDofLayout from a ufc_dofmap
ElementDofLayout create_element_dof_layout(const ufc_dofmap& dofmap,
                                           const mesh::CellType cell_type,
                                           const std::vector<int>& parent_map
                                           = {});

/// Create dof map on mesh from a ufc_dofmap
///
/// @param[in] dofmap The ufc_dofmap.
/// @param[in] mesh The mesh.
DofMap create_dofmap(const ufc_dofmap& dofmap, const mesh::Mesh& mesh);

/// Create form (shared data)
///
/// @param[in] ufc_form The UFC form.
/// @param[in] spaces Vector of function spaces.
Form create_form(
    const ufc_form& ufc_form,
    const std::vector<std::shared_ptr<const function::FunctionSpace>>& spaces);

/// Extract coefficients from UFC form
std::vector<std::tuple<int, std::string, std::shared_ptr<function::Function>>>
get_coeffs_from_ufc_form(const ufc_form& ufc_form);

/// Extract coefficients from UFC form
std::vector<std::pair<std::string, std::shared_ptr<const function::Constant>>>
get_constants_from_ufc_form(const ufc_form& ufc_form);

/// Get dolfin::fem::CoordinateElement from ufc
std::shared_ptr<const fem::CoordinateElement>
get_cmap_from_ufc_cmap(const ufc_coordinate_mapping& ufc_cmap);

/// Create FunctionSpace from UFC
/// @param fptr Function Pointer to a ufc_function_space_create function
/// @param mesh Mesh
/// @return The created FunctionSpace
std::shared_ptr<function::FunctionSpace>
create_functionspace(ufc_function_space* (*fptr)(void),
                     std::shared_ptr<mesh::Mesh> mesh);

// NOTE: This is subject to change
/// Pack form coeffcients ready for assembly
Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
pack_coefficients(const fem::Form& form);

} // namespace fem
} // namespace dolfin
