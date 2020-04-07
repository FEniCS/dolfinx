// Copyright (C) 2013-2020 Johan Hake, Jan Blechta and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateElement.h"
#include "DofMap.h"
#include "ElementDofLayout.h"
#include <dolfinx/common/types.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>
#include <string>
#include <vector>

struct ufc_dofmap;
struct ufc_form;
struct ufc_coordinate_mapping;
struct ufc_function_space;

namespace dolfinx
{
namespace common
{
class IndexMap;
}

namespace function
{
class Constant;
class Function;
class FunctionSpace;
} // namespace function

namespace mesh
{
class Mesh;
class Topology;
} // namespace mesh

namespace fem
{
class Form;

/// Extract FunctionSpaces for (0) rows blocks and (1) columns blocks
/// from a rectangular array of bilinear forms. Raises an exception if
/// there is an inconsistency. e.g. if each form in row i does not have
/// the same test space then an exception is raised.
///
/// @param[in] a A rectangular block on bilinear forms
/// @return Function spaces for each row blocks (0) and for each column
///     blocks (1).
std::array<std::vector<std::shared_ptr<const function::FunctionSpace>>, 2>
block_function_spaces(
    const Eigen::Ref<const Eigen::Array<const fem::Form*, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>& a);

/// Create a matrix
/// @param[in] a  A bilinear form
/// @return A matrix. The matrix is not zeroed.
la::PETScMatrix create_matrix(const Form& a);

/// Create a sparsity pattern for a given form. The pattern is not
/// finalised, i.e. the caller is responsible for calling
/// SparsityPattern::assemble.
/// @param[in] a A bilinear form
/// @return The corresponding sparsity pattern
la::SparsityPattern create_sparsity_pattern(const Form& a);

/// Initialise monolithic matrix for an array for bilinear forms. Matrix
/// is not zeroed.
la::PETScMatrix create_matrix_block(
    const Eigen::Ref<const Eigen::Array<const fem::Form*, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>& a);

/// Create nested (MatNest) matrix. Matrix is not zeroed.
la::PETScMatrix create_matrix_nest(
    const Eigen::Ref<const Eigen::Array<const fem::Form*, Eigen::Dynamic,
                                        Eigen::Dynamic, Eigen::RowMajor>>& a);

/// Initialise monolithic vector. Vector is not zeroed.
la::PETScVector
create_vector_block(const std::vector<const common::IndexMap*>& maps);

/// Create nested (VecNest) vector. Vector is not zeroed.
la::PETScVector
create_vector_nest(const std::vector<const common::IndexMap*>& maps);

/// @todo Update name an check efficiency
///
/// Get new global offset in 'spliced' indices
std::int64_t get_global_offset(const std::vector<const common::IndexMap*>& maps,
                               const int field, const std::int64_t index);

/// Create an ElementDofLayout from a ufc_dofmap
ElementDofLayout create_element_dof_layout(const ufc_dofmap& dofmap,
                                           const mesh::CellType cell_type,
                                           const std::vector<int>& parent_map
                                           = {});

/// Create dof map on mesh from a ufc_dofmap
/// @param[in] comm MPI communicator
/// @param[in] dofmap The ufc_dofmap
/// @param[in] topology The mesh topology
DofMap create_dofmap(MPI_Comm comm, const ufc_dofmap& dofmap,
                     mesh::Topology& topology);

/// Create a form from a form_create function returning a pointer to a
/// ufc_form, taking care of memory allocation
/// @param[in] fptr pointer to a function returning a pointer to
///    ufc_form
/// @param[in] spaces function spaces
/// @return Form
std::shared_ptr<Form> create_form(
    ufc_form* (*fptr)(),
    const std::vector<std::shared_ptr<const function::FunctionSpace>>& spaces);

/// Create a Form from UFC input
/// @param[in] ufc_form The UFC form
/// @param[in] spaces Vector of function spaces
Form create_form(
    const ufc_form& ufc_form,
    const std::vector<std::shared_ptr<const function::FunctionSpace>>& spaces);

/// Extract coefficients from a UFC form
std::vector<std::tuple<int, std::string, std::shared_ptr<function::Function>>>
get_coeffs_from_ufc_form(const ufc_form& ufc_form);

/// Extract coefficients from a UFC form
std::vector<std::pair<std::string, std::shared_ptr<const function::Constant>>>
get_constants_from_ufc_form(const ufc_form& ufc_form);

/// Create a CoordinateElement from ufc
/// @param[in] ufc_cmap UFC coordinate mapping
/// @return A DOLFINX coordinate map
fem::CoordinateElement
create_coordinate_map(const ufc_coordinate_mapping& ufc_cmap);

/// Create a CoordinateElement from ufc
/// @param[in] fptr Function Pointer to a ufc_function_coordinate_map
///   function
/// @return A DOLFINX coordinate map
fem::CoordinateElement
create_coordinate_map(ufc_coordinate_mapping* (*fptr)());

/// Create FunctionSpace from UFC
/// @param[in] fptr Function Pointer to a ufc_function_space_create
///   function
/// @param[in] function_name Name of a function whose function space to
///   create. Function name is the name of Python variable for
///   ufl.Coefficient, ufl.TrialFunction or ufl.TestFunction as defined
///   in the UFL file.
/// @param[in] mesh Mesh
/// @return The created FunctionSpace
std::shared_ptr<function::FunctionSpace>
create_functionspace(ufc_function_space* (*fptr)(const char*),
                     const std::string function_name,
                     std::shared_ptr<mesh::Mesh> mesh);

// NOTE: This is subject to change
/// Pack form coefficients ready for assembly
Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
pack_coefficients(const fem::Form& form);

// NOTE: This is subject to change
/// Pack form constants ready for assembly
Eigen::Array<PetscScalar, Eigen::Dynamic, 1>
pack_constants(const fem::Form& form);

} // namespace fem
} // namespace dolfinx
