// Copyright (C) 2019-2020 Garth N. Wells and Chris Richardson
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FunctionSpace.h"
#include <algorithm>
#include <array>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace dolfinx::fem
{

template <typename T>
class Constant;
template <typename T>
class Function;

/// @brief Type of integral
enum class IntegralType : std::int8_t
{
  cell = 0,           ///< Cell
  exterior_facet = 1, ///< Exterior facet
  interior_facet = 2, ///< Interior facet
  vertex = 3          ///< Vertex
};

/// @brief A representation of finite element variational forms.
///
/// A note on the order of trial and test spaces: FEniCS numbers
/// argument spaces starting with the leading dimension of the
/// corresponding tensor (matrix). In other words, the test space is
/// numbered 0 and the trial space is numbered 1. However, in order to
/// have a notation that agrees with most existing finite element
/// literature, in particular
///
///  \f[   a = a(u, v)        \f]
///
/// the spaces are numbered from right to left
///
///  \f[   a: V_1 \times V_0 \rightarrow \mathbb{R}  \f]
///
/// This is reflected in the ordering of the spaces that should be
/// supplied to generated subclasses. In particular, when a bilinear
/// form is initialized, it should be initialized as `a(V_1, V_0) =
/// ...`, where `V_1` is the trial space and `V_0` is the test space.
/// However, when a form is initialized by a list of argument spaces
/// (the variable `function_spaces` in the constructors below), the list
/// of spaces should start with space number 0 (the test space) and then
/// space number 1 (the trial space).
template <typename T>
class Form
{
  template <typename X, typename = void>
  struct scalar_value_type
  {
    typedef X value_type;
  };
  template <typename X>
  struct scalar_value_type<X, std::void_t<typename X::value_type>>
  {
    typedef typename X::value_type value_type;
  };
  using scalar_value_type_t = typename scalar_value_type<T>::value_type;

public:
  /// @brief Create a finite element form.
  ///
  /// @note User applications will normally call a fem::Form builder
  /// function rather using this interface directly.
  ///
  /// @param[in] function_spaces Function spaces for the form arguments
  /// @param[in] integrals The integrals in the form. The first key is
  /// the domain type. For each key there is a map from the domain id
  /// to a pair containing the integration kernel and a list of entities
  /// over which to integrate
  /// @param[in] coefficients
  /// @param[in] constants Constants in the Form
  /// @param[in] needs_facet_permutations Set to true is any of the
  /// integration kernels require cell permutation data
  /// @param[in] mesh The mesh of the domain. This is required when
  /// there are not argument functions from which the mesh can be
  /// extracted, e.g. for functionals
  // TODO Could change first integrals map to std::array and second
  // to vector of pair
  Form(const std::vector<std::shared_ptr<const FunctionSpace>>& function_spaces,
       const std::map<
           IntegralType,
           std::map<int, std::pair<std::function<
                                       void(T*, const T*, const T*,
                                            const scalar_value_type_t*,
                                            const int*, const std::uint8_t*)>,
                                   std::vector<std::int32_t>>>>& integrals,
       const std::vector<std::shared_ptr<const Function<T>>>& coefficients,
       const std::vector<std::shared_ptr<const Constant<T>>>& constants,
       bool needs_facet_permutations,
       std::shared_ptr<const mesh::Mesh> mesh = nullptr)
      : _function_spaces(function_spaces), _coefficients(coefficients),
        _constants(constants), _mesh(mesh),
        _needs_facet_permutations(needs_facet_permutations)
  {
    // Extract _mesh from FunctionSpace, and check they are the same
    if (!_mesh and !function_spaces.empty())
      _mesh = function_spaces[0]->mesh();
    for (const auto& V : function_spaces)
    {
      if (_mesh != V->mesh())
        throw std::runtime_error("Incompatible mesh");
    }
    if (!_mesh)
      throw std::runtime_error("No mesh could be associated with the Form.");

    // Set integral data (intentially copying)
    // NOTE Could probably simplify Form by combining integral types
    for (const auto& [type, integral] : integrals)
    {
      switch (type)
      {
      case IntegralType::cell:
        _cell_integrals = integral;
        break;
      case IntegralType::exterior_facet:
        _exterior_facet_integrals = integral;
        break;
      case IntegralType::interior_facet:
        _interior_facet_integrals = integral;
        break;
      }
    }

    // FIXME: do this neatly via a static function
    // Set markers for default integrals
    set_default_domains(*_mesh);
  }

  /// Copy constructor
  Form(const Form& form) = delete;

  /// Move constructor
  Form(Form&& form) = default;

  /// Destructor
  virtual ~Form() = default;

  /// Rank of the form (bilinear form = 2, linear form = 1, functional =
  /// 0, etc)
  /// @return The rank of the form
  int rank() const { return _function_spaces.size(); }

  /// Extract common mesh for the form
  /// @return The mesh
  std::shared_ptr<const mesh::Mesh> mesh() const { return _mesh; }

  /// Return function spaces for all arguments
  /// @return Function spaces
  const std::vector<std::shared_ptr<const FunctionSpace>>&
  function_spaces() const
  {
    return _function_spaces;
  }

  /// Get the function for 'kernel' for integral i of given
  /// type
  /// @param[in] type Integral type
  /// @param[in] i Domain index
  /// @return Function to call for tabulate_tensor
  const std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                           const int*, const std::uint8_t*)>&
  kernel(IntegralType type, int i) const
  {
    switch (type)
    {
    case IntegralType::cell:
      return get_kernel_from_integrals(_cell_integrals, i);
    case IntegralType::exterior_facet:
      return get_kernel_from_integrals(_exterior_facet_integrals, i);
    case IntegralType::interior_facet:
      return get_kernel_from_integrals(_interior_facet_integrals, i);
    default:
      throw std::runtime_error(
          "Cannot access kernel. Integral type not supported.");
    }
  }

  /// Get types of integrals in the form
  /// @return Integrals types
  std::set<IntegralType> integral_types() const
  {
    std::set<IntegralType> set;
    if (!_cell_integrals.empty())
      set.insert(IntegralType::cell);
    if (!_exterior_facet_integrals.empty())
      set.insert(IntegralType::exterior_facet);
    if (!_interior_facet_integrals.empty())
      set.insert(IntegralType::interior_facet);

    return set;
  }

  /// Number of integrals of given type
  /// @param[in] type Integral type
  /// @return Number of integrals
  int num_integrals(IntegralType type) const
  {
    switch (type)
    {
    case IntegralType::cell:
      return _cell_integrals.size();
    case IntegralType::exterior_facet:
      return _exterior_facet_integrals.size();
    case IntegralType::interior_facet:
      return _interior_facet_integrals.size();
    default:
      throw std::runtime_error("Integral type not supported.");
    }
  }

  /// Get the IDs for integrals (kernels) for given integral type. The
  /// IDs correspond to the domain IDs which the integrals are defined
  /// for in the form. ID=-1 is the default integral over the whole
  /// domain.
  /// @param[in] type Integral type
  /// @return List of IDs for given integral type
  std::vector<int> integral_ids(IntegralType type) const
  {
    std::vector<int> ids;
    switch (type)
    {
    case IntegralType::cell:
      std::transform(_cell_integrals.cbegin(), _cell_integrals.cend(),
                     std::back_inserter(ids),
                     [](auto& integral) { return integral.first; });
      break;
    case IntegralType::exterior_facet:
      std::transform(_exterior_facet_integrals.cbegin(),
                     _exterior_facet_integrals.cend(), std::back_inserter(ids),
                     [](auto& integral) { return integral.first; });
      break;
    case IntegralType::interior_facet:
      std::transform(_interior_facet_integrals.cbegin(),
                     _interior_facet_integrals.cend(), std::back_inserter(ids),
                     [](auto& integral) { return integral.first; });
      break;
    default:
      throw std::runtime_error(
          "Cannot return IDs. Integral type not supported.");
    }

    return ids;
  }

  /// Get the list of cell indices for the ith integral (kernel)
  /// for the cell domain type
  /// @param[in] i Integral ID, i.e. (sub)domain index
  /// @return List of active cell entities for the given integral (kernel)
  const std::vector<std::int32_t>& cell_domains(int i) const
  {
    auto it = _cell_integrals.find(i);
    if (it == _cell_integrals.end())
      throw std::runtime_error("No mesh entities for requested domain index.");
    return it->second.second;
  }

  /// @brief List of (cell_index, local_facet_index) pairs for the ith
  /// integral (kernel) for the exterior facet domain type.
  /// @param[in] i Integral ID, i.e. (sub)domain index
  /// @return List of (cell_index, local_facet_index) pairs. This data is
  /// flattened with row-major layout, shape=(num_facets, 2)
  const std::vector<std::int32_t>& exterior_facet_domains(int i) const
  {
    auto it = _exterior_facet_integrals.find(i);
    if (it == _exterior_facet_integrals.end())
      throw std::runtime_error("No mesh entities for requested domain index.");
    return it->second.second;
  }

  /// Get the list of (cell_index_0, local_facet_index_0, cell_index_1,
  /// local_facet_index_1) quadruplets for the ith integral (kernel) for the
  /// interior facet domain type.
  /// @param[in] i Integral ID, i.e. (sub)domain index
  /// @return List of tuples of the form
  /// (cell_index_0, local_facet_index_0, cell_index_1, local_facet_index_1).
  /// This data is flattened with row-major layout, shape=(num_facets, 4)
  const std::vector<std::int32_t>& interior_facet_domains(int i) const
  {
    auto it = _interior_facet_integrals.find(i);
    if (it == _interior_facet_integrals.end())
      throw std::runtime_error("No mesh entities for requested domain index.");
    return it->second.second;
  }

  /// Access coefficients
  const std::vector<std::shared_ptr<const Function<T>>>& coefficients() const
  {
    return _coefficients;
  }

  /// Get bool indicating whether permutation data needs to be passed
  /// into these integrals
  /// @return True if cell permutation data is required
  bool needs_facet_permutations() const { return _needs_facet_permutations; }

  /// Offset for each coefficient expansion array on a cell. Used to
  /// pack data for multiple coefficients in a flat array. The last
  /// entry is the size required to store all coefficients.
  std::vector<int> coefficient_offsets() const
  {
    std::vector<int> n = {0};
    for (const auto& c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      n.push_back(n.back() + c->function_space()->element()->space_dimension());
    }
    return n;
  }

  /// Access constants
  const std::vector<std::shared_ptr<const Constant<T>>>& constants() const
  {
    return _constants;
  }

  /// Scalar type (T)
  using scalar_type = T;

private:
  using kern
      = std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                           const int*, const std::uint8_t*)>;

  // Helper function to get the kernel for integral i from a map
  // of integrals i.e. from _cell_integrals
  // @param[in] integrals Map of integrals
  // @param[in] i Domain index
  // @return Function to call for tabulate_tensor
  template <typename U>
  const std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                           const int*, const std::uint8_t*)>&
  get_kernel_from_integrals(const U& integrals, int i) const
  {
    auto it = integrals.find(i);
    if (it == integrals.end())
      throw std::runtime_error("No kernel for requested domain index.");
    return it->second.first;
  }

  /// If there exists a default integral of any type, set the list of
  /// entities for those integrals from the mesh topology. For cell
  /// integrals, this is all cells. For facet integrals, it is either
  /// all interior or all exterior facets.
  /// @param[in] mesh Mesh
  void set_default_domains(const mesh::Mesh& mesh)
  {
    const mesh::Topology& topology = mesh.topology();
    const int tdim = topology.dim();

    // Cells. If there is a default integral, define it on all owned
    // cells
    for (auto& [domain_id, kernel_cells] : _cell_integrals)
    {
      if (domain_id == -1)
      {
        std::vector<std::int32_t>& cells = kernel_cells.second;
        const int num_cells = topology.index_map(tdim)->size_local();
        cells.resize(num_cells);
        std::iota(cells.begin(), cells.end(), 0);
      }
    }

    // Exterior facets. If there is a default integral, define it only
    // on owned surface facets.

    if (!_exterior_facet_integrals.empty())
    {
      mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
      mesh.topology_mutable().create_connectivity(tdim, tdim - 1);
    }

    const std::vector<std::int32_t> boundary_facets
        = _exterior_facet_integrals.empty()
              ? std::vector<std::int32_t>()
              : mesh::exterior_facet_indices(topology);
    for (auto& [domain_id, kernel_facets] : _exterior_facet_integrals)
    {
      if (domain_id == -1)
      {
        std::vector<std::int32_t>& facets = kernel_facets.second;
        facets.clear();

        auto f_to_c = topology.connectivity(tdim - 1, tdim);
        assert(f_to_c);
        auto c_to_f = topology.connectivity(tdim, tdim - 1);
        assert(c_to_f);
        for (std::int32_t f : boundary_facets)
        {
          // There will only be one pair for an exterior facet integral
          std::array<std::int32_t, 2> pair
              = mesh::get_cell_local_facet_pairs<1>(f, f_to_c->links(f),
                                                    *c_to_f)[0];
          facets.insert(facets.end(), pair.cbegin(), pair.cend());
        }
      }
    }

    // Interior facets. If there is a default integral, define it only on
    // owned interior facets.
    for (auto& [domain_id, kernel_facets] : _interior_facet_integrals)
    {
      if (domain_id == -1)
      {
        std::vector<std::int32_t>& facets = kernel_facets.second;
        facets.clear();

        mesh.topology_mutable().create_connectivity(tdim - 1, tdim);
        auto f_to_c = topology.connectivity(tdim - 1, tdim);
        assert(f_to_c);
        mesh.topology_mutable().create_connectivity(tdim, tdim - 1);
        auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
        assert(c_to_f);

        // Get number of facets owned by this process
        assert(topology.index_map(tdim - 1));
        const int num_facets = topology.index_map(tdim - 1)->size_local();
        facets.reserve(num_facets);
        for (int f = 0; f < num_facets; ++f)
        {
          if (f_to_c->num_links(f) == 2)
          {
            const std::array<std::array<std::int32_t, 2>, 2> pairs
                = mesh::get_cell_local_facet_pairs<2>(f, f_to_c->links(f),
                                                      *c_to_f);
            facets.insert(facets.end(), pairs[0].cbegin(), pairs[0].cend());
            facets.insert(facets.end(), pairs[1].cbegin(), pairs[1].cend());
          }
        }
      }
    }
  }

  // Function spaces (one for each argument)
  std::vector<std::shared_ptr<const FunctionSpace>> _function_spaces;

  // Form coefficients
  std::vector<std::shared_ptr<const Function<T>>> _coefficients;

  // Constants associated with the Form
  std::vector<std::shared_ptr<const Constant<T>>> _constants;

  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  // Cell integrals
  std::map<int, std::pair<kern, std::vector<std::int32_t>>> _cell_integrals;

  // Exterior facet integrals
  std::map<int, std::pair<kern, std::vector<std::int32_t>>>
      _exterior_facet_integrals;

  // Interior facet integrals
  std::map<int, std::pair<kern, std::vector<std::int32_t>>>
      _interior_facet_integrals;

  // True if permutation data needs to be passed into these integrals
  bool _needs_facet_permutations;
};
} // namespace dolfinx::fem
