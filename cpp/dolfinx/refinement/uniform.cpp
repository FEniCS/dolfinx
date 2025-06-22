
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <iterator>
#include <vector>

#include "uniform.h"

using namespace dolfinx;

template <typename T>
mesh::Mesh<T> refinement::uniform_refine(const mesh::Mesh<T>& mesh)
{
  // Requires edges (and facets for some 3D meshes) to be built already
  auto topology = mesh.topology();
  int tdim = topology->dim();
  if (tdim < 2)
    throw std::runtime_error("Uniform refinement only for 2D and 3D meshes");

  spdlog::info("Topology dim = {}", tdim);

  // Collect up entity types on each dimension
  std::vector<std::vector<mesh::CellType>> entity_types;
  for (int i = 0; i < tdim + 1; ++i)
    entity_types.push_back(topology->entity_types(i));

  // Get index maps for vertices and edges and create maps for new vertices
  std::vector<std::vector<std::int64_t>> new_v;
  std::vector<std::shared_ptr<const common::IndexMap>> index_maps;

  // Indices for vertices and edges on dim 0 and dim 1.
  std::vector<int> e_index = {0, 0};

  // Check for quadrilateral faces and get index, if any.
  if (auto it = std::find(entity_types[2].begin(), entity_types[2].end(),
                          mesh::CellType::quadrilateral);
      it != entity_types[2].end())
    e_index.push_back(std::distance(entity_types[2].begin(), it));

  if (tdim == 3)
  {
    // In 3D, check for hexahedral cells, and get index, if any.
    if (auto it = std::find(entity_types[3].begin(), entity_types[3].end(),
                            mesh::CellType::hexahedron);
        it != entity_types[3].end())
      e_index.push_back(std::distance(entity_types[3].begin(), it));
  }

  // Add up all local vertices, edges, quad facets and hex cells.
  std::int64_t nlocal = 0;
  for (std::size_t dim = 0; dim < e_index.size(); ++dim)
  {
    if (topology->index_maps(dim).empty())
      throw std::runtime_error(
          "Missing entities of dimension " + std::to_string(dim)
          + ", need to call create_entities(" + std::to_string(dim) + ")");
    index_maps.push_back(topology->index_maps(dim)[e_index[dim]]);
    new_v.push_back(std::vector<std::int64_t>(
        index_maps.back()->size_local() + index_maps.back()->num_ghosts()));
    nlocal += index_maps.back()->size_local();
  }

  // Get current geometry and put into new array for vertices
  std::vector<std::int32_t> vertex_to_x(index_maps[0]->size_local()
                                        + index_maps[0]->num_ghosts());

  // Iterate over cells
  for (int j = 0; j < static_cast<int>(entity_types.back().size()); ++j)
  {
    // Get geometry for each cell type
    auto x_dofmap = mesh.geometry().dofmap(j);
    auto c_to_v = topology->connectivity({tdim, j}, {0, 0});
    auto dof_layout = mesh.geometry().cmaps().at(j).create_dof_layout();
    std::vector<int> entity_dofs(dof_layout.num_dofs());
    for (int k = 0; k < dof_layout.num_dofs(); ++k)
      entity_dofs[k] = dof_layout.entity_dofs(0, k)[0];

    // Iterate over cells of this type
    for (std::int32_t c = 0;
         c < topology->index_maps(tdim)[j]->size_local()
                 + topology->index_maps(tdim)[j]->num_ghosts();
         ++c)
    {
      auto vertices = c_to_v->links(c);
      for (std::size_t i = 0; i < vertices.size(); ++i)
        vertex_to_x[vertices[i]] = x_dofmap(c, entity_dofs[i]);
    }
  }

  // Copy existing vertices
  std::vector<T> new_x(nlocal * 3);
  auto x_g = mesh.geometry().x();
  for (int i = 0; i < index_maps[0]->size_local(); ++i)
    for (int j = 0; j < 3; ++j)
      new_x[i * 3 + j] = x_g[vertex_to_x[i] * 3 + j];

  // Get this process global offset and range
  std::int64_t nscan;
  MPI_Scan(&nlocal, &nscan, 1, MPI_INT64_T, MPI_SUM, mesh.comm());
  std::array<std::int64_t, 2> local_range = {nscan - nlocal, nscan};

  std::int32_t w_off = 0;
  for (int j = 0; j < static_cast<int>(index_maps.size()); ++j)
  {
    std::int32_t num_entities = index_maps[j]->size_local();
    std::iota(new_v[j].begin(), std::next(new_v[j].begin(), num_entities),
              local_range[0] + w_off);

    if (j > 0)
    {
      auto e_to_v = topology->connectivity({j, e_index[j]}, {0, 0});
      for (std::int32_t w = 0; w < num_entities; ++w)
      {
        auto vt = e_to_v->links(w);
        std::size_t nv_ent = vt.size();
        std::array<T, 3> v = {0, 0, 0};
        for (std::size_t i = 0; i < nv_ent; ++i)
        {
          for (int k = 0; k < 3; ++k)
            v[k] += x_g[3 * vertex_to_x[vt[i]] + k];
        }
        for (int k = 0; k < 3; ++k)
          new_x[(w + w_off) * 3 + k] = v[k] / static_cast<T>(nv_ent);
      }
    }
    common::Scatterer sc(*index_maps[j], 1);
    sc.scatter_fwd(std::span<const std::int64_t>(new_v[j]),
                   std::span(std::next(new_v[j].begin(), num_entities),
                             index_maps[j]->num_ghosts()));
    w_off += num_entities;
  }

  // Create new topology...
  const std::vector<mesh::CellType>& cell_entity_types = entity_types.back();
  std::vector<std::vector<std::int64_t>> mixed_topology(
      cell_entity_types.size());

  // Find index of tets in topology list, if any
  int ktet = -1;
  auto it = std::find(cell_entity_types.begin(), cell_entity_types.end(),
                      mesh::CellType::tetrahedron);
  if (it != cell_entity_types.end())
    ktet = std::distance(cell_entity_types.begin(), it);
  // Topology for tetrahedra which arise from pyramid subdivision
  std::vector<int> pyr_to_tet_list
      = {5, 13, 7, 9, 6, 13, 11, 7, 10, 13, 12, 11, 8, 13, 9, 12};

  std::vector<int> refined_cell_list;
  for (int k = 0; k < static_cast<int>(cell_entity_types.size()); ++k)
  {
    // Reserve an estimate of space for the topology of each type
    mixed_topology[k].reserve(mesh.topology()->index_maps(tdim)[k]->size_local()
                              * 8 * 6);

    // Select correct subdivision for celltype
    // Hex -> 8 hex, Prism -> 8 prism, Tet -> 8 tet, Pyr -> 5 pyr + 4 tet
    if (cell_entity_types[k] == mesh::CellType::hexahedron)
    {
      spdlog::info("Hex subdivision [{}]", k);
      refined_cell_list
          = {0, 9,  8,  20, 10, 22, 21, 26, 1, 11, 8,  20, 12, 23, 21, 26,
             2, 13, 9,  20, 14, 24, 22, 26, 3, 13, 11, 20, 15, 24, 23, 26,
             4, 16, 10, 21, 17, 25, 22, 26, 5, 16, 12, 21, 18, 25, 23, 26,
             6, 17, 14, 22, 19, 25, 24, 26, 7, 18, 15, 23, 19, 25, 24, 26};
    }
    else if (cell_entity_types[k] == mesh::CellType::tetrahedron)
    {
      spdlog::info("Tet subdivision [{}]", k);
      refined_cell_list = {0, 7, 8, 9, 1, 5, 6, 9, 2, 4, 6, 8, 3, 4, 5, 7,
                           9, 4, 6, 8, 9, 4, 8, 7, 9, 4, 7, 5, 9, 4, 5, 6};
    }
    else if (cell_entity_types[k] == mesh::CellType::prism)
    {
      spdlog::info("Prism subdivision [{}]", k);
      refined_cell_list
          = {0,  6,  7,  8,  15, 16, 6,  1,  9,  15, 10, 17, 7,  9,  2,  16,
             17, 11, 6,  9,  7,  15, 17, 16, 15, 17, 16, 12, 14, 13, 8,  15,
             16, 3,  12, 13, 11, 17, 16, 5,  14, 13, 10, 15, 17, 4,  12, 14};
    }
    else if (cell_entity_types[k] == mesh::CellType::pyramid)
    {
      spdlog::info("Pyramid subdivision [{}]", k);
      refined_cell_list = {0,  5,  6, 13, 7,  1,  8,  5, 13, 9,  3,  10, 8,
                           13, 12, 2, 6,  10, 13, 11, 7, 9,  11, 12, 4};
      if (ktet == -1)
        throw std::runtime_error("Cannot refine mesh with pyramids and no "
                                 "tetrahedra.");
    }
    else if (cell_entity_types[k] == mesh::CellType::triangle)
    {
      spdlog::info("Triangle subdivision [{}]", k);
      refined_cell_list = {0, 4, 5, 1, 5, 3, 2, 3, 4, 3, 4, 5};
    }
    else if (cell_entity_types[k] == mesh::CellType::quadrilateral)
    {
      spdlog::info("Quad subdivision [{}]", k);
      refined_cell_list = {0, 4, 5, 8, 1, 6, 4, 8, 2, 7, 5, 8, 3, 7, 6, 8};
    }

    auto c_to_v = topology->connectivity({tdim, k}, {0, 0});
    auto c_to_e = topology->connectivity({tdim, k}, {1, 0});

    spdlog::debug("On {}, over {} cells", k,
                  topology->index_maps(tdim)[k]->size_local());
    for (int c = 0; c < topology->index_maps(tdim)[k]->size_local(); ++c)
    {
      std::vector<std::int64_t> entities;
      for (std::int32_t i : c_to_v->links(c))
        entities.push_back(new_v[0][i]);
      for (std::int32_t i : c_to_e->links(c))
        entities.push_back(new_v[1][i]);
      if (e_index.size() > 2)
      {
        if (tdim == 3)
        {
          auto conn = topology->connectivity({3, k}, {2, e_index[2]});
          if (conn)
          {
            for (std::int32_t i :
                 topology->connectivity({3, k}, {2, e_index[2]})->links(c))
              entities.push_back(new_v[2][i]);
          }
        }
        else if (cell_entity_types[k] == mesh::CellType::quadrilateral)
          entities.push_back(new_v[2][c]);
      }

      if (e_index.size() > 3
          and cell_entity_types[k] == mesh::CellType::hexahedron)
        entities.push_back(new_v[3][c]);

      for (int i : refined_cell_list)
        mixed_topology[k].push_back(entities[i]);

      if (cell_entity_types[k] == mesh::CellType::pyramid)
      {
        for (int i : pyr_to_tet_list)
          mixed_topology[ktet].push_back(entities[i]);
      }
    }
  }

  spdlog::debug("Create partitioner");
  auto partitioner = mesh::create_cell_partitioner(mesh::GhostMode::none);

  std::vector<std::span<const std::int64_t>> topo_span(mixed_topology.begin(),
                                                       mixed_topology.end());

  spdlog::debug("Create new mesh");
  mesh::Mesh new_mesh = mesh::create_mesh(
      mesh.comm(), mesh.comm(), topo_span, mesh.geometry().cmaps(), mesh.comm(),
      new_x, {new_x.size() / 3, 3}, partitioner);

  return new_mesh;
}

/// @cond Explicit instatiation for float and double
template mesh::Mesh<double>
refinement::uniform_refine(const mesh::Mesh<double>& mesh);
template mesh::Mesh<float>
refinement::uniform_refine(const mesh::Mesh<float>& mesh);
/// @endcond
