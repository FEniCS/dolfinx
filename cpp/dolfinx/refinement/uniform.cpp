
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <iterator>
#include <vector>

#include "uniform.h"

using namespace dolfinx;

namespace
{

std::vector<std::int64_t>
pyr_subdivision(const std::vector<std::int64_t>& entities)
{
  int cells_pyr[25] = {0,  5,  6, 13, 7,  1,  8,  5, 13, 9,  3,  10, 8,
                       13, 12, 2, 6,  10, 13, 11, 7, 9,  11, 12, 4};

  //  topo_tet
  //      = [[edges [0], facets [0], edges [2], edges [4]],
  //         [edges [1], facets [0], edges [6], edges [2]],
  //         [edges [5], facets [0], edges [7], edges [6]],
  //         [edges [3], facets [0], edges [4], edges [7]]]

  std::vector<std::int64_t> topology(25);
  for (int i = 0; i < 25; ++i)
    topology[i] = entities[cells_pyr[i]];
  return topology;
}

std::vector<std::int64_t>
tet_subdivision(const std::vector<std::int64_t>& entities)
{
  std::vector<std::int64_t> topology;

  int cell_list[32] = {0, 7, 8, 9, 1, 5, 6, 9, 2, 4, 6, 8, 3, 4, 5, 7,
                       9, 4, 6, 8, 9, 4, 8, 7, 9, 4, 7, 5, 9, 4, 5, 6};

  for (int i = 0; i < 32; ++i)
    topology.push_back(entities[cell_list[i]]);
  return topology;
}

std::vector<std::int64_t>
prism_subdivision(const std::vector<std::int64_t>& entities)
{
  std::vector<std::int64_t> topology;
  int cell_list[48]
      = {0,  6,  7,  8,  15, 16, 6,  1,  9,  15, 10, 17, 7,  9,  2,  16,
         17, 11, 6,  9,  7,  15, 17, 16, 15, 17, 16, 12, 14, 13, 8,  15,
         16, 3,  12, 13, 11, 17, 16, 5,  14, 13, 10, 15, 17, 4,  12, 14};
  for (int i = 0; i < 48; ++i)
    topology.push_back(entities[cell_list[i]]);
  return topology;
}

std::vector<std::int64_t>
hex_subdivision(const std::vector<std::int64_t>& entities)
{
  int facet_list[8][3] = {{0, 1, 2}, {0, 1, 3}, {0, 2, 4}, {0, 3, 4},
                          {1, 2, 5}, {1, 3, 5}, {2, 4, 5}, {3, 4, 5}};

  int edge_list[8][3] = {{0, 1, 2}, {0, 3, 4},  {1, 5, 6},  {3, 5, 7},
                         {2, 8, 9}, {4, 8, 10}, {6, 9, 11}, {7, 10, 11}};

  std::vector<std::int64_t> topology;
  for (int vi = 0; vi < 8; ++vi)
  {
    int edge_offset = 8;
    int facet_offset = edge_offset + 12;
    auto ee = edge_list[vi];
    auto ff = facet_list[vi];
    std::array<std::int64_t, 8> new_cell = {entities[vi],
                                            entities[ee[1] + edge_offset],
                                            entities[ee[0] + edge_offset],
                                            entities[ff[0] + facet_offset],
                                            entities[ee[2] + edge_offset],
                                            entities[ff[2] + facet_offset],
                                            entities[ff[1] + facet_offset],
                                            entities.back()};
    topology.insert(topology.end(), new_cell.begin(), new_cell.end());
  }
  return topology;
}
} // namespace

mesh::Mesh<double> refinement::uniform_refine(const mesh::Mesh<double>& mesh)
{
  // Requires edges and facets to be built already
  auto topology = mesh.topology();

  // Collect up entity types on each dimension
  std::vector<std::vector<mesh::CellType>> entity_types;
  for (int i = 0; i < 4; ++i)
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

  // Check for hexahedral cells, and get index, if any.
  if (auto it = std::find(entity_types[3].begin(), entity_types[3].end(),
                          mesh::CellType::hexahedron);
      it != entity_types[3].end())
    e_index.push_back(std::distance(entity_types[3].begin(), it));

  // Add up all local vertices, edges, quad facets and hex cells.
  std::int64_t nlocal = 0;
  for (std::size_t dim = 0; dim < e_index.size(); ++dim)
  {
    index_maps.push_back(topology->index_maps(dim)[e_index[dim]]);
    new_v.push_back(std::vector<std::int64_t>(
        index_maps.back()->size_local() + index_maps.back()->num_ghosts()));
    nlocal += index_maps.back()->size_local();
  }

  // std::int64_t nv_global = index_maps[0]->size_global();

  // Get current geometry and put into new array for vertices
  std::vector<std::int32_t> vertex_to_x(index_maps[0]->size_local()
                                        + index_maps[0]->num_ghosts());

  // Iterate over cells
  for (int j = 0; j < static_cast<int>(entity_types[3].size()); ++j)
  {
    // Get geometry for each cell type
    auto x_dofmap = mesh.geometry().dofmap(j);
    auto c_to_v = topology->connectivity({3, j}, {0, 0});
    auto dof_layout = mesh.geometry().cmaps().at(j).create_dof_layout();
    std::vector<int> entity_dofs(dof_layout.num_dofs());
    for (int k = 0; k < dof_layout.num_dofs(); ++k)
      entity_dofs[k] = dof_layout.entity_dofs(0, k)[0];

    // Iterate over cells of this type
    for (std::int32_t c = 0; c < topology->index_maps(3)[j]->size_local()
                                     + topology->index_maps(3)[j]->num_ghosts();
         ++c)
    {
      auto vertices = c_to_v->links(c);
      for (std::size_t i = 0; i < vertices.size(); ++i)
        vertex_to_x[vertices[i]] = x_dofmap(c, entity_dofs[i]);
    }
  }

  // Copy existing vertices
  std::vector<double> new_x(nlocal * 3);
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
        std::array<double, 3> v = {0, 0, 0};
        for (std::size_t i = 0; i < nv_ent; ++i)
        {
          for (int k = 0; k < 3; ++k)
            v[k] += x_g[3 * vertex_to_x[vt[i]] + k];
        }
        for (int k = 0; k < 3; ++k)
          new_x[(w + w_off) * 3 + k] = v[k] / static_cast<double>(nv_ent);
      }
    }
    common::Scatterer sc(*index_maps[j], 1);
    sc.scatter_fwd(std::span<const std::int64_t>(new_v[j]),
                   std::span(std::next(new_v[j].begin(), num_entities),
                             index_maps[j]->num_ghosts()));
    w_off += num_entities;
  }

  // Create new topology...

  std::vector<std::vector<std::int64_t>> mixed_topology(entity_types[3].size());

  std::vector<std::function<std::vector<std::int64_t>(
      const std::vector<std::int64_t>&)>>
      subdiv;
  for (int k = 0; k < static_cast<int>(entity_types[3].size()); ++k)
  {
    if (entity_types[3][k] == mesh::CellType::hexahedron)
      subdiv.push_back(hex_subdivision);
    if (entity_types[3][k] == mesh::CellType::prism)
      subdiv.push_back(prism_subdivision);
    if (entity_types[3][k] == mesh::CellType::tetrahedron)
      subdiv.push_back(tet_subdivision);
    if (entity_types[3][k] == mesh::CellType::pyramid)
      subdiv.push_back(pyr_subdivision);
  }

  for (int k = 0; k < static_cast<int>(entity_types[3].size()); ++k)
  {
    auto c_to_v = topology->connectivity({3, k}, {0, 0});
    auto c_to_e = topology->connectivity({3, k}, {1, 0});

    for (int c = 0; c < topology->index_maps(3)[k]->size_local(); ++c)
    {
      std::vector<std::int64_t> entities;
      for (std::int32_t i : c_to_v->links(c))
        entities.push_back(new_v[0][i]);
      for (std::int32_t i : c_to_e->links(c))
        entities.push_back(new_v[1][i]);
      if (e_index.size() > 2)
      {
        auto conn = topology->connectivity({3, k}, {2, e_index[2]});
        if (conn)
        {
          spdlog::debug("Get connectivity from (3,{})->(2,{})", k, e_index[2]);
          for (std::int32_t i :
               topology->connectivity({3, k}, {2, e_index[2]})->links(c))
            entities.push_back(new_v[2][i]);
        }
      }
      if (e_index.size() > 3)
        entities.push_back(new_v[3][c]);

      auto new_cells = subdiv[k](entities);
      mixed_topology[k].insert(mixed_topology[k].end(), new_cells.begin(),
                               new_cells.end());
    }
  }

  auto partitioner = mesh::create_cell_partitioner(mesh::GhostMode::none);

  std::vector<std::span<const std::int64_t>> topo_span(mixed_topology.begin(),
                                                       mixed_topology.end());

  mesh::Mesh new_mesh = mesh::create_mesh(
      mesh.comm(), mesh.comm(), topo_span, mesh.geometry().cmaps(), mesh.comm(),
      new_x, {new_x.size() / 3, 3}, partitioner);

  return new_mesh;
}
