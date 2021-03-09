#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/mesh/MeshTags.h>

using namespace dolfinx;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  {
    MPI_Comm comm = MPI_COMM_WORLD;
    // Create mesh and function space
    auto cmap = fem::create_coordinate_map(create_coordinate_map_poisson);
    auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
        comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}}}, {2, 2}, cmap,
        mesh::GhostMode::none));

    int tdim = mesh->topology().dim();
    mesh->topology().create_connectivity(tdim - 1, tdim);
    mesh->topology().create_connectivity(tdim - 1, 0);
    mesh->topology().create_connectivity(0, tdim);

    std::vector<bool> bnd_facets
        = dolfinx::mesh::compute_interface_facets(mesh->topology());

    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);

    int mpi_size;
    MPI_Comm_size(comm, &mpi_size);

    auto fv = mesh->topology().connectivity(tdim - 1, 0);
    auto vc = mesh->topology().connectivity(0, tdim);

    std::vector<std::int32_t> facet_indices;
    for (std::size_t f = 0; f < bnd_facets.size(); ++f)
      if (bnd_facets[f])
        facet_indices.push_back(f);

    // Identify interface vertices
    std::vector<std::int32_t> int_vertices;
    int_vertices.reserve(facet_indices.size() * 2);
    for (auto f : facet_indices)
      for (auto v : fv->links(f))
        int_vertices.push_back(v);

    // Remove repeated and owned vertices
    auto vertex_index_map = mesh->topology().index_map(0);
    std::sort(int_vertices.begin(), int_vertices.end());
    int_vertices.erase(std::unique(int_vertices.begin(), int_vertices.end()),
                       int_vertices.end());

    std::int32_t local_size = vertex_index_map->size_local();
    int_vertices.erase(
        std::remove_if(int_vertices.begin(), int_vertices.end(),
                       [local_size](std::int32_t v) { return v < local_size; }),
        int_vertices.end());

    // Get global indices
    std::vector<std::int64_t> int_vertices_global(int_vertices.size());
    vertex_index_map->local_to_global(int_vertices, int_vertices_global);

    // Get interface vertices owners
    auto ghost_owners = vertex_index_map->ghost_owner_rank();
    auto ghosts = vertex_index_map->ghosts();
    std::vector<std::int32_t> owner(int_vertices_global.size());
    for (std::size_t i = 0; i < int_vertices_global.size(); i++)
    {
      std::int64_t ghost = int_vertices_global[i];
      auto it = std::find(ghosts.begin(), ghosts.end(), ghost);
      assert(it != ghosts.end());
      int pos = std::distance(ghosts.begin(), it);
      owner[i] = ghost_owners[pos];
    }

    // Figure out how much data to send to each neighbor (ghost owner)
    MPI_Comm reverse_com
        = vertex_index_map->comm(common::IndexMap::Direction::reverse);
    auto [sources, destinations] = dolfinx::MPI::neighbors(reverse_com);
    std::vector<int> send_sizes(destinations.size(), 0);
    std::vector<int> recv_sizes(sources.size(), 0);
    for (std::size_t i = 0; i < int_vertices_global.size(); i++)
    {
      auto it = std::find(destinations.begin(), destinations.end(), owner[i]);
      assert(it != destinations.end());
      int pos = std::distance(destinations.begin(), it);
      send_sizes[pos]++;
    }

    MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                          MPI_INT, reverse_com);

    // Prepare communication displacements
    std::vector<int> send_disp(destinations.size() + 1, 0);
    std::vector<int> recv_disp(sources.size() + 1, 0);
    std::partial_sum(send_sizes.begin(), send_sizes.end(),
                     send_disp.begin() + 1);
    std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                     recv_disp.begin() + 1);

    // Pack the data to send the owning rank
    std::vector<std::int64_t> send_data(send_disp.back());
    std::vector<std::int64_t> recv_data(recv_disp.back());
    std::vector<int> insert_pos = send_disp;
    auto cell_index_map = mesh->topology().index_map(tdim);
    for (std::size_t i = 0; i < int_vertices_global.size(); i++)
    {
      auto it = std::find(destinations.begin(), destinations.end(), owner[i]);
      assert(it != destinations.end());
      int p = std::distance(destinations.begin(), it);
      int& pos = insert_pos[p];
      send_data[pos++] = int_vertices_global[i];
    }

    // A rank in the neighborhood communicator can have no incoming or
    // outcoming edges. This may cause OpenMPI to crash. Workaround:
    if (send_sizes.empty())
      send_sizes.reserve(1);
    if (recv_sizes.empty())
      recv_sizes.reserve(1);

    MPI_Neighbor_alltoallv(send_data.data(), send_sizes.data(),
                           send_disp.data(), MPI_INT64_T, recv_data.data(),
                           recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                           reverse_com);

    {
      MPI_Comm forward_com
          = vertex_index_map->comm(common::IndexMap::Direction::forward);
      auto [sources, destinations] = dolfinx::MPI::neighbors(forward_com);
      std::map<std::int64_t, std::vector<int>> vertex_neighbour_map;
      for (std::size_t i = 0; i < recv_sizes.size(); i++)
      {
        for (int j = recv_disp[i]; j < recv_disp[i + 1]; j++)
          vertex_neighbour_map[recv_data[j]].push_back(i);
      }

      // Figure out how much data to send to each neighbor
      std::vector<int> send_sizes(destinations.size(), 0);
      std::vector<int> recv_sizes(sources.size(), 0);
      for (auto const& [vertex, neighbors] : vertex_neighbour_map)
        for (auto p : neighbors)
          send_sizes[p] += (2 + neighbors.size() + 1);

      MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                            MPI_INT, forward_com);

      // Prepare communication displacements
      std::vector<int> send_disp(destinations.size() + 1, 0);
      std::vector<int> recv_disp(sources.size() + 1, 0);
      std::partial_sum(send_sizes.begin(), send_sizes.end(),
                       send_disp.begin() + 1);
      std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                       recv_disp.begin() + 1);

      // Pack the data to send
      // [V1 3 P1 P2 P3 V2 2 P2 P3 ...]
      std::vector<std::int64_t> send_data(send_disp.back());
      std::vector<std::int64_t> recv_data(recv_disp.back());
      std::vector<int> insert_pos = send_disp;
      for (auto const& [vertex, neighbors] : vertex_neighbour_map)
      {
        for (auto p : neighbors)
        {
          send_data[insert_pos[p]++] = vertex;
          send_data[insert_pos[p]++] = neighbors.size() + 1;
          send_data[insert_pos[p]++] = mpi_rank;
          for (auto other : neighbors)
            send_data[insert_pos[p]++] = destinations[other];
        }
      }

      // A rank in the neighborhood communicator can have no incoming or
      // outcoming edges. This may cause OpenMPI to crash. Workaround:
      if (send_sizes.empty())
        send_sizes.reserve(1);
      if (recv_sizes.empty())
        recv_sizes.reserve(1);

      MPI_Neighbor_alltoallv(send_data.data(), send_sizes.data(),
                             send_disp.data(), MPI_INT64_T, recv_data.data(),
                             recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                             forward_com);

      auto cell_map = mesh->topology().index_map(tdim);
      auto vc = mesh->topology().connectivity(0, tdim);

      std::map<std::int64_t, std::vector<std::int32_t>> vertex_procs;
      for (auto it = recv_data.begin(); it < recv_data.end();)
      {
        const std::int64_t global_index = *it++;
        int num_procs = *it++;
        auto& processes = vertex_procs[global_index];
        std::copy_n(it, num_procs, std::back_inserter(processes));
        std::advance(it, num_procs);
      }

      std::vector<std::int32_t> num_dest(cell_map->size_local(), 1);
      for (auto const& [vertex, neighbors] : vertex_procs)
      {
        for (auto cell : vc->links(vertex))
          num_dest[cell] += neighbors.size();
      }

      std::vector<std::int32_t> offsets(num_dest.size() + 1);
      std::partial_sum(num_dest.begin(), num_dest.end(), offsets.begin() + 1);
      std::vector<std::int32_t> pos = offsets;
      std::vector<std::int32_t> data(offsets.back());

      for (auto const& [vertex, neighbors] : vertex_procs)
      {
        for (auto cell : vc->links(vertex))
        {
          std::copy(neighbors.begin(), neighbors.end(),
                    data.begin() + pos[cell]);
          pos[cell] += neighbors.size();
        }
      }

      graph::AdjacencyList<std::int32_t> dest(data, offsets);

      std::cout << dest.str();

      // std::vector<std::int32_t> local_vertices(global_vertices.size());
      // vertex_index_map->global_to_local(global_vertices, local_vertices);

      //
      // for (std::size_t i = 0; i < local_vertices.size(); i++)
      //   for (auto cell : vc->links(local_vertices[i]))
      //     if (cell < cell_index_map->size_local())
      //       num_dest[cell] += num_procs_vert[i];

      // std::vector<std::int32_t> offsets(num_dest.size() + 1);
      // std::partial_sum(num_dest.begin(), num_dest.end(), offsets.begin() +
      // 1); std::vector<std::int32_t> array(num_dest.back(), -1); auto pos = 0;
      // for (std::size_t i = 0; i < local_vertices.size(); i++)
      //   for (auto cell : vc->links(local_vertices[i]))

      // for

      // for (auto a : local_vertices)
      //   std::cout << a << " ";
    }

    // if (mpi_rank == 0)
    // {
    //   std::cout << std::endl;
    //   for (auto a : recv_data)
    //     std::cout << a << " ";

    //   auto vec = vertex_index_map->shared_indices();
    //   std::cout << std::endl;
    //   for (auto a : vec)
    //     std::cout << a << " ";
    // }

    //   std::vector<int> values(facet_indices.size(), 1);
    //   dolfinx::mesh::MeshTags<int> mt(mesh, tdim - 1, facet_indices,
    //   values);

    //   // Save solution in VTK format
    //   io::XDMFFile file(comm, "mesh.xdmf", "w");
    //   std::string geometry_xpath =
    //   "/Xdmf/Domain/Grid[@Name='mesh']/Geometry"; file.write_mesh(*mesh);
    //   file.write_meshtags(mt, geometry_xpath);
    // }
  }

  common::subsystem::finalize_mpi();
  return 0;
}
