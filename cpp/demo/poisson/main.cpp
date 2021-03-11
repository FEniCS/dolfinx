#include "poisson.h"
#include <cmath>
#include <dolfinx.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/MeshTags.h>

using namespace dolfinx;

int main(int argc, char* argv[])
{
  common::subsystem::init_logging(argc, argv);
  common::subsystem::init_mpi(argc, argv);

  {
    MPI_Comm comm = MPI_COMM_WORLD;
    int mpi_rank;
    MPI_Comm_rank(comm, &mpi_rank);
    int mpi_size;
    MPI_Comm_size(comm, &mpi_size);

    // Create mesh and function space
    auto cmap = fem::create_coordinate_map(create_coordinate_map_poisson);
    auto mesh = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
        comm, {{{0.0, 0.0, 0.0}, {5.0, 1.0, 1.0}}}, {500, 100, 100}, cmap,
        mesh::GhostMode::none));

    int tdim = mesh->topology().dim();
    mesh->topology().create_connectivity(tdim - 1, tdim);
    mesh->topology().create_connectivity(tdim - 1, 0);
    mesh->topology().create_connectivity(0, tdim);

    auto fv = mesh->topology().connectivity(tdim - 1, 0);
    auto vc = mesh->topology().connectivity(0, tdim);
    auto vert_map = mesh->topology().index_map(0);

    // Data to used in step 2.
    std::vector<std::int64_t> recv_data;
    std::vector<int> recv_sizes;
    std::vector<int> recv_disp;
    std::vector<std::int32_t> facet_indices;

    // Step 1: Identify boundary entities and send information to the entity
    // owner.
    {
      std::vector<bool> bnd_facets
          = dolfinx::mesh::compute_interface_facets(mesh->topology());
      // Get indices of interface facets
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
      std::int32_t local_size = vert_map->size_local();
      std::sort(int_vertices.begin(), int_vertices.end());
      int_vertices.erase(std::unique(int_vertices.begin(), int_vertices.end()),
                         int_vertices.end());
      int_vertices.erase(std::remove_if(int_vertices.begin(),
                                        int_vertices.end(),
                                        [local_size](std::int32_t v) {
                                          return v < local_size;
                                        }),
                         int_vertices.end());

      // Compute global indices for interface vertices
      std::vector<std::int64_t> int_vertices_global(int_vertices.size());
      vert_map->local_to_global(int_vertices, int_vertices_global);

      // Get owners of each interface vertex
      auto ghost_owners = vert_map->ghost_owner_rank();
      auto ghosts = vert_map->ghosts();
      std::vector<std::int32_t> owner(int_vertices_global.size());
      // FIXME: This could be faster if ghosts were sorted
      for (std::size_t i = 0; i < int_vertices_global.size(); i++)
      {
        std::int64_t ghost = int_vertices_global[i];
        auto it = std::find(ghosts.begin(), ghosts.end(), ghost);
        assert(it != ghosts.end());
        int pos = std::distance(ghosts.begin(), it);
        owner[i] = ghost_owners[pos];
      }

      // Each process reports to the owners of the vertices it has on
      // its boundary. reverse_comm,: Ghost -> owner communication

      // Figure out how much data to send to each neighbor (ghost owner).
      MPI_Comm reverse_comm
          = vert_map->comm(common::IndexMap::Direction::reverse);
      auto [sources, destinations] = dolfinx::MPI::neighbors(reverse_comm);
      std::vector<int> send_sizes(destinations.size(), 0);
      recv_sizes.resize(sources.size(), 0);
      for (std::size_t i = 0; i < int_vertices_global.size(); i++)
      {
        auto it = std::find(destinations.begin(), destinations.end(), owner[i]);
        assert(it != destinations.end());
        int pos = std::distance(destinations.begin(), it);
        send_sizes[pos]++;
      }
      MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                            MPI_INT, reverse_comm);

      // Prepare communication displacements
      std::vector<int> send_disp(destinations.size() + 1, 0);
      recv_disp.resize(sources.size() + 1, 0);
      std::partial_sum(send_sizes.begin(), send_sizes.end(),
                       send_disp.begin() + 1);
      std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                       recv_disp.begin() + 1);

      // Pack the data to send the owning rank:
      // Each process send its interface vertices to the respective
      // owner, which will be the "match-maker".
      std::vector<std::int64_t> send_data(send_disp.back());
      recv_data.resize(recv_disp.back());
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
                             reverse_comm);
    }

    // Step 2: Each process now has a list of all processes for which one of its
    // owned vertices is an interface vertice. Gather information an send the
    // list to all processes that share the same vertice.
    {
      MPI_Comm forward_comm
          = vert_map->comm(common::IndexMap::Direction::forward);
      auto [sources, destinations] = dolfinx::MPI::neighbors(forward_comm);

      // Pack information into a more manageable format
      std::map<std::int64_t, std::vector<int>> vertex_procs_owned;
      for (std::size_t i = 0; i < recv_sizes.size(); i++)
        for (int j = recv_disp[i]; j < recv_disp[i + 1]; j++)
          vertex_procs_owned[recv_data[j]].push_back(i);

      std::vector<std::int64_t> global_indices_owned(vertex_procs_owned.size());
      std::transform(vertex_procs_owned.begin(), vertex_procs_owned.end(),
                     global_indices_owned.begin(),
                     [](auto& pair) { return pair.first; });
      std::vector<std::int32_t> local_indices_owned(
          global_indices_owned.size());
      vert_map->global_to_local(global_indices_owned, local_indices_owned);

      // Figure out how much data to send to each neighbor
      // For every shared vertice we send:
      // [Global index, Number of Processes, P1, ...,  PN]
      std::vector<int> send_sizes(destinations.size(), 0);
      recv_sizes.resize(sources.size(), 0);
      for (auto const& [vertex, neighbors] : vertex_procs_owned)
        for (auto p : neighbors)
          send_sizes[p] += (2 + neighbors.size() + 1);

      MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1,
                            MPI_INT, forward_comm);

      // Prepare communication displacements
      std::vector<int> send_disp(destinations.size() + 1, 0);
      recv_disp.resize(sources.size() + 1, 0);
      std::partial_sum(send_sizes.begin(), send_sizes.end(),
                       send_disp.begin() + 1);
      std::partial_sum(recv_sizes.begin(), recv_sizes.end(),
                       recv_disp.begin() + 1);

      // Pack the data to send: EG
      // [V100 3 P1 P2 P3 V2 2 P2 P3 ...]
      std::vector<std::int64_t> send_data(send_disp.back());
      std::vector<std::int64_t> recv_data(recv_disp.back());
      std::vector<int> insert_pos = send_disp;
      for (auto const& [vertex, neighbors] : vertex_procs_owned)
      {
        for (auto p : neighbors)
        {
          send_data[insert_pos[p]++] = vertex;
          // Should include this process to the list (+1) as the
          // vertex owner.
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

      // Send packaged information only to relevant neighbors
      MPI_Neighbor_alltoallv(send_data.data(), send_sizes.data(),
                             send_disp.data(), MPI_INT64_T, recv_data.data(),
                             recv_sizes.data(), recv_disp.data(), MPI_INT64_T,
                             forward_comm);

      // Unpack data and create a more manageable structure
      auto cell_map = mesh->topology().index_map(tdim);
      auto vc = mesh->topology().connectivity(0, tdim);
      std::map<std::int64_t, std::vector<std::int32_t>> vertex_procs_ghost;
      for (auto it = recv_data.begin(); it < recv_data.end();)
      {
        const std::int64_t global_index = *it++;
        int num_procs = *it++;
        auto& processes = vertex_procs_ghost[global_index];
        std::copy_n(it, num_procs, std::back_inserter(processes));
        std::advance(it, num_procs);
      }

      std::vector<std::int64_t> global_indices(vertex_procs_ghost.size());
      std::transform(vertex_procs_ghost.begin(), vertex_procs_ghost.end(),
                     global_indices.begin(),
                     [](auto& pair) { return pair.first; });

      std::vector<std::int32_t> local_indices(global_indices.size());
      vert_map->global_to_local(global_indices, local_indices);

      // Start getting the destination of local cells
      std::int32_t num_local_cells = cell_map->size_local();
      std::vector<std::int32_t> num_dest(num_local_cells, 1);

      // merge maps
      {
        int pos = 0;
        for (auto const& [vertex, neighbors] : vertex_procs_ghost)
        {
          for (auto cell : vc->links(local_indices[pos++]))
            num_dest[cell] += neighbors.size();
        }

        pos = 0;
        for (auto const& [vertex, neighbors] : vertex_procs_owned)
        {
          for (auto cell : vc->links(local_indices_owned[pos++]))
            num_dest[cell] += neighbors.size() + 1;
        }
      }

      std::vector<std::int32_t> offsets(num_dest.size() + 1);
      std::partial_sum(num_dest.begin(), num_dest.end(), offsets.begin() + 1);
      std::vector<std::int32_t> pos = offsets;
      std::vector<std::int32_t> data(offsets.back(), mpi_rank);
      {
        // Merge maps?
        int j = 0;
        for (auto const& [vertex, neighbors] : vertex_procs_ghost)
        {
          for (auto cell : vc->links(local_indices[j++]))
          {
            if (cell < num_local_cells)
            {
              std::copy(neighbors.begin(), neighbors.end(),
                        data.begin() + pos[cell]);
              pos[cell] += neighbors.size();
            }
          }
        }

        j = 0;
        for (auto const& [vertex, neighbors] : vertex_procs_owned)
        {
          for (auto cell : vc->links(local_indices_owned[j++]))
          {
            if (cell < num_local_cells)
            {
              for (auto n : neighbors)
                data[pos[cell]++] = destinations[n];
              data[pos[cell]++] = mpi_rank;
            }
          }
        }
      }
      graph::AdjacencyList<std::int32_t> dest_duplicates(data, offsets);

      // Remove duplicates entries in the destination Adjacency List
      std::vector<int> counter(num_local_cells, 0);
      std::vector<std::int32_t> cell_data;
      for (std::int32_t c = 0; c < num_local_cells; c++)
      {
        // unordered_set is potentially faster, but data is not ordered
        std::set<std::int32_t> local_set(dest_duplicates.links(c).begin(),
                                         dest_duplicates.links(c).end());
        local_set.erase(mpi_rank);
        cell_data.push_back(mpi_rank);
        cell_data.insert(cell_data.end(), local_set.begin(), local_set.end());
        counter[c] = local_set.size() + 1;
      }

      std::vector<std::int32_t> new_offsets(counter.size() + 1, 0);
      std::partial_sum(counter.begin(), counter.end(), new_offsets.begin() + 1);
      graph::AdjacencyList<std::int32_t> dest(cell_data, new_offsets);
      /*  [[maybe_unused]] auto partitioner = [&dest](...) { return dest; }; */

      [[maybe_unused]] auto partitioner = [&dest](...) { return dest; };

      auto cv = mesh->topology().connectivity(tdim, 0);
      {
        int num_cells = cell_map->size_local() + cell_map->num_ghosts();
        std::vector<std::int32_t> vertex_to_x(vert_map->size_local()
                                              + vert_map->num_ghosts());
        for (int c = 0; c < num_cells; ++c)
        {
          auto vertices = cv->links(c);
          auto dofs = mesh->geometry().dofmap().links(c);
          for (std::size_t i = 0; i < vertices.size(); ++i)
            vertex_to_x[vertices[i]] = dofs[i];
        }

        auto& local = cv->array();
        std::vector<std::int64_t> global(local.size());
        vert_map->local_to_global(local, global);
        std::vector<std::int64_t> global_array;
        std::vector<int> counter(num_local_cells);
        for (std::int32_t i = 0; i < num_local_cells; i++)
        {
          const auto& local = cv->links(i);
          std::vector<int64_t> global(local.size());
          vert_map->local_to_global(local, global);
          global_array.insert(global_array.end(), global.begin(), global.end());
          counter[i] += global.size();
        }
        std::vector<std::int32_t> offsets(counter.size() + 1, 0);
        std::partial_sum(counter.begin(), counter.end(), offsets.begin() + 1);

        graph::AdjacencyList<std::int64_t> cell_vertices(global_array, offsets);

        // Copy over existing mesh vertices
        const std::int32_t num_vertices = vert_map->size_local();
        const array2d<double>& x_g = mesh->geometry().x();
        array2d<double> x(num_vertices, x_g.shape[1]);
        for (int v = 0; v < num_vertices; ++v)
          for (std::size_t j = 0; j < x_g.shape[1]; ++j)
            x(v, j) = x_g(vertex_to_x[v], j);

        std::cout << "\n======================\n";

        /*         auto partitioner =
           static_cast<graph::AdjacencyList<std::int32_t> (*)( MPI_Comm, int,
           const mesh::CellType, const graph::AdjacencyList<std::int64_t>&,
           mesh::GhostMode)>( &mesh::partition_cells_graph); */

        auto new_mesh = std::make_shared<mesh::Mesh>(mesh::create_mesh(
            mesh->mpi_comm(), cell_vertices, mesh->geometry().cmap(), x,
            mesh::GhostMode::shared_facet, partitioner));

        auto V = fem::create_functionspace(create_functionspace_form_poisson_a,
                                           "u", new_mesh);

        auto u = std::make_shared<fem::Function<PetscScalar>>(V);

        auto& vector = u->x()->mutable_array();

        int sz = u->x()->map()->size_local();
        std::fill(vector.begin(), vector.begin() + sz, mpi_rank + 1);
        std::fill(vector.begin() + sz, vector.end(), 0);

        std::cout << vector.size() << " ";
        la::scatter_rev(*u->x(), common::IndexMap::Mode::insert);

        int tdim = new_mesh->topology().dim();
        new_mesh->topology().create_connectivity(tdim - 1, tdim);
        new_mesh->topology().create_connectivity(tdim - 1, 0);
        new_mesh->topology().create_connectivity(0, tdim);

        std::vector<bool> bnd_facets
            = dolfinx::mesh::compute_interface_facets(new_mesh->topology());
        // Get indices of interface facets
        for (std::size_t f = 0; f < bnd_facets.size(); ++f)
          if (bnd_facets[f])
            facet_indices.push_back(f);

        // Save solution in VTK format
        io::XDMFFile file(comm, "mesh.xdmf", "w");
        std::string geometry_xpath = "/Xdmf/Domain/Grid[@Name='mesh']/Geometry";
        file.write_mesh(*new_mesh);
        file.write_function(*u, 0, "/Xdmf/Domain/Grid[@Name='mesh']");
      }
    }
  }

  common::subsystem::finalize_mpi();
  return 0;
}
