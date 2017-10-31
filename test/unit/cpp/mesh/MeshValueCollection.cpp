// Copyright (C) 2007 Anders Logg
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
// Unit tests for the mesh library

#include <dolfin.h>
#include <catch.hpp>

using namespace dolfin;

TEST_CASE("MeshValueCollections")
{
  SECTION("Test assign 2D cells")
  {
    auto mesh = std::make_shared<UnitSquareMesh>(3, 3);
    const std::size_t ncells = mesh->num_cells();
    MeshValueCollection<int> f(mesh, 2);
    bool all_new = true;
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      bool this_new;
      const int value = ncells - cell->index();
      this_new = f.set_value(cell->index(), value);
      all_new = all_new && this_new;
    }
    MeshValueCollection<int> g(mesh, 2);
    g = f;
    CHECK(ncells == f.size());
    CHECK(ncells == g.size());
    CHECK(all_new);
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      const int value = ncells - cell->index();
      CHECK(value == g.get_value(cell->index(), 0));
    }
  }

  SECTION("Test assign 2D facets")
  {
    auto mesh = std::make_shared<UnitSquareMesh>(3, 3);
    mesh->init(2,1);
    const std::size_t ncells = mesh->num_cells();
    MeshValueCollection<int> f(mesh, 1);
    bool all_new = true;
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      const int value = ncells - cell->index();
      for (std::size_t i = 0; i < cell->num_entities(1); ++i)
      {
        bool this_new;
        this_new = f.set_value(cell->index(), i, value + i);
        all_new = all_new && this_new;
      }
    }
    MeshValueCollection<int> g(mesh, 1);
    g = f;
    CHECK(ncells*3 == f.size());
    CHECK(ncells*3 == g.size());
    CHECK(all_new);
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      for (std::size_t i = 0; i < cell->num_entities(1); ++i)
      {
        const int value = ncells - cell->index() + i;
        CHECK(value == g.get_value(cell->index(), i));
      }
    }
  }

  SECTION("Test assign 2D vertices")
  {
    auto mesh = std::make_shared<UnitSquareMesh>(3, 3);
    mesh->init(2, 0);
    const std::size_t ncells = mesh->num_cells();
    MeshValueCollection<int> f(mesh, 0);
    bool all_new = true;
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      const int value = ncells - cell->index();
      for (std::size_t i = 0; i < cell->num_entities(0); ++i)
      {
        bool this_new;
        this_new = f.set_value(cell->index(), i, value+i);
        all_new = all_new && this_new;
      }
    }
    MeshValueCollection<int> g(mesh, 0);
    g = f;
    CHECK(ncells*3 == f.size());
    CHECK(ncells*3 == g.size());
    CHECK(all_new);
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      for (std::size_t i = 0; i < cell->num_entities(0); ++i)
      {
        const int value = ncells - cell->index() + i;
        CHECK(value == g.get_value(cell->index(), i));
      }
    }
  }

  SECTION("Test MeshFunction assign 2D cells")
  {
    auto mesh = std::make_shared<UnitSquareMesh>(3, 3);
    const std::size_t ncells = mesh->num_cells();
    MeshFunction<int> f(mesh, 2, 0);
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
      f[cell->index()] = ncells - cell->index();
    MeshValueCollection<int> g(mesh, 2);
    g = f;
    CHECK(ncells == f.size());
    CHECK(ncells == g.size());
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      const int value = ncells - cell->index();
      CHECK(value == g.get_value(cell->index(), 0));
    }
  }

  SECTION("Test MeshFunction assign 2D facets")
  {
    auto mesh = std::make_shared<UnitSquareMesh>(3, 3);
    mesh->init(1);
    MeshFunction<int> f(mesh, 1, 25);
    MeshValueCollection<int> g(mesh, 1);
    g = f;
    CHECK(mesh->num_facets() == f.size());
    CHECK(mesh->num_cells()*3 == g.size());
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      for (std::size_t i = 0; i < cell->num_entities(1); ++i)
        CHECK(25 == g.get_value(cell->index(), i));
    }
  }

  SECTION("Test MeshFunction assign 2D vertices")
  {
    auto mesh = std::make_shared<UnitSquareMesh>(3, 3);
    mesh->init(0);
    MeshFunction<int> f(mesh, 0, 25);
    MeshValueCollection<int> g(mesh, 0);
    g = f;
    CHECK(mesh->num_vertices() == f.size());
    CHECK(mesh->num_cells()*3 == g.size());
    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
      for (std::size_t i = 0; i < cell->num_entities(0); ++i)
        CHECK(25 == g.get_value(cell->index(), i));
    }
  }
}
