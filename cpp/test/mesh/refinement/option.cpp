// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include <dolfinx/refinement/option.h>

using namespace dolfinx::refinement;

TEST_CASE("Refinement Option", "refinement,option")
{
  // Internal binary structure
  CHECK(static_cast<std::uint8_t>(Option::none) == 0);
  CHECK(static_cast<std::uint8_t>(Option::parent_facet) == 1);
  CHECK(static_cast<std::uint8_t>(Option::parent_cell) == 2);
  CHECK(static_cast<std::uint8_t>(Option::parent_cell_and_facet) == 3);

  // Extraction of flags from possible multiple set flags
  CHECK(option_parent_cell(Option::none) == false);
  CHECK(option_parent_facet(Option::none) == false);

  CHECK(option_parent_cell(Option::parent_facet) == false);
  CHECK(option_parent_facet(Option::parent_facet) == true);

  CHECK(option_parent_cell(Option::parent_cell) == true);
  CHECK(option_parent_facet(Option::parent_cell) == false);

  CHECK(option_parent_cell(Option::parent_cell_and_facet) == true);
  CHECK(option_parent_facet(Option::parent_cell_and_facet) == true);

  // Logical combination of options
  CHECK((Option::none | Option::none) == Option::none);
  CHECK((Option::none | Option::parent_facet) == Option::parent_facet);
  CHECK((Option::none | Option::parent_cell) == Option::parent_cell);

  CHECK((Option::parent_cell_and_facet | Option::none)
        == Option::parent_cell_and_facet);
  CHECK((Option::parent_cell_and_facet | Option::parent_facet)
        == Option::parent_cell_and_facet);
  CHECK((Option::parent_cell_and_facet | Option::parent_cell)
        == Option::parent_cell_and_facet);

  CHECK((Option::parent_cell | Option::none) == Option::parent_cell);
  CHECK((Option::parent_cell | Option::parent_facet)
        == Option::parent_cell_and_facet);
}