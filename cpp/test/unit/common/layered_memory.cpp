// Copyright (C) 2020 Matthias Rambausel
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <catch.hpp>
#include <dolfinx/common/Memory.h>

using namespace dolfinx;

namespace
{

struct StorageLayer
{
  int value = 0;
};

void test_memory_write_read_over_write_remove()
{
  using namespace dolfinx::common::memory;
  LayerManager<StorageLayer> layered_memory{false};

  // Create first layer and acquire lock
  LayerLock<StorageLayer> first_layer_lock{layered_memory.hold_layer()};

  // Write something
  auto write_value = [](StorageLayer& layer, int a) { layer.value = a; };
  const int target = 3;
  layered_memory.write(write_value, target);

  // Read what has been written
  int read_target = 0;
  auto read_value = [](const StorageLayer& layer, int& read_target) {
    read_target = layer.value;
    return true;
  };
  layered_memory.read(read_value, read_target);
  CHECK(read_target == target);

  auto find_value
      = [](const StorageLayer& layer, const int searched_for, int& layer_count) {
          ++layer_count;
          return layer.value == searched_for;
        };

  { // enter new scope
    // Create second layer above the first and acquire lock
    LayerLock<StorageLayer> second_layer_lock{layered_memory.hold_layer(true)};

    // Write something
    layered_memory.write(write_value, target * 2);

    // Search the for the former target (search starts from the top)
    int layer_count = 0;
    layered_memory.read(find_value, target, layer_count);
    CHECK(layer_count == 2);

    // Search the for the latter target (=2*target)
    layer_count = 0;
    layered_memory.read(find_value, target * 2, layer_count);
    CHECK(layer_count == 1);
  }

  // Now we should be back at only one layer
  int layer_count = 0;
  layered_memory.read(find_value, target, layer_count);
  CHECK(layer_count == 1);

  // Create a memory on top of the first and try to read from the background memory
  LayerManager<StorageLayer> layered_memory_2{false, &(layered_memory)};

  // Now we still have only one layer
  layer_count = 0;
  layered_memory_2.read(find_value, target, layer_count);
  CHECK(layer_count == 1);
}

} // namespace

TEST_CASE("Perform typical set of operations on a layered memory",
          "[memory_write_read_over_write_remove]")
{
  CHECK_NOTHROW(test_memory_write_read_over_write_remove());
}
