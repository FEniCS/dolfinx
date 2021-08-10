// Copyright (C) 2021 Joseph Dean, Sarah Roggendorf
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#pragma once

#include "MeshTags.h"
#include "Topology.h"
#include <iostream>
#include <memory>

namespace dolfinx::mesh
{
class MeshView
{
public:
  // MeshView initializer
  MeshView(std::shared_ptr<const MeshTags<std::int32_t>> meshtag);

  // Return the MeshView topology
  std::shared_ptr<mesh::Topology> topology();
  // Return the full mesh
  std::shared_ptr<const Mesh> mesh();
  // Return dimension of Meshview Entities
  std::int32_t dim();
  // Return MeshView entities
  const std::vector<std::int32_t>& entities();

private:
  std::shared_ptr<const Mesh> _mesh;
  const std::vector<std::int32_t>& _entities;
  const std::int32_t _dim;
  std::shared_ptr<Topology> _topology;
};
} // namespace dolfinx::mesh
