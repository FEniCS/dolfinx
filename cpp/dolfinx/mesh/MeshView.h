// Copyright (C) 2021 Joseph Dean, Sarah Roggendorf
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#pragma once

#include<iostream>
#include<memory>
#include "MeshTags.h"

namespace dolfinx::mesh
{
    class MeshView
    {
        public:
          MeshView(std::shared_ptr<const MeshTags<std::int32_t>> meshtag);

        private:
          std::shared_ptr<const Mesh> _mesh;
          const std::vector<std::int32_t>& _indices;
          const std::int32_t _dim;
    };
}
