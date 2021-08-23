// Copyright (C) 2021 Jorgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "MeshView.h"

using namespace dolfinx;
using namespace dolfinx::mesh;

//-----------------------------------------------------------------------------
MeshView::MeshView(std::shared_ptr<const Mesh> parent_mesh, int dim,
                   tcb::span<std::int32_t> entities)
    : _parent_mesh(parent_mesh){};