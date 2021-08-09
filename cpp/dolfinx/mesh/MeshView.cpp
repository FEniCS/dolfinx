// Copyright (C) 2021 Joseph Dean, Sarah Roggendorf
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
#include "MeshView.h"


using namespace dolfinx;
using namespace dolfinx::mesh;

MeshView::MeshView(std::shared_ptr<const MeshTags<std::int32_t>> meshtag)
    : _mesh(meshtag->mesh()), _indices(meshtag->indices()), _dim(meshtag->dim())
{
// Do nothing for now

}