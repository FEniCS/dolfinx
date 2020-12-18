// Copyright (C) 2020 Matthew Scroggs
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "libtab_wrapper.h"
#include <libtab.h>
#include <ufc.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
const std::shared_ptr<const LibtabElement> dolfinx::fem::create_libtab_element(
    const ufc_finite_element& ufc_element)
{
  const std::string family = ufc_element.family;

  if (family == "mixed element")
  {
    std::vector<std::shared_ptr<const LibtabElement>> sub_elements;
    for (int i = 0; i < ufc_element.num_sub_elements; ++i)
      sub_elements.push_back(create_libtab_element(*ufc_element.create_sub_element(i)));
    return std::make_shared<MixedLibtabElement>(MixedLibtabElement(sub_elements));
  }

  static const std::map<ufc_shape, std::string> ufc_to_cell
      = {{vertex, "point"},
         {interval, "interval"},
         {triangle, "triangle"},
         {tetrahedron, "tetrahedron"},
         {quadrilateral, "quadrilateral"},
         {hexahedron, "hexahedron"}};
  const std::string cell_shape = ufc_to_cell.at(ufc_element.cell_shape);

  std::shared_ptr<const libtab::FiniteElement> libtab_e
    = std::make_shared<const libtab::FiniteElement>(libtab::create_element(
          family, cell_shape, ufc_element.degree));

  if(ufc_element.block_size != 1)
    return std::make_shared<BlockedLibtabElement>(BlockedLibtabElement(libtab_e, ufc_element.block_size));

  return std::make_shared<WrappedLibtabElement>(WrappedLibtabElement(libtab_e));
}
//-----------------------------------------------------------------------------
const std::shared_ptr<const LibtabElement>
dolfinx::fem::create_libtab_element(const ufc_coordinate_mapping& ufc_cmap)
{
  const std::string family = ufc_cmap.element_family;

  static const std::map<ufc_shape, std::string> ufc_to_cell
      = {{vertex, "point"},
         {interval, "interval"},
         {triangle, "triangle"},
         {tetrahedron, "tetrahedron"},
         {quadrilateral, "quadrilateral"},
         {hexahedron, "hexahedron"}};
  const std::string cell_shape = ufc_to_cell.at(ufc_cmap.cell_shape);

  std::shared_ptr<const libtab::FiniteElement> libtab_e
      = std::make_shared<const libtab::FiniteElement>(
          libtab::create_element(family, cell_shape, ufc_cmap.element_degree));

  //  if(block_size != 1)
  //    return
  //    std::make_shared<BlockedLibtabElement>(BlockedLibtabElement(libtab_e,
  //    block_size));

  return std::make_shared<WrappedLibtabElement>(WrappedLibtabElement(libtab_e));
}
//-----------------------------------------------------------------------------
const Eigen::ArrayXXd& LibtabElement::points() const
{
  throw std::runtime_error("points not implemented for this element");
}
//-----------------------------------------------------------------------------
std::vector<Eigen::ArrayXXd> LibtabElement::tabulate(int nd, const Eigen::ArrayXXd& x) const
{
  assert(nd >= 0);
  assert(x.cols() > 0);
  throw std::runtime_error("tabulate not implemented for this element");
}
//-----------------------------------------------------------------------------
int LibtabElement::block_size() const { return 1; }
//-----------------------------------------------------------------------------
int LibtabElement::degree() const
{
  throw std::runtime_error("degree not implemented for this element");
}
//-----------------------------------------------------------------------------
libtab::cell::type LibtabElement::cell_type() const
{
  throw std::runtime_error("degree not implemented for this element");
}
//-----------------------------------------------------------------------------
mesh::CellType LibtabElement::dolfinx_cell_type() const
{
  switch (cell_type())
  {
  case libtab::cell::type::interval:
    return mesh::CellType::interval;
  case libtab::cell::type::triangle:
    return mesh::CellType::triangle;
  case libtab::cell::type::quadrilateral:
    return mesh::CellType::quadrilateral;
  case libtab::cell::type::hexahedron:
    return mesh::CellType::hexahedron;
  case libtab::cell::type::tetrahedron:
    return mesh::CellType::tetrahedron;
  default:
    throw std::runtime_error("Invalid cell shape in CoordinateElement");
  }
}
//-----------------------------------------------------------------------------
int LibtabElement::topological_dimension() const
{
  return libtab::cell::topological_dimension(cell_type());
}
//-----------------------------------------------------------------------------
WrappedLibtabElement::WrappedLibtabElement(
  std::shared_ptr<const libtab::FiniteElement> libtab_element)
 : _libtab_element(libtab_element)
{}
//-----------------------------------------------------------------------------
const Eigen::ArrayXXd& WrappedLibtabElement::points() const
{
  return _libtab_element->points();
}
//-----------------------------------------------------------------------------
std::vector<Eigen::ArrayXXd> WrappedLibtabElement::tabulate(int nd, const Eigen::ArrayXXd& x) const
{
  return _libtab_element->tabulate(nd, x);
}
//-----------------------------------------------------------------------------
int WrappedLibtabElement::degree() const { return _libtab_element->degree(); }
//-----------------------------------------------------------------------------
libtab::cell::type WrappedLibtabElement::cell_type() const
{
  return _libtab_element->cell_type();
}
//-----------------------------------------------------------------------------
BlockedLibtabElement::BlockedLibtabElement(
  std::shared_ptr<const libtab::FiniteElement> libtab_element, int block_size)
 : _libtab_element(libtab_element), _block_size(block_size)
{}
//-----------------------------------------------------------------------------
const Eigen::ArrayXXd& BlockedLibtabElement::points() const
{
  return _libtab_element->points();
}
//-----------------------------------------------------------------------------
std::vector<Eigen::ArrayXXd> BlockedLibtabElement::tabulate(int nd, const Eigen::ArrayXXd& x) const
{
  return _libtab_element->tabulate(nd, x);
}
//-----------------------------------------------------------------------------
int BlockedLibtabElement::block_size() const { return _block_size; }
//-----------------------------------------------------------------------------
int BlockedLibtabElement::degree() const { return _libtab_element->degree(); }
//-----------------------------------------------------------------------------
libtab::cell::type BlockedLibtabElement::cell_type() const
{
  return _libtab_element->cell_type();
}
//-----------------------------------------------------------------------------
MixedLibtabElement::MixedLibtabElement(
  std::vector<std::shared_ptr<const LibtabElement>> sub_elements)
 : _sub_elements(sub_elements)
{
  int point_count = 0;
  int point_dim = 0;
  for (std::size_t i = 0; i < _sub_elements.size(); ++i)
  {
    const Eigen::ArrayXXd& subpoints = _sub_elements[i]->points();
    point_count += subpoints.rows() * _sub_elements[i]->block_size();
    if (i == 0)
      point_dim = subpoints.cols();
    else
      assert(point_dim == subpoints.cols());
  }

  _points.resize(point_count, point_dim);
  point_count = 0;
  for (std::size_t i = 0; i < _sub_elements.size(); ++i)
  {
    const Eigen::ArrayXXd& subpoints = _sub_elements[i]->points();
    const int bs = _sub_elements[i]->block_size();
    if (bs == 1)
      _points.block(point_count, 0, subpoints.rows(), point_dim) = subpoints;
    else
    {
      for(int j = 0; j < subpoints.rows(); j++)
        for(int k=0; k < bs; ++k)
          _points.block(point_count + j * bs + k, 0, 1, point_dim) = subpoints.row(j);
    }
    point_count += subpoints.rows() * bs;
  }
}
//-----------------------------------------------------------------------------
const Eigen::ArrayXXd& MixedLibtabElement::points() const
{
  return _points;
}
//-----------------------------------------------------------------------------
std::vector<Eigen::ArrayXXd> MixedLibtabElement::tabulate(int nd, const Eigen::ArrayXXd& x) const
{
  assert(nd >= 0);
  assert(x.cols() > 0);
  throw std::runtime_error("tabulate not implemented for this element");
}
//-----------------------------------------------------------------------------
int MixedLibtabElement::degree() const
{
  int out = 0;
  for (std::size_t i = 0; i < _sub_elements.size(); ++i)
    out = std::max(out, _sub_elements[i]->degree());
  return out;
}
//-----------------------------------------------------------------------------
libtab::cell::type MixedLibtabElement::cell_type() const
{
  return _sub_elements[0]->cell_type();
}
//-----------------------------------------------------------------------------
