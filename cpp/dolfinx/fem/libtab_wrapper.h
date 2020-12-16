// Copyright (C) 2020 Matthew Scroggs
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/types.h>
#include <dolfinx/mesh/cell_types.h>
#include <memory>
#include <vector>
#include <Eigen/Dense>

struct ufc_finite_element;

namespace libtab
{
class FiniteElement;
}

namespace dolfinx::fem
{
/// A wrapper class for libtab. This allows mixed, vector, and other elements to be handled
/// more easily by dolfin.
class LibtabElement
{
public:
  /// Constructor
  LibtabElement() {};

  /// Destructor
  virtual ~LibtabElement() = default;

  /// Wrapper for libtab points
  virtual const Eigen::ArrayXXd& points() const;

  /// Wrapper for libtab tabulate
  virtual std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const;

  /// The block size of the element
  virtual int block_size() const;
};

/// Subclass that wraps an element implemented in libtab
class WrappedLibtabElement : public LibtabElement
{
public:
  /// Constructor
  explicit WrappedLibtabElement(std::shared_ptr<const libtab::FiniteElement> libtab_element);

  /// Wrapper for libtab points
  const Eigen::ArrayXXd& points() const override;

  /// Wrapper for libtab tabulate
  std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const override;

private:
  /// The libtab element being wrapped
  std::shared_ptr<const libtab::FiniteElement> _libtab_element;
};

/// A blocked element (vector or tensor element)
class BlockedLibtabElement : public LibtabElement
{
public:
  /// Constructor
  explicit BlockedLibtabElement(std::shared_ptr<const libtab::FiniteElement> libtab_element, int block_size);

  /// Wrapper for libtab points
  const Eigen::ArrayXXd& points() const override;

  /// Wrapper for libtab tabulate
  std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const override;

  /// The block size of the element
  int block_size() const override;

private:
  /// The scalar element
  std::shared_ptr<const libtab::FiniteElement> _libtab_element;

  /// The block size of the element
  int _block_size;
};

/// A mixed element (combination of two or more elements)
class MixedLibtabElement : public LibtabElement
{
public:
  /// Constructor
  explicit MixedLibtabElement(std::vector<std::shared_ptr<const LibtabElement>> sub_elements);

  /// Wrapper for libtab points
  const Eigen::ArrayXXd& points() const override;

  /// Wrapper for libtab tabulate
  std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const override;

private:
  /// The subelements
  std::vector<std::shared_ptr<const LibtabElement>> _sub_elements;

  /// The points
  Eigen::ArrayXXd _points;
};

/// Create a libtab element from a ufc element
const std::shared_ptr<const LibtabElement> create_libtab_element(
    const ufc_finite_element& ufc_element);
} // namespace dolfix::fem
