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
/// TODO: document
class LibtabElement
{
public:
  /// TODO: document
  LibtabElement() {};

  /// TODO: document
  virtual const Eigen::ArrayXXd& points() const;

  /// TODO: document
  virtual std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const;

  /// TODO: document
  virtual int block_size() const;
};

/// TODO: document
class WrappedLibtabElement : public LibtabElement
{
public:
  /// TODO: document
  explicit WrappedLibtabElement(std::shared_ptr<const libtab::FiniteElement> libtab_element);

  /// TODO: document
  const Eigen::ArrayXXd& points() const override;

  /// TODO: document
  std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const override;

private:
  /// TODO: document
  std::shared_ptr<const libtab::FiniteElement> _libtab_element;
};

/// TODO: document
class BlockedLibtabElement : public LibtabElement
{
public:
  /// TODO: document
  explicit BlockedLibtabElement(std::shared_ptr<const libtab::FiniteElement> libtab_element, int block_size);

  /// TODO: document
  const Eigen::ArrayXXd& points() const override;

  /// TODO: document
  std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const override;

  /// TODO: document
  int block_size() const override;

private:
  /// TODO: document
  std::shared_ptr<const libtab::FiniteElement> _libtab_element;

  /// TODO: document
  int _block_size;
};

/// TODO: document
class MixedLibtabElement : public LibtabElement
{
public:
  /// TODO: document
  explicit MixedLibtabElement(std::vector<std::shared_ptr<const LibtabElement>> sub_elements);

  /// TODO: document
  const Eigen::ArrayXXd& points() const override;

  /// TODO: document
  std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const override;

private:
  /// TODO: document
  std::vector<std::shared_ptr<const LibtabElement>> _sub_elements;

  /// TODO: document
  Eigen::ArrayXXd _points;
};

const std::shared_ptr<const LibtabElement> create_libtab_element(
    const ufc_finite_element& ufc_element);
} // namespace dolfix::fem
