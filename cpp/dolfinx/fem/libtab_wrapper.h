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
class LibtabElement
{
public:
  LibtabElement() {};
  virtual const Eigen::ArrayXXd& points() const;
  virtual std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const;
  virtual int block_size() const;
};

class WrappedLibtabElement : public LibtabElement
{
public:
  explicit WrappedLibtabElement(std::shared_ptr<const libtab::FiniteElement> libtab_element);

  const Eigen::ArrayXXd& points() const override;
  std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const override;

private:
  std::shared_ptr<const libtab::FiniteElement> _libtab_element;
};

class BlockedLibtabElement : public LibtabElement
{
public:
  explicit BlockedLibtabElement(std::shared_ptr<const libtab::FiniteElement> libtab_element, int block_size);

  const Eigen::ArrayXXd& points() const override;
  std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const override;
  int block_size() const override;

private:
  std::shared_ptr<const libtab::FiniteElement> _libtab_element;
  int _block_size;
};

class MixedLibtabElement : public LibtabElement
{
public:
  explicit MixedLibtabElement(std::vector<std::shared_ptr<const LibtabElement>> sub_elements);

  const Eigen::ArrayXXd& points() const override;
  std::vector<Eigen::ArrayXXd> tabulate(int nd, const Eigen::ArrayXXd& x) const override;

private:
  std::vector<std::shared_ptr<const LibtabElement>> _sub_elements;
  Eigen::ArrayXXd _points;
};

const std::shared_ptr<const LibtabElement> create_libtab_element(
    const ufc_finite_element& ufc_element);
} // namespace dolfix::fem
