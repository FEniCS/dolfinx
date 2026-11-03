// Copyright (C) 2026 Wang Guanyao
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace dolfinx::io
{

/// @brief Mode used to open a file.
enum class FileMode : std::int8_t
{
  read,  ///< Open an existing file for reading.
  write, ///< Create a new file, replacing any existing file.
  append ///< Open an existing file for appending, or create it.
};

/// @brief Convert a file mode to its standard one-character representation.
/// @param[in] mode File mode.
/// @return The corresponding `fopen`-style mode string.
constexpr std::string_view file_mode_to_string(FileMode mode) noexcept
{
  switch (mode)
  {
  case FileMode::read:
    return "r";
  case FileMode::write:
    return "w";
  case FileMode::append:
    return "a";
  }

  return {};
}

/// @brief Convert a standard one-character mode to a `FileMode`.
/// @param[in] mode `fopen`-style mode string.
/// @return The corresponding file mode.
/// @throws std::invalid_argument if @p mode is not `r`, `w` or `a`.
inline FileMode file_mode_from_string(std::string_view mode)
{
  if (mode == "r")
    return FileMode::read;
  if (mode == "w")
    return FileMode::write;
  if (mode == "a")
    return FileMode::append;

  throw std::invalid_argument("Unknown file mode. Use \"r\", \"w\" or \"a\".");
}

} // namespace dolfinx::io
