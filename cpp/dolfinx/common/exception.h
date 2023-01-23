#pragma once

#include <stdexcept>

class dolfinx::runtime_error : public std::runtime_error
{
public:
  /// Create exception
  /// @param[in] message The error message
  runtime_error(const std::string& what_arg) : std::runtime_error(what_arg) {}
  runtime_error(const char* what_arg) : std::runtime_error(what_arg) {}
};