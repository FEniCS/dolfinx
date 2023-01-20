#pragma once

#include <stdexcept>

class DolfinXException : public std::runtime_error
{
public:
  /// Create exception
  /// @param[in] message The error message
  DolfinXException(const std::string& message)
      : std::runtime_error(message)
  {
  }
};
