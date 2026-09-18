# Provides dolfinx_enable_clang_tidy(<target> <config-file>), which runs
# clang-tidy over the target's C++ sources when ENABLE_CLANG_TIDY is set.
#
# The .clang-tidy configuration is not installed with DOLFINx, so the
# caller passes the path to the one it wants to use.

include_guard(GLOBAL)

function(dolfinx_enable_clang_tidy target config_file)
  if(NOT ENABLE_CLANG_TIDY)
    return()
  endif()
  find_program(CLANG_TIDY NAMES clang-tidy REQUIRED)
  set_target_properties(
    ${target}
    PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY};--config-file=${config_file}"
  )
endfunction()
