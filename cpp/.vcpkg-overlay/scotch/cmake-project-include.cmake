# Scotch sets no WINDOWS_EXPORT_ALL_SYMBOLS/dllexport annotations itself,
# so DLLs would export nothing. Its public API is function-only (no
# exported data symbols), so auto-exporting everything is sufficient.
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if(WIN32)
  # scotch/ptscotch link PUBLIC against ${LIBSCOTCHERR}/${LIBPTSCOTCHERR}
  # (scotcherr/ptscotcherr) via a forward-referenced target name defined
  # later in the same CMakeLists.txt. That does not reliably become a
  # Ninja build-order edge for the two-pass MSVC DLL link
  # (cmake -E vs_link_dll), so scotch.dll/ptscotch.dll can start linking
  # before scotcherr.lib/ptscotcherr.lib exist (LNK1104). Force the
  # ordering explicitly once all targets exist.
  macro(_scotch_force_link_order)
    if(TARGET scotch AND TARGET scotcherr)
      add_dependencies(scotch scotcherr)
    endif()
    if(TARGET ptscotch AND TARGET ptscotcherr)
      add_dependencies(ptscotch ptscotcherr)
    endif()
  endmacro()
  cmake_language(DEFER CALL _scotch_force_link_order)
endif()
