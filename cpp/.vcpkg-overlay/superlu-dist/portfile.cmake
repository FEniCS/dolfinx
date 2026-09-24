# Windows/MSVC build fixes (wingetopt.c/.h CMakeLists.txt typos, guarding
# out unistd.h in util.c) are applied upstream of this port, on
# https://github.com/jhale/superlu_dist, branch jhale/windows-fixes; also
# submitted upstream as https://github.com/xiaoyeli/superlu_dist/pull/225.
vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO jhale/superlu_dist
  REF 592dbe8cc8db1366dd4ef940230ea191f64fafeb
  SHA512 3b8347af968052732b880f9adedc15ec3decb56afa9e81db349a5b3a5b0345f899d0059e8a87468bff36f0510818ec897f5b6627a8d984d4f09fc716d4c52a3a
  HEAD_REF jhale/windows-fixes
)

# SuperLU_DIST's ParMETIS TPL is satisfied by the ScotchParMETIS
# compatibility layer (scotch[metis,parmetis,ptscotch]): scotchmetisv5
# provides the METIS API used by SuperLU_DIST itself, and
# ptscotchparmetisv3 provides the ParMETIS API. TPL_PARMETIS_LIBRARIES is
# a raw (non-target) link line; scotch is built shared, so its own
# dependencies are pulled in transitively and don't need listing here.
if(VCPKG_TARGET_IS_WINDOWS)
  set(
    SUPERLU_DIST_SCOTCH_LIBRARY_NAMES
    ptscotchparmetisv3
    ptscotch
    ptscotcherr
    scotchmetisv5
    scotch
    scotcherr
  )
  # Debug/release scotch import libs share the same file names (no debug
  # postfix), but live under debug/lib and lib respectively, so the two
  # configs need distinct TPL_PARMETIS_LIBRARIES values. This must be a
  # proper CMake list (semicolon-separated): SuperLU_DIST's own CMakeLists
  # does set(PARMETIS_LIB ${TPL_PARMETIS_LIBRARIES}) unquoted, and CMake
  # only splits unquoted variable references on semicolons, never on
  # spaces. A space-joined string therefore collapses into a single list
  # element, and CMake's Ninja generator then emits that whole element as
  # one bogus combined implicit dependency, which ninja rejects with
  # "FindFirstFileExA(...): The filename ... is incorrect".
  set(SUPERLU_DIST_PARMETIS_LIBRARIES_DEBUG "")
  set(SUPERLU_DIST_PARMETIS_LIBRARIES_RELEASE "")
  foreach(
    SUPERLU_DIST_SCOTCH_LIBRARY_NAME
    IN
    LISTS SUPERLU_DIST_SCOTCH_LIBRARY_NAMES
  )
    list(
      APPEND SUPERLU_DIST_PARMETIS_LIBRARIES_DEBUG
      "${CURRENT_INSTALLED_DIR}/debug/lib/${SUPERLU_DIST_SCOTCH_LIBRARY_NAME}.lib"
    )
    list(
      APPEND SUPERLU_DIST_PARMETIS_LIBRARIES_RELEASE
      "${CURRENT_INSTALLED_DIR}/lib/${SUPERLU_DIST_SCOTCH_LIBRARY_NAME}.lib"
    )
  endforeach()
else()
  # No debug/release split needed: vcpkg's toolchain-managed library
  # search paths are already config-aware. A proper list here too, for
  # the same reason as the Windows branch above.
  set(
    SUPERLU_DIST_PARMETIS_LIBRARIES
    -lptscotchparmetisv3
    -lptscotch
    -lptscotcherr
    -lscotchmetisv5
    -lscotch
    -lscotcherr
  )
  set(
    SUPERLU_DIST_PARMETIS_LIBRARIES_DEBUG
    "${SUPERLU_DIST_PARMETIS_LIBRARIES}"
  )
  set(
    SUPERLU_DIST_PARMETIS_LIBRARIES_RELEASE
    "${SUPERLU_DIST_PARMETIS_LIBRARIES}"
  )
endif()

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  OPTIONS
    -DXSDK_ENABLE_Fortran=OFF
    -Denable_tests=OFF
    -Denable_examples=OFF
    -Denable_python=OFF
    -Denable_openmp=OFF
    -DBUILD_STATIC_LIBS=OFF
    -DTPL_ENABLE_INTERNAL_BLASLIB=OFF
    -DTPL_ENABLE_LAPACKLIB=ON
    -DTPL_ENABLE_PARMETISLIB=ON
    "-DTPL_PARMETIS_INCLUDE_DIRS=${CURRENT_INSTALLED_DIR}/include"
    "-DCMAKE_C_FLAGS_INIT=-DSCOTCH_METIS_VERSION=5"
  OPTIONS_DEBUG
    "-DTPL_PARMETIS_LIBRARIES=${SUPERLU_DIST_PARMETIS_LIBRARIES_DEBUG}"
  OPTIONS_RELEASE
    "-DTPL_PARMETIS_LIBRARIES=${SUPERLU_DIST_PARMETIS_LIBRARIES_RELEASE}"
)
vcpkg_cmake_install()
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/License.txt")
