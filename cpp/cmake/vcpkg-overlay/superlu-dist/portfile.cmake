# The triplet's linkage, which the dependencies use; vcpkg_check_linkage
# below sets VCPKG_LIBRARY_LINKAGE to static for this port only.
set(SUPERLU_DIST_DEPENDENCY_LINKAGE "${VCPKG_LIBRARY_LINKAGE}")
vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO xiaoyeli/superlu_dist
  REF "v${VERSION}"
  SHA512 aaf399384d74e770081b18c54cd05a1dd0921cfe99ca6b871eb66f4b90246fe21c1dcf572099fbbf1a36a38f3c270271b65a5f4db0b02a0c7026c925d5718b0d
  HEAD_REF master
  PATCHES msvc-no-libm.patch
)

# SuperLU_DIST's ParMETIS TPL is satisfied by the ScotchParMETIS
# compatibility layer (scotch[metis,parmetis,ptscotch]): scotchmetisv5
# provides the METIS API used by SuperLU_DIST itself, and
# ptscotchparmetisv3 provides the ParMETIS API. TPL_PARMETIS_LIBRARIES
# is a raw (non-target) link line, so scotch's static-library
# dependencies have to be listed out by hand, since the scotch port
# is always built static.

# Append to the link line `out_var` the first library in `lib_dir`
# named by ARGN, trying each name with `debug_suffix` first. Library
# names differ between port versions, triplets and configurations.
function(superlu_dist_append_library out_var lib_dir debug_suffix)
  set(file_names "")
  foreach(name IN LISTS ARGN)
    list(APPEND file_names "${name}${debug_suffix}.lib" "${name}.lib")
  endforeach()
  list(REMOVE_DUPLICATES file_names)
  find_file(
    library
    NAMES ${file_names}
    PATHS "${lib_dir}"
    NO_DEFAULT_PATH
    NO_CACHE
    REQUIRED
  )
  set(${out_var} "${${out_var}}${library} " PARENT_SCOPE)
endfunction()

# zlib installs an import or a static library depending on the
# triplet; prefer the matching kind if both are present.
set(SUPERLU_DIST_ZLIB_IMPORT_NAMES z zlib libz)
set(SUPERLU_DIST_ZLIB_STATIC_NAMES zs zlibstatic libzs)
if(SUPERLU_DIST_DEPENDENCY_LINKAGE STREQUAL "static")
  set(
    SUPERLU_DIST_ZLIB_NAMES
    ${SUPERLU_DIST_ZLIB_STATIC_NAMES}
    ${SUPERLU_DIST_ZLIB_IMPORT_NAMES}
  )
else()
  set(
    SUPERLU_DIST_ZLIB_NAMES
    ${SUPERLU_DIST_ZLIB_IMPORT_NAMES}
    ${SUPERLU_DIST_ZLIB_STATIC_NAMES}
  )
endif()

# Release libraries are in lib/, debug libraries in debug/lib/
foreach(config IN ITEMS RELEASE DEBUG)
  if(config STREQUAL "RELEASE")
    set(lib_dir "${CURRENT_INSTALLED_DIR}/lib")
    set(debug_suffix "")
  else()
    set(lib_dir "${CURRENT_INSTALLED_DIR}/debug/lib")
    set(debug_suffix "d")
  endif()
  string(TOLOWER "${config}" build_type)
  if(VCPKG_BUILD_TYPE AND NOT VCPKG_BUILD_TYPE STREQUAL build_type)
    continue()
  endif()

  set(libs "")
  foreach(
    name
    ptscotchparmetisv3
    ptscotch
    ptscotcherr
    scotchmetisv5
    scotch
    scotcherr
  )
    superlu_dist_append_library(libs "${lib_dir}" "${debug_suffix}" ${name})
  endforeach()
  superlu_dist_append_library(libs "${lib_dir}" "${debug_suffix}" bz2 libbz2)
  superlu_dist_append_library(libs "${lib_dir}" "${debug_suffix}" lzma liblzma)
  superlu_dist_append_library(
    libs
    "${lib_dir}"
    "${debug_suffix}"
    ${SUPERLU_DIST_ZLIB_NAMES}
  )
  set(SUPERLU_DIST_PARMETIS_LIBRARIES_${config} "${libs}")
endforeach()

# SuperLU_DIST's configure writes make.inc and superlu_dist_config.h
# into the source tree. Its contents differ between configurations
# (TPL_PARMETIS_LIBRARIES), so configuring in parallel races.
vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  DISABLE_PARALLEL_CONFIGURE
  OPTIONS
    -DXSDK_ENABLE_Fortran=OFF
    -Denable_tests=OFF
    -Denable_examples=OFF
    -Denable_python=OFF
    -Denable_openmp=OFF
    -DBUILD_STATIC_LIBS=ON
    -DTPL_ENABLE_INTERNAL_BLASLIB=OFF
    -DTPL_ENABLE_LAPACKLIB=ON
    -DTPL_ENABLE_PARMETISLIB=ON
    "-DTPL_PARMETIS_INCLUDE_DIRS=${CURRENT_INSTALLED_DIR}/include"
    "-DCMAKE_C_FLAGS_INIT=-DSCOTCH_METIS_VERSION=5"
  OPTIONS_RELEASE
    "-DTPL_PARMETIS_LIBRARIES=${SUPERLU_DIST_PARMETIS_LIBRARIES_RELEASE}"
  OPTIONS_DEBUG
    "-DTPL_PARMETIS_LIBRARIES=${SUPERLU_DIST_PARMETIS_LIBRARIES_DEBUG}"
)
vcpkg_cmake_install()
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/License.txt")
