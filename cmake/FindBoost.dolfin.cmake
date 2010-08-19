# - Try to find Boost

set(Boost_ADDITIONAL_VERSIONS 1.43 1.43.0)
set(Boost_USE_STATIC_LIBS OFF)
set(BOOST_ROOT $ENV{BOOST_DIR})
find_package(Boost 1.36 COMPONENTS filesystem program_options system REQUIRED)
