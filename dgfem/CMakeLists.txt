cmake_minimum_required(VERSION 3.12)
project(DGFEM_CPP VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard to C++20 for modern features
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Trilinos: sole linear algebra dependency (see TRILINOS.md for the build recipe --
# no prebuilt package exists for macOS ARM64, so it must be built from source).
find_package(Trilinos REQUIRED)
message(STATUS "Found Trilinos ${Trilinos_VERSION} at ${Trilinos_DIR}")

# Find GMSH
find_path(GMSH_INCLUDE_DIR gmsh.h)
find_library(GMSH_LIBRARY NAMES gmsh)

if(NOT GMSH_INCLUDE_DIR OR NOT GMSH_LIBRARY)
    message(FATAL_ERROR "GMSH not found")
endif()

message(STATUS "Found GMSH: ${GMSH_LIBRARY}")
message(STATUS "GMSH include dir: ${GMSH_INCLUDE_DIR}")

# Options for building examples and tests
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build test programs" ON)

include(CTest)

if(BUILD_TESTS)
    enable_testing()
endif()

# Find GTest (for testing) - only if building tests
if(BUILD_TESTS)
    set(CMAKE_PREFIX_PATH "/opt/homebrew" ${CMAKE_PREFIX_PATH})
    find_package(GTest REQUIRED)
    if(NOT GTest_FOUND)
        message(FATAL_ERROR "Google Test not found")
    endif()
    message(STATUS "Found GTest")
endif()

# Include directories
include_directories(include)

# Collect source files
file(GLOB_RECURSE DGFEM_SOURCES 
    "src/*.cpp"
)

# Create library
add_library(dgfem ${DGFEM_SOURCES})

# Compiler optimization flags
target_compile_options(dgfem PRIVATE
    $<$<CONFIG:Release>:-O3 -ffast-math -march=native>
    $<$<CONFIG:Debug>:-O0 -g>
)

target_compile_definitions(dgfem PRIVATE
    $<$<CONFIG:Release>:NDEBUG>
)

# Link libraries
target_link_libraries(dgfem
    PUBLIC
        Trilinos::all_libs
        ${GMSH_LIBRARY}
)

# Include directories for the library
target_include_directories(dgfem 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${GMSH_INCLUDE_DIR}
)

# Compiler flags (removed GMSH_CFLAGS_OTHER as it's not available)

# Add examples (optional)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Add tests (optional)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Trilinos toolchain smoke test (Milestone 1 of the migration -- see plan/TRILINOS.md):
# proves Kokkos init + Tpetra assembly + Amesos2 solve work end to end, independent of the
# rest of the library.
add_subdirectory(trilinos_smoke_test)

# Installation
install(TARGETS dgfem
    EXPORT dgfem-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/ DESTINATION include)

install(EXPORT dgfem-targets
    FILE dgfem-targets.cmake
    NAMESPACE dgfem::
    DESTINATION lib/cmake/dgfem
)

# Generate config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dgfem-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/dgfem-config.cmake"
    INSTALL_DESTINATION lib/cmake/dgfem
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dgfem-config.cmake"
    DESTINATION lib/cmake/dgfem
)