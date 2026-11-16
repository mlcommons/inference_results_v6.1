# CMakeLists.txt for TestPybind module
#
# Build instructions:
#   mkdir build && cd build
#   cmake ..
#   make
#
# Dependencies:
#   - pybind11: pip install pybind11
#   - loadgen: MLPerf loadgen library

cmake_minimum_required(VERSION 3.14)
project(TestPybind)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# MLPerf Loadgen
set(LOADGEN_LIB "/opt/inference/loadgen/build/libmlperf_loadgen.a")
set(LOADGEN_INCLUDE_DIR "/opt/inference/loadgen")

# Find NVTX (part of CUDA toolkit)
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "NVTX include: ${CUDAToolkit_INCLUDE_DIRS}")
    set(HAVE_NVTX TRUE)
else()
    message(STATUS "CUDA Toolkit not found - NVTX disabled")
    set(HAVE_NVTX FALSE)
endif()

# Create the Python module
pybind11_add_module(TestPybind ffi_utils.cpp)

# Link libraries
target_link_libraries(TestPybind PRIVATE 
    ${LOADGEN_LIB}
)

# Add NVTX if available
if(HAVE_NVTX)
    target_link_libraries(TestPybind PRIVATE CUDA::nvtx3)
    target_compile_definitions(TestPybind PRIVATE HAVE_NVTX=1)
endif()

# Include directories
target_include_directories(TestPybind PUBLIC 
    ${LOADGEN_INCLUDE_DIR}
)

# Compiler flags for performance
target_compile_options(TestPybind PRIVATE
    -O3
    -march=native
)

# Print final configuration
message(STATUS "=== Final Configuration ===")
message(STATUS "Loadgen: ${LOADGEN_LIB}")
message(STATUS "NVTX: ${HAVE_NVTX}")

