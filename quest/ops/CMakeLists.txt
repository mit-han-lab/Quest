cmake_minimum_required(VERSION 3.26.4)  # Specify your minimum CMake version

set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# ------------- configure rapids-cmake --------------#
include(${CMAKE_SOURCE_DIR}/cmake/fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

project(_kernels LANGUAGES CUDA CXX)  # Replace with your project's name

# ------------- configure raft -----------------#
rapids_cpm_init()
include(${CMAKE_SOURCE_DIR}/cmake/get_raft.cmake)

# Check: https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
# Fix linking error: https://github.com/pytorch/pytorch/issues/108041
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# Try combine pybind
# Check: https://qiita.com/syoyo/items/c3e8e6e5c3e2d69c2325
add_subdirectory(${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/pybind ${CMAKE_BINARY_DIR}/pybind11)

file(GLOB PYTORCH_SOURCES "csrc/*.cu")
pybind11_add_module(_kernels MODULE ${PYTORCH_CPP_SOURCES} ${PYTORCH_SOURCES})

target_compile_definitions(_kernels PRIVATE -DBSK_TORCH_CHECK) # Enable Torch Tensor Dimension Check
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/flashinfer/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/pybind/include)
target_compile_options(_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} raft::raft Python::Python pybind11::module ${TORCH_PYTHON_LIBRARY})