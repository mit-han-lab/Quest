cmake_minimum_required(VERSION 3.26.4)  # Specify your minimum CMake version

# Must be set before project() call
option(USE_HIP "Build with HIP for AMD GPUs" OFF)

if(NOT USE_HIP)
  set(CMAKE_C_COMPILER "/usr/bin/gcc-11")
  set(CMAKE_CXX_COMPILER "/usr/bin/g++-11")
  # enable_language(CUDA) caches nvcc's own default the moment this is unset, so it has
  # to be decided here rather than next to the other CUDA settings below.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
else()
  # nvcc optimizes device code whatever the build type, but clang compiles HIP device code
  # at -O0 when none is given, which leaves the kernels many times slower. This has to come
  # before project(): on Windows project() seeds the cache with Debug. A build type given
  # in the CMAKE_BUILD_TYPE environment variable is applied by project() and is left alone.
  get_property(_quest_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  if(NOT CMAKE_BUILD_TYPE AND "$ENV{CMAKE_BUILD_TYPE}" STREQUAL "" AND NOT _quest_multi_config)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  endif()
  # On Windows CMake needs the same clang driver for C++ and HIP, and only ROCm's GNU-style
  # clang++ works for both; left alone it would take cl.exe for C++. Use the compiler
  # CMake is going to pick for HIP unless C++ was chosen explicitly.
  if(CMAKE_HOST_WIN32 AND NOT CMAKE_CXX_COMPILER AND "$ENV{CXX}" STREQUAL "")
    if(CMAKE_HIP_COMPILER)
      set(CMAKE_CXX_COMPILER "${CMAKE_HIP_COMPILER}")
    elseif(NOT "$ENV{HIPCXX}" STREQUAL "")
      set(CMAKE_CXX_COMPILER "$ENV{HIPCXX}")
    else()
      # A pip-installed ROCm SDK puts hipconfig in the Python environment's Scripts
      # directory, which is on PATH only while that environment is activated.
      set(_quest_hipconfig_hints)
      if(Python_EXECUTABLE)
        get_filename_component(_quest_python_dir "${Python_EXECUTABLE}" DIRECTORY)
        list(APPEND _quest_hipconfig_hints "${_quest_python_dir}" "${_quest_python_dir}/Scripts")
      endif()
      foreach(_quest_env_root "$ENV{VIRTUAL_ENV}" "$ENV{CONDA_PREFIX}")
        if(NOT _quest_env_root STREQUAL "")
          file(TO_CMAKE_PATH "${_quest_env_root}/Scripts" _quest_env_scripts)
          list(APPEND _quest_hipconfig_hints "${_quest_env_scripts}")
        endif()
      endforeach()
      find_program(_quest_hipconfig hipconfig
        HINTS ENV HIP_PATH ENV ROCM_PATH ${_quest_hipconfig_hints} PATH_SUFFIXES bin)
      if(_quest_hipconfig)
        execute_process(COMMAND "${_quest_hipconfig}" --hipclangpath
          OUTPUT_VARIABLE _quest_hip_clang_path OUTPUT_STRIP_TRAILING_WHITESPACE)
        file(TO_CMAKE_PATH "${_quest_hip_clang_path}" _quest_hip_clang_path)
        if(EXISTS "${_quest_hip_clang_path}/clang++.exe")
          set(CMAKE_CXX_COMPILER "${_quest_hip_clang_path}/clang++.exe")
          set(CMAKE_HIP_COMPILER "${CMAKE_CXX_COMPILER}")
        endif()
      endif()
      if(NOT CMAKE_CXX_COMPILER)
        set(_quest_hip_clang_not_found TRUE)
      endif()
    endif()
  endif()
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Choose languages based on backend
# We MUST have exactly one project() call. Use enable_language() for the GPU language.
project(_kernels LANGUAGES CXX)

if(USE_HIP)
  if(WIN32 AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" AND _quest_hip_clang_not_found)
    message(FATAL_ERROR
      "The HIP build on Windows needs ROCm's GNU-style clang++ as the C++ compiler, but "
      "none was found (no hipconfig on PATH or in the Python environment), so CMake took "
      "${CMAKE_CXX_COMPILER}. Activate the Python environment that has the ROCm SDK, pass "
      "-DPython_EXECUTABLE=<its python.exe>, or pass "
      "-DCMAKE_HIP_COMPILER=<ROCm>/lib/llvm/bin/clang++.exe, in a fresh build directory.")
  elseif(WIN32 AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    message(FATAL_ERROR
      "The HIP build on Windows needs ROCm's GNU-style clang++ as the C++ compiler, but "
      "CMake was given ${CMAKE_CXX_COMPILER} (MSVC-style command line). Unset the CXX "
      "environment variable or pass -DCMAKE_CXX_COMPILER=<ROCm>/lib/llvm/bin/clang++.exe, "
      "in a fresh build directory.")
  endif()
  # Same ordering constraint as CUDA above: enable_language(HIP) caches whatever
  # rocm_agent_enumerator reports for the machine doing the build unless the target is
  # already known, so let PYTORCH_ROCM_ARCH be read first. An explicit
  # -DCMAKE_HIP_ARCHITECTURES is already in the cache here and wins over both.
  # It has to reach the cache: a normal variable would suppress the enumerator branch that
  # normally writes that entry, leaving the build directory with no architecture at all for
  # the next configure -- which ninja runs by itself whenever a listfile changes. FORCE
  # cannot clobber a -DCMAKE_HIP_ARCHITECTURES because the guard already excluded that case.
  if(NOT CMAKE_HIP_ARCHITECTURES AND DEFINED ENV{PYTORCH_ROCM_ARCH})
    string(REPLACE " " ";" _quest_hip_archs "$ENV{PYTORCH_ROCM_ARCH}")
    set(CMAKE_HIP_ARCHITECTURES "${_quest_hip_archs}" CACHE STRING "HIP architectures" FORCE)
  endif()
  # Windows has no rocm_agent_enumerator, so neither CMake nor find_package(Torch) can
  # detect the installed GPUs there.
  if(WIN32 AND NOT CMAKE_HIP_ARCHITECTURES)
    message(FATAL_ERROR "Name the GPU architecture on Windows, for example "
      "-DCMAKE_HIP_ARCHITECTURES=gfx1100 or the PYTORCH_ROCM_ARCH environment variable.")
  endif()
  enable_language(HIP)
  # find_package(Torch) below overwrites CMAKE_HIP_ARCHITECTURES with the architecture
  # list PyTorch itself was built for, using a normal variable that hides the cache entry.
  # Keep the target resolved here and give that to the module instead.
  set(QUEST_HIP_ARCHITECTURES ${CMAKE_HIP_ARCHITECTURES})
  list(REMOVE_DUPLICATES QUEST_HIP_ARCHITECTURES)
  # find_package(Torch) takes the architecture only from the environment.
  if(WIN32 AND NOT DEFINED ENV{PYTORCH_ROCM_ARCH})
    string(REPLACE ";" " " _quest_hip_archs "${QUEST_HIP_ARCHITECTURES}")
    set(ENV{PYTORCH_ROCM_ARCH} "${_quest_hip_archs}")
  endif()
  # find_package(Torch) looks for ROCm only in ROCM_PATH or through rocm-sdk on PATH, which
  # a Python environment that is not activated does not provide.
  if(WIN32 AND NOT DEFINED ENV{ROCM_PATH} AND CMAKE_HIP_COMPILER_ROCM_ROOT)
    set(ENV{ROCM_PATH} "${CMAKE_HIP_COMPILER_ROCM_ROOT}")
  endif()
  set(CMAKE_HIP_STANDARD 20)
  set(CMAKE_CXX_STANDARD 20)
else()
  # CUDA build path (original)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  # ------------- configure rapids-cmake --------------#
  include(${CMAKE_SOURCE_DIR}/cmake/fetch_rapids.cmake)
  include(rapids-cmake)
  include(rapids-cpm)
  include(rapids-cuda)
  include(rapids-export)
  include(rapids-find)
  rapids_cpm_init()
  include(${CMAKE_SOURCE_DIR}/cmake/get_raft.cmake)
endif()

# Check: https://stackoverflow.com/questions/68401650/how-can-i-make-a-pytorch-extension-with-cmake
# Fix linking error: https://github.com/pytorch/pytorch/issues/108041
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

# Try combine pybind
# Check: https://qiita.com/syoyo/items/c3e8e6e5c3e2d69c2325
add_subdirectory(${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/pybind ${CMAKE_BINARY_DIR}/pybind11)

file(GLOB PYTORCH_SOURCES "csrc/*.cu")
if(USE_HIP)
  set_source_files_properties(${PYTORCH_SOURCES} PROPERTIES LANGUAGE HIP)
endif()

pybind11_add_module(_kernels MODULE ${PYTORCH_CPP_SOURCES} ${PYTORCH_SOURCES})

if(USE_HIP)
  # Not ${CMAKE_HIP_ARCHITECTURES}: find_package(Torch) has replaced it by now.
  set_target_properties(_kernels PROPERTIES HIP_ARCHITECTURES "${QUEST_HIP_ARCHITECTURES}")
endif()

target_compile_definitions(_kernels PRIVATE -DBSK_TORCH_CHECK) # Enable Torch Tensor Dimension Check
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/flashinfer/include)
target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/../../kernels/3rdparty/pybind/include)

if(USE_HIP)
  target_compile_definitions(_kernels PRIVATE USE_HIP)
  # clang defaults to -ffp-contract=fast, which contracts across expressions and drifts
  # further from the reference than nvcc's expression-only contraction.
  target_compile_options(_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:--offload-compress -ffp-contract=on>
  )
  # Ahead of everything else so the CUDA toolkit headers the kernels include by name
  # resolve to the HIP shims in kernels/include/hip_compat.
  set(HIP_COMPAT_DIR ${CMAKE_SOURCE_DIR}/../../kernels/include/hip_compat)
  target_include_directories(_kernels BEFORE PRIVATE ${HIP_COMPAT_DIR})
  # These flashinfer headers are pulled in by a sibling-relative include, which the
  # include path cannot shadow, so they are force-included ahead of it; each carries the
  # upstream include guard, which turns the CUDA original into a no-op. Editing one of them
  # does not invalidate an object file, so wipe the build directory after touching them.
  target_compile_options(_kernels PRIVATE $<$<COMPILE_LANGUAGE:HIP>:
    -include${HIP_COMPAT_DIR}/flashinfer/math.cuh
    -include${HIP_COMPAT_DIR}/flashinfer/mma.cuh
    -include${HIP_COMPAT_DIR}/flashinfer/vec_dtypes.cuh>)
  target_include_directories(_kernels PRIVATE ${CMAKE_SOURCE_DIR}/csrc)
else()
  target_compile_options(_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
endif()

if(USE_HIP)
  target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} Python::Python pybind11::module ${TORCH_PYTHON_LIBRARY})
else()
  target_link_libraries(_kernels PRIVATE ${TORCH_LIBRARIES} raft::raft Python::Python pybind11::module ${TORCH_PYTHON_LIBRARY})
endif()
