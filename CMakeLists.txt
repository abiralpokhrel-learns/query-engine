cmake_minimum_required(VERSION 3.20)
project(query_engine VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Arrow CONFIG REQUIRED)
find_package(Parquet CONFIG REQUIRED)

option(QE_ENABLE_CUDA "Build the optional CUDA global-aggregate backend" OFF)

add_library(qe_core
  src/sql/sql.cpp
  src/storage/parquet.cpp
  src/engine/expression.cpp
  src/engine/key.cpp
  src/engine/operators.cpp
  src/engine/plan.cpp
  src/engine/executor.cpp
  src/cli/printer.cpp
)

if(QE_ENABLE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to compile for" FORCE)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(qe_core PRIVATE src/gpu/cuda_runtime.cu)
  target_link_libraries(qe_core PUBLIC CUDA::cudart)
  target_compile_definitions(qe_core PUBLIC QE_ENABLE_CUDA)
  message(STATUS "CUDA backend enabled with CUDAToolkit ${CUDAToolkit_VERSION}")
else()
  target_sources(qe_core PRIVATE src/gpu/cuda_runtime_stub.cpp)
endif()

target_include_directories(qe_core PUBLIC src)

# Different Arrow distributions export different target names. Prefer the
# shared targets used by the conda-forge build, then fall back to the names
# used by distro packages.
if(TARGET Arrow::arrow_shared)
  set(QE_ARROW_TARGET Arrow::arrow_shared)
elseif(TARGET Arrow::arrow)
  set(QE_ARROW_TARGET Arrow::arrow)
else()
  message(FATAL_ERROR "No usable Arrow C++ target was exported by find_package(Arrow)")
endif()

if(TARGET Parquet::parquet_shared)
  set(QE_PARQUET_TARGET Parquet::parquet_shared)
elseif(TARGET Parquet::parquet)
  set(QE_PARQUET_TARGET Parquet::parquet)
else()
  message(FATAL_ERROR "No usable Parquet C++ target was exported by find_package(Parquet)")
endif()

target_link_libraries(qe_core PUBLIC ${QE_ARROW_TARGET} ${QE_PARQUET_TARGET})

option(QE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
set(QE_WARNING_FLAGS -Wall -Wextra -Wpedantic)
if(QE_WARNINGS_AS_ERRORS)
  list(APPEND QE_WARNING_FLAGS -Werror)
endif()

target_compile_options(qe_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${QE_WARNING_FLAGS}>)

add_executable(qe-cli src/main.cpp)
target_link_libraries(qe-cli PRIVATE qe_core)
target_compile_options(qe-cli PRIVATE ${QE_WARNING_FLAGS})
target_compile_definitions(qe-cli PRIVATE
  QE_VERSION="${PROJECT_VERSION}"
  QE_ARROW_VERSION="${Arrow_VERSION}"
)

enable_testing()
add_executable(qe_sql_test tests/sql_test.cpp)
target_link_libraries(qe_sql_test PRIVATE qe_core)
target_compile_options(qe_sql_test PRIVATE ${QE_WARNING_FLAGS})
add_test(NAME sql_parser COMMAND qe_sql_test)

if(QE_ENABLE_CUDA)
  add_executable(qe_cuda_test tests/cuda_test.cpp)
  target_link_libraries(qe_cuda_test PRIVATE qe_core)
  target_compile_options(qe_cuda_test PRIVATE ${QE_WARNING_FLAGS})
  add_test(NAME cuda_runtime COMMAND qe_cuda_test)
  set_tests_properties(cuda_runtime PROPERTIES SKIP_RETURN_CODE 77)
endif()
