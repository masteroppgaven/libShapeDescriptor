include(CheckLanguage)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
	message(STATUS "CUDA support was detected, GPU kernels enabled")
	project(ShapeDescriptor CXX CUDA)
	set(CUDA_NVCC_FLAGS -lcurand -lineinfo -lcudart)
	include_directories(
		lib/nvidia-samples-common/
		${CUDA_INCLUDE_DIRS}
	)
	find_package(CUDA REQUIRED)
else()
	message(STATUS "No CUDA support was detected, GPU kernels disabled")
	project(ShapeDescriptor CXX)
endif()

cmake_minimum_required (VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)

# Utility for baking in the current git commit into source code
set(PRE_CONFIGURE_FILE "lib/cmake-repo-info/git.cc.in")
set(POST_CONFIGURE_FILE "${CMAKE_CURRENT_BINARY_DIR}/git.cc")
include(lib/cmake-repo-info/git_watcher.cmake)
add_library(gitinfo STATIC ${POST_CONFIGURE_FILE})
target_include_directories(gitinfo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(gitinfo check_git)

find_package(OpenMP)
add_subdirectory(lib/Catch2)
add_subdirectory(lib/fast-lzma2 ${CMAKE_CURRENT_BINARY_DIR}/fast-lzma2)

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp -lpthread -Wall")
endif()

if (MSVC)
    add_definitions(-D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING)
endif()

include_directories (
        src/
        lib/glm/
        lib/lodepng/
		lib/arrrgh/
		lib/json/
		lib/fast-lzma2/src/
		lib/cmake-repo-info/
		lib/
		lib/fast-obj
		tools/
)


file (GLOB_RECURSE LIBRARY_HEADERS             src/*.hpp
                                               src/*.h
                                               src/*.cu
											   tools/benchmarking/*.h)
file (GLOB_RECURSE LIBRARY_KERNELS             src/*.cu
                                               lib/nvidia-samples-common/*.cu)
file (GLOB_RECURSE LIBRARY_SOURCES             src/*.cpp
                                               src/*.cxx
                                               src/*.cc
                                               src/*.c
                                               lib/lodepng/lodepng.cpp
                                               lib/fast-obj/fast_obj.cpp)
file (GLOB_RECURSE TEST_SOURCES                tests/*.cpp)
file (GLOB_RECURSE RENDER_TOOL_SOURCES         tools/imagerenderer/*.cpp)
file (GLOB_RECURSE QUICCI_DUMPER_TOOL_SOURCES  tools/quiccidumper/*.cpp)
file (GLOB_RECURSE QUICCI_VIEWER_TOOL_SOURCES  tools/quicciviewer/*.cpp)
file (GLOB_RECURSE BENCHMARK 				   tools/benchmarking/*.cpp)
file (GLOB_RECURSE BENCHMARK_HEADERS 		   tools/benchmarking/*.h)

if (CMAKE_CUDA_COMPILER)
	add_library(${PROJECT_NAME} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS} ${LIBRARY_KERNELS})
	target_compile_definitions(${PROJECT_NAME} PUBLIC DESCRIPTOR_CUDA_KERNELS_ENABLED)
	target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
else()
	add_library(${PROJECT_NAME} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS} ${LIBRARY_KERNELS})
	# Ensures .cu files are compiled using the regular C++ compiler
	set_source_files_properties(${LIBRARY_KERNELS} PROPERTIES LANGUAGE CXX)
	set_source_files_properties(${LIBRARY_KERNELS} PROPERTIES COMPILE_OPTIONS -xc++)
endif()


target_link_libraries(${PROJECT_NAME} fast-lzma2)


target_include_directories(${PROJECT_NAME} PUBLIC src/ lib/nvidia-samples-common)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75)

add_executable(runTests ${TEST_SOURCES})
target_link_libraries(runTests ShapeDescriptor Catch2::Catch2)
set_property(TARGET runTests PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75)

add_executable(imagerenderer ${RENDER_TOOL_SOURCES})
target_link_libraries(imagerenderer ShapeDescriptor)
set_property(TARGET imagerenderer PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75)

add_executable(quiccidumper ${QUICCI_DUMPER_TOOL_SOURCES})
target_link_libraries(quiccidumper ShapeDescriptor)
set_property(TARGET quiccidumper PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75)

add_executable(benchmarking ${BENCHMARK} ${BENCHMARK_HEADERS})
target_link_libraries(benchmarking ShapeDescriptor)
target_link_libraries(benchmarking gitinfo)
set_property(TARGET benchmarking PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75)

add_executable(quicciviewer ${QUICCI_VIEWER_TOOL_SOURCES})
target_link_libraries(quicciviewer ShapeDescriptor)
