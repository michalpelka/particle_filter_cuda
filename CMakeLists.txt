cmake_minimum_required(VERSION 2.8.3)
project(particle_filter_cuda)

find_package(catkin REQUIRED COMPONENTS 
	roscpp
	cmake_modules
	pcl_conversions
	pcl_ros
	roscpp
	sensor_msgs
	tf
	nav_msgs
  	message_generation
  	tf_conversions   
	dynamic_reconfigure 
  	)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake" "${CMAKE_SOURCE_DIR}/particle_filter_cuda/cmake")
generate_dynamic_reconfigure_options(
  cfg/particle_filter_cuda.cfg
)
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES particle_filter_cuda
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(PCL 1.5 REQUIRED)
find_package(CUDA 7.5 REQUIRED)
find_package(CUDADevice 3.0 REQUIRED)

include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDADEVICE_NVCC_FLAGS} -Xptxas -v")
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
SET(COMMON_SOURCES
	common/src/data_model.cpp
	common/src/cudaWrapper.cpp
	common/src/cudaStructures.cpp
	common/src/basicFunctions.cpp
	common/src/cudaBasicFunctions.cu
	common/src/cudaRegularGridDecompositionFunctions.cu
	common/src/cudaNearestNeighbourFunctions.cu
)

include_directories(common/include)

#cuda_add_executable(particle_filter_localization
#	src/particle_filter_localization_main.cpp
#	src/particle_filter.cpp
#	${COMMON_SOURCES}
#)

#target_link_libraries(particle_filter_localization
#	${OPENGL_LIBRARIES}
#	${GLUT_LIBRARIES}
#	${PCL_LIBRARIES}
#)

#cuda_add_executable(particle_filter_localization_fast
#	src/particle_filter_localization_fast_main.cpp
#	src/particle_filter_fast.cpp
#	${COMMON_SOURCES}
#)

#target_link_libraries(particle_filter_localization_fast
#	${OPENGL_LIBRARIES}
#	${GLUT_LIBRARIES}
#	${PCL_LIBRARIES}
#)

cuda_add_executable(particle_filter_localization_fast_node
	src/particle_filter_localization_fast_node.cpp
	src/particle_filter_fast.cpp
	${COMMON_SOURCES}
)
add_dependencies(particle_filter_localization_fast_node particle_filter_cuda_gencfg)
target_link_libraries(particle_filter_localization_fast_node
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
	${catkin_LIBRARIES}
)
cuda_add_executable(classifyMap
	src/classify.cpp
	src/particle_filter_fast.cpp
	${COMMON_SOURCES}
)

target_link_libraries(classifyMap
	${OPENGL_LIBRARIES}
	${GLUT_LIBRARIES}
	${PCL_LIBRARIES}
	${catkin_LIBRARIES}
)

