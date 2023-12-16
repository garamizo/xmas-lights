cmake_minimum_required (VERSION 2.8.11)
project (MODEL_TREE)

find_package (Eigen3 REQUIRED)
find_package (Ceres REQUIRED)
find_package (OpenCV 4.7.0 REQUIRED)

include_directories(
  include
  ${CERES_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable (bundle src/bundle_adjustment.cpp)
target_link_libraries (bundle 
    ${CERES_LIBRARIES}
    ${OpenCV_LIBRARIES}
)