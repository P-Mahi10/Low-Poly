cmake_minimum_required(VERSION 3.15)
project(lowpoly LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CGAL REQUIRED)

add_executable(lowpoly main.cpp)
target_link_libraries(lowpoly PRIVATE CGAL::CGAL)

# OpenCV is only needed to read the silhouette masks (distance transform).
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
if(OpenCV_FOUND)
  target_compile_definitions(lowpoly PRIVATE HAVE_OPENCV)
  target_include_directories(lowpoly PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(lowpoly PRIVATE ${OpenCV_LIBS})
else()
  message(WARNING "OpenCV not found: silhouette term disabled (sudo pacman -S opencv)")
endif()