cmake_minimum_required(VERSION 3.17)
project(DateCopter)

set(CMAKE_CXX_STANDARD 14)

# DLIB
get_directory_property(has_parent PARENT_DIRECTORY)
if(NOT has_parent)
    set(DLIB_IN_PROJECT_BUILD false)
endif()
add_subdirectory(dlib)

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Boost (for easy filesystem manipulation)
# Installation taken from https://stackoverflow.com/questions/3897839/how-to-link-c-program-with-boost-using-cmake
set(Boost_INCLUDE_DIR boost_1_75_0)
set(Boost_LIBRARY_DIR boost_1_75_0/stage/lib)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# Tensorflow
find_library(TENSORFLOW_LIB tensorflow HINT libtensorflow/lib)

add_executable(DateCopter main.cpp ExtractFaces.cpp ExtractFaces.h DataPrepare.cpp DataPrepare.h TrainModel.cpp TrainModel.h)
target_link_libraries(DateCopter dlib::dlib)
target_link_libraries(DateCopter ${OpenCV_LIBS})
target_link_libraries(DateCopter ${Boost_LIBRARIES} )
target_include_directories(DateCopter PRIVATE cppflow/include libtensorflow/include)
target_link_libraries (DateCopter "${TENSORFLOW_LIB}")