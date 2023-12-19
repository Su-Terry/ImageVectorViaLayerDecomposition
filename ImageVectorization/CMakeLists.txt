cmake_minimum_required(VERSION 3.12)
project(ImageVectorization)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Eigen
include_directories("/usr/include/eigen3")

# Find autodiff
set(AUTODIFF_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/autodiff-master)
include_directories(${AUTODIFF_DIR})

# Find NLopt package
find_package(NLopt REQUIRED)

# Add your source files
set(SOURCES
    ${CMAKE_SOURCE_DIR}/ImageVectorization/Main.cpp
    # Add other source files here
)

# Create an executable
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${NLOPT_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${NLOPT_LIBRARIES}
)

# Set additional compiler flags if needed
# For example, to enable optimization
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
