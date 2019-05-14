include(Linter)

find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(PythonLibs REQUIRED)

set(QRACK_SIMULATOR_CPP_SRC_DIR "${PROJECT_SOURCE_DIR}/qiskit/providers/qrack/backends/wrappers")
message("Qrack include directory:" ${QRACK_SIMULATOR_CPP_SRC_DIR})

# We need to remove the -static flag, because Python Extension system only supports
# dynamic linked libraries, but we want to build a shared libraries with the least
# dependencies we can, so some of these dependencies are linked statically into our
# shared library.
string(REPLACE " -static " "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Set some general flags
if(APPLE)
    message(STATUS "On Mac, we force linking with undefined symbols for Python library, they will be
                    solved at runtime by the loader")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(AER_LINKER_FLAGS "-undefined dynamic_lookup")
    else()
        # -flat_namespace linker flag is needed otherwise dynamic symbol resolution doesn't work as expected with GCC.
        # Symbols with the same name exist in different .so, so the loader just takes the first one it finds,
        # which is usually the one from the first .so loaded.
        # See: Two-Leve namespace symbol resolution
        set(AER_LINKER_FLAGS "-undefined dynamic_lookup -flat_namespace")
    endif()
    unset(PYTHON_LIBRARIES)
endif()

# Qrack Controller

add_cython_target(qrack_controller_wrapper qrack_controller_wrapper.pyx CXX)
add_library(qrack_controller_wrapper MODULE ${qrack_controller_wrapper})
set_target_properties(qrack_controller_wrapper PROPERTIES
	LINKER_LANGUAGE CXX
    CXX_STANDARD 14)
if(APPLE)
    set_target_properties(qrack_controller_wrapper PROPERTIES
        LINK_FLAGS ${AER_LINKER_FLAGS})
endif()
target_include_directories(qrack_controller_wrapper
    PRIVATE ${QRACK_SIMULATOR_CPP_SRC_DIR}
    PRIVATE ${PYTHON_INCLUDE_DIRS})

option (ENABLE_OPENCL "Use OpenCL optimizations" ON)

set (OPENCL_AMDSDK /opt/AMDAPPSDK-3.0 CACHE PATH "Installation path for the installed AMD OpenCL SDK, if used")

# Options used when building the project
find_package (OpenCL)
if (NOT OpenCL_FOUND)
    # Attempt with AMD's OpenCL SDK
    find_library (LIB_OPENCL OpenCL PATHS ${OPENCL_AMDSDK}/lib/x86_64/)
    if (NOT LIB_OPENCL)
        set (ENABLE_OPENCL OFF)
    else ()
        # Found, set the required include path.
        set (OpenCL_INCLUDE_DIRS ${OPENCL_AMDSDK}/include CACHE PATH "AMD OpenCL SDK Header include path")
        set (OpenCL_COMPILATION_OPTIONS
            -Wno-ignored-attributes
            -Wno-deprecated-declarations
            CACHE STRING "AMD OpenCL SDK Compilation Option Requirements")
        message ("OpenCL support found in the AMD SDK")
    endif()
endif ()

message ("OpenCL Support is: ${ENABLE_OPENCL}")

if (ENABLE_OPENCL)
    message ("    libOpenCL: ${OpenCL_LIBRARIES}")
    message ("    Includes:  ${OpenCL_INCLUDE_DIRS}")
    message ("    Options:   ${OpenCL_COMPILATION_OPTIONS}")
endif ()

if (ENABLE_OPENCL)
    target_link_libraries(qrack_controller_wrapper
        qrack
        ${OpenCL_LIBRARIES}
        ${AER_LIBRARIES}
        ${PYTHON_LIBRARIES})
else ()
    target_link_libraries(qrack_controller_wrapper
        qrack
        ${AER_LIBRARIES}
        ${PYTHON_LIBRARIES})
endif ()

python_extension_module(qrack_controller_wrapper
    FORWARD_DECL_MODULES_VAR fdecl_module_list)

python_modules_header(modules
    FORWARD_DECL_MODULES_LIST ${fdecl_module_list})
include_directories(${modules_INCLUDE_DIRS})
install(TARGETS qrack_controller_wrapper LIBRARY DESTINATION qiskit/providers/qrack/backends)
