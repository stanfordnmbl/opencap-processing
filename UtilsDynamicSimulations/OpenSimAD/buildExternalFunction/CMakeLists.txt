cmake_minimum_required (VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)

set(TARGET_NAME 
    "" 
    CACHE 
    STRING
    "Target name")

set(TEST_TARGET ${TARGET_NAME})
project(${TEST_TARGET})

add_library(${TEST_TARGET} SHARED foo_jac.c)
install(TARGETS ${TEST_TARGET}
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
				GROUP_READ GROUP_WRITE GROUP_EXECUTE 
				WORLD_READ WORLD_EXECUTE
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	)
    
set(INSTALL_DIR "Path to install directory" CACHE PATH "Location of directory")
# get_filename_component(BASE_DIR ${CMAKE_CURRENT_BINARY_DIR} DIRECTORY)
# Default install prefix libraries.
set(CMAKE_INSTALL_PREFIX 
    ${INSTALL_DIR}
    CACHE
    PATH
    "Default CMAKE_INSTALL_PREFIX." FORCE)
		