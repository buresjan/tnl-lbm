set(POTENTIAL_TARGETS
        sim_2
)

foreach(target IN ITEMS ${POTENTIAL_TARGETS})
    set(source_file "${CMAKE_CURRENT_SOURCE_DIR}/${target}.cu")
    if(EXISTS "${source_file}")
        add_executable(${target} ${target}.cu)
        target_link_libraries(${target} PUBLIC TNL::TNL_LBM)
    else()
        message(STATUS "Skipping ${target} – source file ${target}.cu not found")
    endif()
endforeach()
