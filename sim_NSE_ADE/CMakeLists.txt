set(TARGETS
        sim_T1
)

find_package(OpenMP COMPONENTS CXX)
find_package(MPI COMPONENTS CXX)

foreach(target IN ITEMS ${TARGETS})
    # create the target
    add_executable(${target} ${target}.cu)

    # add TNL to the target
    target_link_libraries(${target} PUBLIC TNL::TNL_CUDA)

    # add special flags used in lbm3d
    target_compile_definitions(${target} PUBLIC "-DUSE_CUDA")

    # enable OpenMP for the target
    if(OPENMP_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_OPENMP")
        target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
        # nvcc needs -Xcompiler
        target_compile_options(${target} PUBLIC $<$<CUDA_COMPILER_ID:NVIDIA>: -Xcompiler=-fopenmp >)
    endif()

    # enable MPI for the target
    if(MPI_FOUND)
        target_compile_definitions(${target} PUBLIC "-DHAVE_MPI")
        target_link_libraries(${target} PUBLIC MPI::MPI_CXX)
    endif()
endforeach()
