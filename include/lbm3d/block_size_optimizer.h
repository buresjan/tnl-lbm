#pragma once

#include <cmath>

#include <TNL/Containers/ndarray/Meta.h>
#include <TNL/Containers/StaticVector.h>

/*
 * Empirical observation: performance of the LBM kernel depends on the number of threads in the CUDA thread block.
 *
 * There are some limitations:
 *
 * - due to registers, the upper limit is lower than hardware (1024) -> must be provided as the max_threads parameter
 * - the leading dimension must be a multiple of the warp_size (32 in current hardware)
 * - necessary condition for optimality: the domain size is a multiple of the selected block size
 *
 * The value of max_threads=256 seems to be the best value (at least on GTX 1080, V100 and A100 cards).
 */
template< typename Permutation, typename idx3d >
idx3d get_optimal_block_size(idx3d domain_size, int max_threads = 256, int warp_size = 32)
{
	using idx = typename idx3d::ValueType;

	// leading dimension
	int i = TNL::Containers::detail::get< 2 >( Permutation{} );
	// second dimension for optimization
	int j = TNL::Containers::detail::get< 1 >( Permutation{} );
	// last dimension (unimportant)
	//int k = TNL::Containers::detail::get< 0 >( Permutation{} );

	idx3d best = {1, 1, 1};
	best[i] = warp_size;
	const idx multiple = domain_size[i] / warp_size;
	if( multiple * warp_size != domain_size[i] )
		return best;

	for( idx bs_j = 1; bs_j <= domain_size[j]; bs_j++ ) {
		if( domain_size[j] % bs_j != 0 )
			continue;

		for( idx m = 1; m <= multiple; m++ ) {
			idx bs_i = m * warp_size;
			// check feasibility condition
			if( domain_size[i] % bs_i != 0 )
				continue;

			// check constraint
			if( bs_i * bs_j > max_threads )
				break;

			// check optimiality condition
			if( bs_i * bs_j > TNL::product( best ) ) {
				best[i] = bs_i;
				best[j] = bs_j;
			}
		}
	}

	std::cout << "CUDA block size optimizer: using block size " << best << " for subdomain size " << domain_size << std::endl;
	return best;
}
