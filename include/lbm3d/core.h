#pragma once

#include "defs.h"
#include "state.h"

template < typename STATE >
void execute(STATE& state)
{
#if defined(HAVE_MPI) && defined(USE_CUDA)
	// get the range of stream priorities for current GPU
	int priority_high, priority_low;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
	// high-priority streams for boundaries
	cudaStreamCreateWithPriority(&cuda_streams[0], cudaStreamNonBlocking, priority_high);
	cudaStreamCreateWithPriority(&cuda_streams[1], cudaStreamNonBlocking, priority_high);
	// low-priority stream for the interior
	cudaStreamCreateWithPriority(&cuda_streams[2], cudaStreamNonBlocking, priority_low);
#endif

	// initialize the simulation -- load state or allocate, copy to device and synchronize overlaps with MPI
	state.SimInit();

	// make snapshot for the initial condition
	state.AfterSimUpdate();

	bool quit = false;
	while (!quit)
	{
		// update kernel data (viscosity, swap df1 and df2)
		state.updateKernelData();
		state.updateKernelVelocities();

		state.SimUpdate();

		// post-processing: snapshots etc.
		state.AfterSimUpdate();

		// check wall time
		// (Note that state.wallTimeReached() must be called exactly once per iteration!)
		if (state.wallTimeReached())
		{
			// copy all LBM quantities from device to host
			state.copyAllToHost();

			state.log("maximum wall time reached");
			// copy data to CPU (if needed)
			state.saveState(true);
			quit = true;
		}
		// check savestate
//		else if (state.cnt[SAVESTATE].action(nse.physTime()))
		else if (state.cnt[SAVESTATE].action(state.getWallTime(true)))
		{
			// copy all LBM quantities from device to host
			state.copyAllToHost();

			state.saveState();
			state.cnt[SAVESTATE].count++;
		}

		// check final time
		if (state.nse.physTime() > state.nse.physFinalTime)
		{
			state.log("physFinalTime reached");
			quit = true;
		}

		// handle termination locally
		if (state.nse.quit())
		{
			state.log("terminate flag triggered");
			quit = true;
		}

		// distribute quit among all MPI processes
		quit = TNL::MPI::reduce(quit, MPI_LOR, MPI_COMM_WORLD);
	}

	state.AfterSimFinished();

#if defined(HAVE_MPI) && defined(USE_CUDA)
	for (int i = 0; i < 3; i++)
		cudaStreamDestroy(cuda_streams[i]);
#endif
}
