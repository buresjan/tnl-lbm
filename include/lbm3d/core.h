#pragma once

#include "defs.h"
#include "state.h"

template < typename STATE >
void execute(STATE& state)
{
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

			spdlog::info("maximum wall time reached");
			// copy data to CPU (if needed)
			state.saveState();
			quit = true;
		}
		// check savestate
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
			spdlog::info("physFinalTime reached");
			quit = true;
			state.flagCreate("finished");
			state.flagDelete("loadstate");
		}

		// handle termination (we must reduce the terminate flag first, because
		// only rank 0 can create and delete flags)
		state.nse.terminate = TNL::MPI::reduce(state.nse.terminate, MPI_LOR, MPI_COMM_WORLD);
		if (state.nse.terminate)
		{
			spdlog::info("terminate flag triggered");
			quit = true;
			state.flagCreate("terminated");
			state.flagDelete("loadstate");
		}

		// distribute quit among all MPI processes
		quit = TNL::MPI::reduce(quit, MPI_LOR, MPI_COMM_WORLD);
	}

	state.AfterSimFinished();
}
