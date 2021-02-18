#include "core.h"

// 3D test domain
template < typename LBM_TYPE >
struct StateLocal : State<LBM_TYPE>
{
	using TRAITS = typename LBM_TYPE::TRAITS;
	using BC = typename LBM_TYPE::BC;
	using MACRO = typename LBM_TYPE::MACRO;

	using State<LBM_TYPE>::lbm;
	using State<LBM_TYPE>::vtk_helper;
	using State<LBM_TYPE>::log;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;

	real lbmInflowDensity = no1;

	virtual void setupBoundaries()
	{
		lbm.setBoundaryX(0, BC::GEO_INFLOW); 		// left
//		lbm.setBoundaryX(lbm.global_X-1, BC::GEO_OUTFLOW_RIGHT);		// right
		lbm.setBoundaryX(lbm.global_X-1, BC::GEO_OUTFLOW_EQ);
		lbm.setBoundaryZ(0, BC::GEO_WALL);		// top
		lbm.setBoundaryZ(lbm.global_Z-1, BC::GEO_WALL);	// bottom
		lbm.setBoundaryY(0, BC::GEO_WALL); 		// back
		lbm.setBoundaryY(lbm.global_Y-1, BC::GEO_WALL);		// front
	}

	virtual bool outputData(int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs)
	{
		int k=0;
		if (index==k++) return vtk_helper("lbm_density", lbm.hmacro(MACRO::e_rho,x,y,z), 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("velocity", lbm.hmacro(MACRO::e_vx,x,y,z), 3, desc, value, dofs);
				case 1: return vtk_helper("velocity", lbm.hmacro(MACRO::e_vy,x,y,z), 3, desc, value, dofs);
				case 2: return vtk_helper("velocity", lbm.hmacro(MACRO::e_vz,x,y,z), 3, desc, value, dofs);
			}
		}
		return false;
	}

	virtual void probe1()
	{
		if (lbm.iterations != 0)
		{
			// inflow density extrapolation
			idx x = 5;
			idx y = lbm.global_Y/2;
			idx z = lbm.global_Z/2;
			if (lbm.isLocalIndex(x, y, z))
			{
				real oldlbmInflowDensity = lbmInflowDensity;
				lbmInflowDensity = lbm.dmacro.getElement(MACRO::e_rho, x, y, z);
				log("[probe: lbm inflow density changed from %e to %e", oldlbmInflowDensity, lbmInflowDensity);
			}
		}
	}

	virtual void updateKernelVelocities()
	{
		lbm.data.inflow_rho = lbmInflowDensity;
	}

	StateLocal(int iX, int iY, int iZ, real iphysViscosity, real iphysVelocity, real iphysDl, real iphysDt)
		: State<LBM_TYPE>(iX, iY, iZ, iphysViscosity, iphysDl, iphysDt)
	{
		lbm.data.inflow_rho = no1;
		lbm.data.inflow_vx = lbm.phys2lbmVelocity(iphysVelocity);
		lbm.data.inflow_vy = 0;
		lbm.data.inflow_vz = 0;
	}

	virtual void saveState(bool forced=false)
	{
		if (this->flagExists("savestate") || !this->check_savestate_flag || forced)
		{
			log("[saveState invoked]");
			this->saveAndLoadState(MemoryToFile, "current_state");
			if (this->delete_savestate_flag && !forced)
			{
				this->flagDelete("savestate");
				this->flagCreate("savestate_done");
			}
			if (forced) this->flagCreate("loadstate");

			// set lbmInflowDensity from hmacro -- for consistency with restarted computations
			idx x = 5;
			idx y = lbm.global_Y/2;
			idx z = lbm.global_Z/2;
			if (lbm.isLocalIndex(x, y, z))
			{
				real oldlbmInflowDensity = lbmInflowDensity;
				lbmInflowDensity = lbm.hmacro(MACRO::e_rho, x, y, z);
				log("[loadState: lbm inflow density changed from %e to %e", oldlbmInflowDensity, lbmInflowDensity);
			}
		}
	}

	virtual void loadState(bool forced=false)
	{
		if (this->flagExists("loadstate") || forced)
		{
			log("[loadState invoked]");
			this->saveAndLoadState(FileToMemory, "current_state");

			// set lbmInflowDensity from hmacro
			idx x = 5;
			idx y = lbm.global_Y/2;
			idx z = lbm.global_Z/2;
			if (lbm.isLocalIndex(x, y, z))
			{
				real oldlbmInflowDensity = lbmInflowDensity;
				lbmInflowDensity = lbm.hmacro(MACRO::e_rho, x, y, z);
				log("[loadState: lbm inflow density changed from %e to %e", oldlbmInflowDensity, lbmInflowDensity);
			}
		}
	}
};

template < typename LBM_TYPE >
int sim01_test(int RESOLUTION = 2)
{
	using real = typename LBM_TYPE::TRAITS::real;

	int block_size=32;
	int X = 128*RESOLUTION;// width in pixels --- product of 128.
	//	int Y = 41*RESOLUTION;// height in pixels --- top and bottom walls 1px
	//	int Z = 41*RESOLUTION;// height in pixels --- top and bottom walls 1px
	int Y = block_size*RESOLUTION;// height in pixels --- top and bottom walls 1px
	int Z = Y;// height in pixels --- top and bottom walls 1px
	real LBM_VISCOSITY = 0.00001;//1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_HEIGHT = 0.41; // [m] domain height (physical)
	real PHYS_VISCOSITY = 1.5e-5;// [m^2/s] fluid viscosity .... blood?
	//	real PHYS_VELOCITY = 2.25; // m/s ... will be multip
	real PHYS_VELOCITY = 1.0; // this is only average velocity .... will be multiplied by 9/4 to get the correct value Um from Schafer Turek
	real PHYS_DL = PHYS_HEIGHT/((real)Y-2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY*PHYS_DL*PHYS_DL;//PHYS_HEIGHT/(real)LBM_HEIGHT;

	StateLocal< LBM_TYPE > state(X, Y, Z, PHYS_VISCOSITY, PHYS_VELOCITY, PHYS_DL, PHYS_DT);
	state.setid("sim_1_res%02d_np%03d", RESOLUTION, state.lbm.nproc);
	state.lbm.block_size = 32;
	state.lbm.physCharLength = 0.1; // [m]
//	state.printIter = 100;
//	state.printIter = 100;
//	state.vtk.writePeriod = 0.01;
//	state.cnt[PRINT].period = 0.001;
//	state.cnt[PROBE1].period = 0.001;
	// test
	state.cnt[PRINT].period = 100*PHYS_DT;
	state.lbm.physFinalTime = 1000*PHYS_DT;
//	state.cnt[VTK3D].period = 1000*PHYS_DT;
//	state.cnt[SAVESTATE].period = 600;  // save state every [period] of wall time
//	state.check_savestate_flag = false;
//	state.wallTime = 60;
	// RCI
//	state.lbm.physFinalTime = 0.5;
//	state.cnt[VTK3D].period = 0.5;
//	state.cnt[SAVESTATE].period = 3600;  // save state every [period] of wall time
//	state.check_savestate_flag = false;
//	state.wallTime = 3600 * 23.5;

	// add cuts
//	state.cnt[VTK2D].period = 0.001;
//	state.add2Dcut_X(X/2,"cutsX/cut_X");
//	state.add2Dcut_Y(Y/2,"cutsY/cut_Y");
//	state.add2Dcut_Z(Z/2,"cutsZ/cut_Z");

//	state.cnt[VTK3DCUT].period = 0.01;
//	state.add3Dcut(X/4,Y/4,Z/4, X/2,Y/2,Z/2, 2, "box");

	// draw a sphere
	if (0)
	{
		int cy=floor(0.2/PHYS_DL);
		int cz=floor(0.2/PHYS_DL);
		int cx=floor(0.45/PHYS_DL);
		real radius=0.05; // 10 cm diameter
		int range=ceil(radius/PHYS_DL)+1;
		for (int py=cy-range;py<=cy+range;py++)
		for (int pz=cz-range;pz<=cz+range;pz++)
		for (int px=cx-range;px<=cx+range;px++)
			if (NORM( (real)(px-cx)*PHYS_DL, (real)(py-cy)*PHYS_DL, (real)(pz-cz)*PHYS_DL) < radius )
				state.lbm.defineWall(px,py,pz,true);
	}

	// draw a cylinder
	if (0)
	{
		int cy=floor(0.2/PHYS_DL);
		int cz=floor(0.2/PHYS_DL);
		int cx=floor(0.45/PHYS_DL);
		real radius=0.05; // 10 cm diameter
		int range=ceil(radius/PHYS_DL)+1;
		//		for (int py=cy-range;py<=cy+range;py++)
		for (int pz=cz-range;pz<=cz+range;pz++)
		for (int px=cx-range;px<=cx+range;px++)
		for (int py=0;py<=Y-1;py++)
			if (NORM( (real)(px-cx)*PHYS_DL,0, (real)(pz-cz)*PHYS_DL) < radius )
				state.lbm.defineWall(px,py,pz,true);
	}

	// draw a block
	if (1)
	{
		//		int cy=floor(0.2/PHYS_DL);
		int cz=floor(0.20/PHYS_DL);
		int cx=floor(0.20/PHYS_DL);
		//		int range=Z/4;
		int width=Z/10;
		//		for (int py=cy-range;py<=cy+range;py++)
		//		for (int pz=0;pz<=cz;pz++)
		for (int px=cx;px<=cx+width;px++)
		for (int pz=0;pz<=Z-1;pz++)
		for (int py=0;py<=Y-1;py++)
			if (!((pz>=Z*4/10 &&  pz<=Z*6/10) && (py>=Y*4/10 && py<=Y*6/10)))
				state.lbm.defineWall(px,py,pz,true);
	}

	execute(state);

	return 0;
}

template < typename TRAITS=TraitsSP >
void run(int RES)
{
	using COLL = D3Q27_CUM< TRAITS >;
//	using COLL = D3Q27_CUM< TRAITS, D3Q27_EQ_INV_CUM<TRAITS> >;

	using NSE_TYPE = D3Q27<
				COLL,
				LBM_Data_ConstInflow< TRAITS >,
				D3Q27_BC_All,
				typename COLL::EQ,
				D3Q27_STREAMING< TRAITS >,
				D3Q27_MACRO_Default< TRAITS >,
				D3Q27_MACRO_Void< TRAITS >,
				TRAITS
			>;

	sim01_test<NSE_TYPE>(RES);
}

int Main(int argc, char **argv)
{
	TNLMPI_INIT mpi(argc, argv);

	const int pars=1;
	if (argc <= pars)
	{
		printf("error: required %d parameters:\n %s res[1,...]\n", pars, argv[0]);
		return 1;
	}
	int res = atoi(argv[1]);
	if (res < 1) { printf("error: res=%d out of bounds [1, ...]\n",res); return 1; }

	run(res);

	return 0;
}
