#include "lbm3d/core.h"

// 3D test domain
template < typename NSE >
struct StateLocal : State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK< NSE >;

	using State<NSE>::nse;
	using State<NSE>::vtk_helper;
	using State<NSE>::log;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	real lbmInflowDensity = no1;

	virtual void setupBoundaries()
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW); 		// left
		nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_OUTFLOW_EQ);

		nse.setBoundaryZ(1, BC::GEO_WALL);		// top
		nse.setBoundaryZ(nse.lat.global.z()-2, BC::GEO_WALL);	// bottom
		nse.setBoundaryY(1, BC::GEO_WALL); 		// back
		nse.setBoundaryY(nse.lat.global.y()-2, BC::GEO_WALL);		// front

		// extra layer needed due to A-A pattern
		nse.setBoundaryZ(0, BC::GEO_NOTHING);		// top
		nse.setBoundaryZ(nse.lat.global.z()-1, BC::GEO_NOTHING);	// bottom
		nse.setBoundaryY(0, BC::GEO_NOTHING); 		// back
		nse.setBoundaryY(nse.lat.global.y()-1, BC::GEO_NOTHING);		// front

		// draw a sphere
		if (0)
		{
			int cy=floor(0.2/nse.lat.physDl);
			int cz=floor(0.2/nse.lat.physDl);
			int cx=floor(0.45/nse.lat.physDl);
			real radius=0.05; // 10 cm diameter
			int range=ceil(radius/nse.lat.physDl)+1;
			for (int py=cy-range;py<=cy+range;py++)
			for (int pz=cz-range;pz<=cz+range;pz++)
			for (int px=cx-range;px<=cx+range;px++)
				if (NORM( (real)(px-cx)*nse.lat.physDl, (real)(py-cy)*nse.lat.physDl, (real)(pz-cz)*nse.lat.physDl) < radius )
					nse.setMap(px,py,pz,BC::GEO_WALL);
		}

		// draw a cylinder
		if (0)
		{
			//int cy=floor(0.2/nse.lat.physDl);
			int cz=floor(0.2/nse.lat.physDl);
			int cx=floor(0.45/nse.lat.physDl);
			real radius=0.05; // 10 cm diameter
			int range=ceil(radius/nse.lat.physDl)+1;
			//		for (int py=cy-range;py<=cy+range;py++)
			for (int pz=cz-range;pz<=cz+range;pz++)
			for (int px=cx-range;px<=cx+range;px++)
			for (int py=0;py<=nse.lat.global.y()-1;py++)
				if (NORM( (real)(px-cx)*nse.lat.physDl,0, (real)(pz-cz)*nse.lat.physDl) < radius )
					nse.setMap(px,py,pz,BC::GEO_WALL);
		}

		// draw a block
		if (1)
		{
			//int cy=floor(0.2/nse.lat.physDl);
			//int cz=floor(0.20/nse.lat.physDl);
			int cx=floor(0.20/nse.lat.physDl);
			//int range=nse.lat.global.z()/4;
			int width=nse.lat.global.z()/10;
			//for (int py=cy-range;py<=cy+range;py++)
			//for (int pz=0;pz<=cz;pz++)
			for (int px=cx;px<=cx+width;px++)
			for (int pz=1;pz<=nse.lat.global.z()-2;pz++)
			for (int py=1;py<=nse.lat.global.y()-2;py++)
				if (!((pz>=nse.lat.global.z()*4/10 &&  pz<=nse.lat.global.z()*6/10) && (py>=nse.lat.global.y()*4/10 && py<=nse.lat.global.y()*6/10)))
					nse.setMap(px,py,pz,BC::GEO_WALL);
		}
	}

	virtual bool outputData(const BLOCK& block, int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs)
	{
		int k=0;
		if (index==k++) return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho,x,y,z), 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("velocity", block.hmacro(MACRO::e_vx,x,y,z), 3, desc, value, dofs);
				case 1: return vtk_helper("velocity", block.hmacro(MACRO::e_vy,x,y,z), 3, desc, value, dofs);
				case 2: return vtk_helper("velocity", block.hmacro(MACRO::e_vz,x,y,z), 3, desc, value, dofs);
			}
		}
		return false;
	}

	virtual void probe1()
	{
		if (nse.iterations != 0)
		{
			// inflow density extrapolation
			idx x = 5;
			idx y = nse.lat.global.y()/2;
			idx z = nse.lat.global.z()/2;
			for (auto& block : nse.blocks)
			if (block.isLocalIndex(x, y, z))
			{
				real oldlbmInflowDensity = lbmInflowDensity;
				lbmInflowDensity = block.dmacro.getElement(MACRO::e_rho, x, y, z);
				log("[probe: lbm inflow density changed from %e to %e", oldlbmInflowDensity, lbmInflowDensity);
			}
		}
	}

	virtual void updateKernelVelocities()
	{
		for (auto& block : nse.blocks)
			block.data.inflow_rho = lbmInflowDensity;
	}

	StateLocal(const TNL::MPI::Comm& communicator, lat_t ilat, real iphysViscosity, real iphysVelocity, real iphysDt)
		: State<NSE>(communicator, ilat, iphysViscosity, iphysDt)
	{
		for (auto& block : nse.blocks)
		{
			block.data.inflow_rho = no1;
			block.data.inflow_vx = nse.phys2lbmVelocity(iphysVelocity);
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
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
			idx y = nse.lat.global.y()/2;
			idx z = nse.lat.global.z()/2;
			for (auto& block : nse.blocks)
			if (block.isLocalIndex(x, y, z))
			{
				real oldlbmInflowDensity = lbmInflowDensity;
				lbmInflowDensity = block.hmacro(MACRO::e_rho, x, y, z);
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
			idx y = nse.lat.global.y()/2;
			idx z = nse.lat.global.z()/2;
			for (auto& block : nse.blocks)
			if (block.isLocalIndex(x, y, z))
			{
				real oldlbmInflowDensity = lbmInflowDensity;
				lbmInflowDensity = block.hmacro(MACRO::e_rho, x, y, z);
				log("[loadState: lbm inflow density changed from %e to %e", oldlbmInflowDensity, lbmInflowDensity);
			}
		}
	}
};

template < typename NSE >
int sim01_test(int RESOLUTION = 2)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size=32;
	int X = 128*RESOLUTION;// width in pixels
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
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType( X, Y, Z );
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;

	StateLocal< NSE > state(MPI_COMM_WORLD, lat, PHYS_VISCOSITY, PHYS_VELOCITY, PHYS_DT);
	state.setid("sim_1_res%02d_np%03d", RESOLUTION, state.nse.nproc);

	#ifdef HAVE_MPI
	// disable MPI communication over the periodic boundary
	for (auto& block : state.nse.blocks) {
		if (block.id == 0)
			block.left_id = -1;
		if (block.id == block.nproc - 1)
			block.right_id = -1;
	}
	#endif

//	state.printIter = 100;
	state.nse.physFinalTime = 1.0;
	state.cnt[PRINT].period = 0.001;
	state.cnt[PROBE1].period = 0.001;
	// test
//	state.cnt[PRINT].period = 100*PHYS_DT;
//	state.nse.physFinalTime = 1000*PHYS_DT;
//	state.cnt[VTK3D].period = 1000*PHYS_DT;
//	state.cnt[SAVESTATE].period = 600;  // save state every [period] of wall time
//	state.check_savestate_flag = false;
//	state.wallTime = 60;
	// RCI
//	state.nse.physFinalTime = 0.5;
//	state.cnt[VTK3D].period = 0.5;
//	state.cnt[SAVESTATE].period = 3600;  // save state every [period] of wall time
//	state.check_savestate_flag = false;
//	state.wallTime = 3600 * 23.5;

	// add cuts
	state.cnt[VTK2D].period = 0.001;
	state.add2Dcut_X(X/2,"cutsX/cut_X");
	state.add2Dcut_Y(Y/2,"cutsY/cut_Y");
	state.add2Dcut_Z(Z/2,"cutsZ/cut_Z");

//	state.cnt[VTK3DCUT].period = 0.01;
//	state.add3Dcut(X/4,Y/4,Z/4, X/2,Y/2,Z/2, 2, "box");

	execute(state);

	return 0;
}

template < typename TRAITS=TraitsSP >
void run(int RES)
{
//	using COLL = D3Q27_CUM< TRAITS >;
	using COLL = D3Q27_CUM< TRAITS, D3Q27_EQ_INV_CUM<TRAITS> >;

	using NSE_CONFIG = LBM_CONFIG<
				TRAITS,
				D3Q27_KernelStruct,
				NSE_Data_ConstInflow< TRAITS >,
				COLL,
				typename COLL::EQ,
				D3Q27_STREAMING< TRAITS >,
				D3Q27_BC_All,
				D3Q27_MACRO_Default< TRAITS >,
				D3Q27_MACRO_Void< TRAITS >
			>;

	sim01_test<NSE_CONFIG>(RES);
}

int main(int argc, char **argv)
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
