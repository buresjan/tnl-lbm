#include "core.h"

// 3D test problem: forcing/input velocity
// analytical solution for rectangular duct: forcing accelerated

template < typename TRAITS >
struct NSE_Data_XProfileInflow : NSE_Data < TRAITS >
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

//	dreal inflow_rho=no1;
	dreal* vx_profile=NULL;
	idx size_y=0;

	template< typename LBM_KS >
	CUDA_HOSTDEV void inflow(LBM_KS &KS, idx x, idx y, idx z)
	{
//		KS.rho = inflow_rho;
		KS.vx  = vx_profile[y+z*size_y];
		KS.vy  = 0;
		KS.vz  = 0;
	}
};

// uloha: periodic, forcing accelerated
template < typename LBM_TYPE >
struct StateLocal : State<LBM_TYPE>
{
	using TRAITS = typename LBM_TYPE::TRAITS;
	using BC = typename LBM_TYPE::BC;
	using MACRO = typename LBM_TYPE::MACRO;
	using BLOCK = LBM_BLOCK< LBM_TYPE >;

	using State<LBM_TYPE>::nse;
	using State<LBM_TYPE>::vtk_helper;
	using State<LBM_TYPE>::log;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	real **an_cache=0;
	int an_n=50;

	int errors_count;
	real* l1errors;
	int error_idx = 0;

	real raw_analytical_ux(int n, int ilbm_y, int ilbm_z)
	{
		if (ilbm_y==0 || ilbm_y==nse.lat.global.y()-1 || ilbm_z==0 || ilbm_z==nse.lat.global.z()-1) return 0;
		int lbm_y=ilbm_y;
		int lbm_z=ilbm_z;
		real a = nse.lat.global.y()/2.0 - 1.0;
		real b = nse.lat.global.z()/2.0 - 1.0;
		real y = ((real)lbm_y + 0.5 - nse.lat.global.y()/2.)/a;
		real z = ((real)lbm_z + 0.5 - nse.lat.global.z()/2.)/a;
		real b_ku_a = b/a;
		real sum=0;
		real minusonek=1.0;
		real kkk;
		real omega=PI/2.0;
		for (int k=0;k<=n;k++)
		{
			kkk=2.0*k+1.;
			sum += minusonek*(1.0 - exp( omega*kkk*(z-b_ku_a) ) * (1.0 + exp( -omega*2.0*kkk*z ))/(1.0 + exp(-omega*2.0*kkk*b_ku_a) ))*cos( omega*kkk*y )/kkk/kkk/kkk;
			minusonek *= -1.0;
		}

//		real coef = (nse.blocks.front().data.fx != 0) ? nse.blocks.front().data.fx : nse.blocks.front().data.inflow_vx;
		real coef = nse.blocks.front().data.fx;
		return coef * 16.0*a*a/PI/PI/PI*sum/nse.lbmViscosity();
	}

	real analytical_ux(int lbm_y, int lbm_z)
	{
		if (!an_cache) cache_analytical();
		return an_cache[lbm_z][lbm_y];
	}

	void cache_analytical()
	{
		an_cache = new real*[nse.blocks.front().local.z()];
		for (int i=0;i<nse.blocks.front().local.z();i++) an_cache[i] = new real[nse.blocks.front().local.y()];
		#pragma omp parallel for schedule(static) collapse(2)
		for (int z=0;z<nse.blocks.front().local.z();z++)
		for (int y=0;y<nse.blocks.front().local.y();y++)
			an_cache[z][y] = raw_analytical_ux(an_n, nse.blocks.front().offset.y() + y, nse.blocks.front().offset.z() + z);
	}

	virtual void setupBoundaries()
	{
//		if (nse.blocks.front().data.inflow_vx != 0)
		if (nse.blocks.front().data.vx_profile)
		{
			nse.setBoundaryX(0, BC::GEO_INFLOW); 		// left
//			nse.setBoundaryX(0, BC::GEO_INFLOW_FREE_RHO); 		// left
			nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_OUTFLOW_EQ);		// right
//			nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_OUTFLOW_RIGHT);		// right
//			nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_OUTFLOW_RIGHT_INTERP);		// right
		} else
		{
			nse.setBoundaryX(0, BC::GEO_PERIODIC); 		// left
			nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_PERIODIC);		// right
		}

		nse.setBoundaryZ(1, BC::GEO_WALL);		// top
		nse.setBoundaryZ(nse.lat.global.z()-2, BC::GEO_WALL);	// bottom
		nse.setBoundaryY(1, BC::GEO_WALL); 		// back
		nse.setBoundaryY(nse.lat.global.y()-2, BC::GEO_WALL);		// front

		// extra layer needed due to A-A pattern
		nse.setBoundaryZ(0, BC::GEO_NOTHING);		// top
		nse.setBoundaryZ(nse.lat.global.z()-1, BC::GEO_NOTHING);	// bottom
		nse.setBoundaryY(0, BC::GEO_NOTHING); 		// back
		nse.setBoundaryY(nse.lat.global.y()-1, BC::GEO_NOTHING);		// front
	}

	virtual bool outputData(const BLOCK& block, int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs)
	{
		int k=0;
		if (index==k++) return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_delta_density", block.hmacro(MACRO::e_rho,x,y,z) - 1.0, 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vx,x,y,z), 3, desc, value, dofs);
				case 1: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vy,x,y,z), 3, desc, value, dofs);
				case 2: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vz,x,y,z), 3, desc, value, dofs);
			}
		}
		if (index==k++) return vtk_helper("lbm_analytical_ux", analytical_ux(y, z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_ux", block.hmacro(MACRO::e_vx,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uy", block.hmacro(MACRO::e_vy,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uz", block.hmacro(MACRO::e_vz,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_error_ux", fabs(block.hmacro(MACRO::e_vx,x,y,z) - analytical_ux(y,z)), 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(MACRO::e_vx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(MACRO::e_vy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(MACRO::e_vz,x,y,z)), 3, desc, value, dofs);
			}
		}
		if (index==k++) return vtk_helper("lbm_analytical_ux", nse.lbm2physVelocity(analytical_ux(y, z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_ux", nse.lbm2physVelocity(block.hmacro(MACRO::e_vx,x,y,z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uy", nse.lbm2physVelocity(block.hmacro(MACRO::e_vy,x,y,z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uz", nse.lbm2physVelocity(block.hmacro(MACRO::e_vz,x,y,z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_error_ux", nse.lbm2physVelocity(fabs(block.hmacro(MACRO::e_vx,x,y,z) - analytical_ux(y,z))), 1, desc, value, dofs);
		return false;
	}

/*
	// clear Lattice + boundary setup
	virtual void resetLattice(real irho, real ivx, real ivy, real ivz)
	{
		for (int x = 0; x < nse.X;x++)
		for (int y = 0; y < nse.Y;y++)
		for (int z = 0; z < nse.Z;z++)
		{
			dreal rho= no1;
			dreal vx= 0;//(lbmInputVelocityX!=0) ? (dreal)analytical_ux(y,z) : 0;
			dreal vy= 0;
			dreal vz= 0;
			nse.setEqLat(x,y,z,rho,vx,vy,vz);
		}

	}
*/
	virtual void probe1()
	{
		// compute exact error
		// uy,uz should be zero
		// use openmp
		// warning: ux,uy,uz are in LBM units ... measure that in phys units
		// analytical ux has no dpdx --- add
		// compute error
		real local_l1sum=0;
		real local_l2sum=0;
		real local_la1sum=0;
		real local_la2sum=0;
		real diff;
		real an;
		for (int i = nse.blocks.front().offset.x() + 1; i < nse.blocks.front().offset.x() + nse.blocks.front().local.x() - 1; i++)
		for (int j = nse.blocks.front().offset.y() + 1; j < nse.blocks.front().offset.y() + nse.blocks.front().local.y() - 1; j++)
		for (int k = nse.blocks.front().offset.z() + 1; k < nse.blocks.front().offset.z() + nse.blocks.front().local.z() - 1; k++)
		{
			an = analytical_ux(j,k);
			diff = fabs(nse.blocks.front().hmacro(MACRO::e_vx,i,j,k) - an);
			local_la1sum += an;
			local_la2sum += SQ(an);
			local_l1sum += diff;
			local_l2sum += SQ(diff);
		}

		// MPI reduction of the local results
		real l1sum=TNL::MPI::reduce(local_l1sum, MPI_SUM, MPI_COMM_WORLD);
		real l2sum=TNL::MPI::reduce(local_l2sum, MPI_SUM, MPI_COMM_WORLD);
		real la1sum=TNL::MPI::reduce(local_la1sum, MPI_SUM, MPI_COMM_WORLD);
		real la2sum=TNL::MPI::reduce(local_la2sum, MPI_SUM, MPI_COMM_WORLD);

		// considering PHYS_DL, converting to physical units
		real l1error_phys = l1sum*nse.lat.physDl*nse.lat.physDl*nse.lat.physDl;
		real l2error_phys = l2sum*nse.lat.physDl*nse.lat.physDl*nse.lat.physDl;
		l2error_phys = sqrt(l2error_phys);
		l1error_phys = nse.lbm2physVelocity(l1error_phys);
		l2error_phys = nse.lbm2physVelocity(l2error_phys);

		// dynamic stopping criterion
		real threshold = 1e-4;
		real threshold_stddev = 1e-3;
		real l1prev = 0.0;
		for (int i = 0; i < errors_count; i++) l1prev += l1errors[i];
		l1prev /= errors_count;
		real stddev = 0.0;
		for (int i = 0; i < errors_count; i++) stddev += SQ(l1errors[i] - l1prev);
		stddev /= (errors_count-1);
		stddev = sqrt(stddev);
		real stopping = abs(l1prev - l1error_phys) / l1error_phys;
		if( stopping < threshold && stddev < threshold_stddev ) nse.terminate = true;

		error_idx = (error_idx + 1) % errors_count;
		l1errors[error_idx] = l1error_phys;

		if (nse.rank == 0)
			log("at t=%1.2fs, iterations=%d l1error_phys=%e l2error_phys=%e stopping=%e",
				nse.physTime(), nse.iterations, l1error_phys, l2error_phys, stopping);
	}


	StateLocal(const TNL::MPI::Comm& communicator, lat_t ilat, real iphysViscosity, real iphysDt, int RES)
		: State<LBM_TYPE>(communicator, ilat, iphysViscosity, iphysDt)
	{
		errors_count = 10;
		l1errors = new real[errors_count];
		for (int i = 0; i < errors_count; i++) l1errors[i] = 1;
	}

	~StateLocal()
	{
		if (an_cache)
		{
			for (int i=0;i<nse.blocks.front().local.z();i++) delete [] an_cache[i];
			delete [] an_cache;
		}

		delete[] l1errors;
	}
};

template < typename LBM_TYPE >
int sim02(int RES=1, bool use_forcing=true)
{
	using idx = typename LBM_TYPE::TRAITS::idx;
	using real = typename LBM_TYPE::TRAITS::real;
	using dreal = typename LBM_TYPE::TRAITS::dreal;
	using point_t = typename LBM_TYPE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size=32;
	int LBM_X = block_size;
	if (!use_forcing) LBM_X *= RES;
	int LBM_Y = RES*block_size;
	int LBM_Z = RES*block_size;
	// NOTE: LBM_VISCOSITY must be less than 1/6
//	real LBM_VISCOSITY = 0.01*RES;
	real LBM_VISCOSITY = std::min(0.1, 0.01*RES);
	real PHYS_HEIGHT = 0.25;
	real PHYS_VISCOSITY = 1.5e-5;// [m^2/s] fluid viscosity air: 1.81e-5
	real PHYS_DL = PHYS_HEIGHT/((real)LBM_Y-2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY*PHYS_DL*PHYS_DL;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType( LBM_X, LBM_Y, LBM_Z );
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;

	StateLocal<LBM_TYPE> state(MPI_COMM_WORLD, lat, PHYS_VISCOSITY, PHYS_DT, RES);
//	state.nse.use_multiple_gpus = false;
	state.nse.physCharLength = 0.125; // [m]

// NOTE: this is for NSE_Data_ConstInflow
//	if (use_forcing)
//	{
//		state.nse.blocks.front().data.fx = state.nse.phys2lbmForce(1e-4);
//		state.nse.blocks.front().data.fy = 0;
//		state.nse.blocks.front().data.fz = 0;
//		state.nse.blocks.front().data.inflow_vx = 0;
//		state.nse.blocks.front().data.inflow_vy = 0;
//		state.nse.blocks.front().data.inflow_vz = 0;
//	} else
//	{
//		state.nse.blocks.front().data.fx = 0;
//		state.nse.blocks.front().data.fy = 0;
//		state.nse.blocks.front().data.fz = 0;
//		state.nse.blocks.front().data.inflow_vx = state.nse.phys2lbmVelocity(2e-6);
//		state.nse.blocks.front().data.inflow_vy = 0;
//		state.nse.blocks.front().data.inflow_vz = 0;
//	}

// NOTE: this is for NSE_Data_XProfileInflow
	dreal force = 1e-4;
	if (use_forcing)
	{
		state.nse.blocks.front().data.fx = state.nse.phys2lbmForce(force);
		state.nse.blocks.front().data.fy = 0;
		state.nse.blocks.front().data.fz = 0;
		state.nse.blocks.front().data.vx_profile = NULL;
	} else
	{
		// calculate analytical solution using forcing just like above
		state.nse.blocks.front().data.fx = state.nse.phys2lbmForce(force);
		state.nse.blocks.front().data.fy = 0;
		state.nse.blocks.front().data.fz = 0;
		state.cache_analytical();
		// reset the forcing for the LBM simulation
		state.nse.blocks.front().data.fx = 0;

		// allocate array for the inflow profile
		#ifdef USE_CUDA
			cudaMalloc((void**)&state.nse.blocks.front().data.vx_profile, state.nse.blocks.front().local.y()*state.nse.blocks.front().local.z()*sizeof(dreal));
		#else
			state.nse.blocks.front().data.vx_profile = new dreal[state.nse.blocks.front().local.y()*state.nse.blocks.front().local.z()];
		#endif

		#ifdef USE_CUDA
			// convert analytical solution from double to float
			dreal analytical[state.nse.blocks.front().local.y()*state.nse.blocks.front().local.z()];
			for (int j = 0; j < state.nse.blocks.front().local.y(); j++)
			for (int k = 0; k < state.nse.blocks.front().local.z(); k++)
				analytical[k*state.nse.blocks.front().local.y()+j] = state.an_cache[k][j];
			// copy the analytical profile to the GPU
			cudaMemcpy(state.nse.blocks.front().data.vx_profile, analytical, state.nse.blocks.front().local.y()*state.nse.blocks.front().local.z()*sizeof(dreal), cudaMemcpyHostToDevice);
		#else
			for (int j = 0; j < state.nse.blocks.front().local.y(); j++)
			for (int k = 0; k < state.nse.blocks.front().local.z(); k++)
				state.nse.blocks.front().data.vx_profile[k*state.nse.blocks.front().local.y()+j] = state.an_cache[k][j];
		#endif
		state.nse.blocks.front().data.size_y = state.nse.blocks.front().local.y();
	}

	state.cnt[PRINT].period = 10.0;
	state.cnt[PROBE1].period = 10.0;// / RES;
//	state.nse.physFinalTime = PHYS_DT * 1e7;
	state.nse.physFinalTime = 5000;
//	state.cnt[VTK2D].period = 1.0;

	const char* prec = (std::is_same<dreal,float>::value) ? "float" : "double";
	state.setid("sim_2_%s_%s_%s_res_%d", LBM_TYPE::COLL::id, prec, (use_forcing)?"forcing":"velocity", RES);

	if (state.isMark())
		return 0;

	state.log("PHYS_DL = %e", PHYS_DL);
//	state.log("in lbm units: forcing=%e velocity=%e", state.nse.blocks.front().data.fx, state.nse.blocks.front().data.inflow_vx);
	state.log("in lbm units: forcing=%e", force);

	// add cuts
//	state.add2Dcut_X(LBM_X/2,"cut_X");
//	state.add2Dcut_Y(LBM_Y/2,"cut_Y");
//	state.add2Dcut_Z(LBM_Z/2,"cut_Z");
//	state.add1Dcut_Z(LBM_X/2*PHYS_DL, LBM_Y/2*PHYS_DL, "cut_Z");

	execute(state);

	// deallocate inflow data
	if (state.nse.blocks.front().data.vx_profile)
	{
		#ifdef USE_CUDA
			cudaFree(state.nse.blocks.front().data.vx_profile);
		#else
			delete[] state.nse.blocks.front().data.vx_profile;
		#endif
	}

	return 0;
}

template < typename TRAITS=TraitsSP >
void run()
{
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS> >;
//	using COLL = D3Q27_FCLBM<TRAITS>;
//	using COLL = D3Q27_SRT<TRAITS>;
//	using COLL = D3Q27_SRT_WELL<TRAITS>;
//	using COLL = D3Q27_SRT_MODIF_FORCE<TRAITS>;
//	using COLL = D3Q27_BGK<TRAITS>;
//	using COLL = D3Q27_KBC_N1<TRAITS>;
//	using COLL = D3Q27_CUM<TRAITS>;
//	using COLL = D3Q27_CLBM<TRAITS>;
//	using COLL = D3Q27_CLBM_WELL<TRAITS>;
//	using COLL = D3Q27_CUM_SGS<TRAITS>;
//	using COLL = D3Q27_CUM_FIX<TRAITS>;
//	using COLL = D3Q27_CUM_WELL<TRAITS>;

	using NSE_TYPE = D3Q27<
				COLL,
				NSE_Data_XProfileInflow< TRAITS >,
				D3Q27_BC_All,
				typename COLL::EQ,
				D3Q27_STREAMING< TRAITS >,
				D3Q27_MACRO_Default< TRAITS >,
				D3Q27_MACRO_Void< TRAITS >,
				TRAITS
			>;

	bool use_forcing = false;
	for (int i = 2; i <= 4; i++)
	{
//		int res=4;
		int res = pow(2, i);
		sim02<NSE_TYPE>(res, use_forcing);
	}
}

int main(int argc, char **argv)
{
	TNLMPI_INIT mpi(argc, argv);
	run();
	return 0;
}
