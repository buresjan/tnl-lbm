#include "core.h"

// 3D test problem: forcing/input velocity
// analytical solution for rectangular duct: forcing accelerated

template < typename TRAITS >
struct LBM_Data_XProfileInflow : LBM_Data < TRAITS >
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

	using State<LBM_TYPE>::lbm;
	using State<LBM_TYPE>::vtk_helper;
	using State<LBM_TYPE>::log;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;

	real **an_cache=0;
	int an_n=50;

	int errors_count;
	real* l1errors;
	int error_idx = 0;

	real raw_analytical_ux(int n, int ilbm_y, int ilbm_z)
	{
		if (ilbm_y==0 || ilbm_y==lbm.global_Y-1 || ilbm_z==0 || ilbm_z==lbm.global_Z-1) return 0;
		int lbm_y=ilbm_y;
		int lbm_z=ilbm_z;
		real a = lbm.global_Y/2.0 - 1.0;
		real b = lbm.global_Z/2.0 - 1.0;
		real y = ((real)lbm_y + 0.5 - lbm.global_Y/2.)/a;
		real z = ((real)lbm_z + 0.5 - lbm.global_Z/2.)/a;
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

//		real coef = (lbm.data.fx != 0) ? lbm.data.fx : lbm.data.inflow_vx;
		real coef = lbm.data.fx;
		return coef * 16.0*a*a/PI/PI/PI*sum/lbm.lbmViscosity();
	}

	real analytical_ux(int lbm_y, int lbm_z)
	{
		if (!an_cache) cache_analytical();
		return an_cache[lbm_z][lbm_y];
	}

	void cache_analytical()
	{
		an_cache = new real*[lbm.local_Z];
		for (int i=0;i<lbm.local_Z;i++) an_cache[i] = new real[lbm.local_Y];
		#pragma omp parallel for schedule(static) collapse(2)
		for (int z=0;z<lbm.local_Z;z++)
		for (int y=0;y<lbm.local_Y;y++)
			an_cache[z][y] = raw_analytical_ux(an_n, lbm.offset_Y + y, lbm.offset_Z + z);
	}

	virtual void setupBoundaries()
	{
//		if (lbm.data.inflow_vx != 0)
		if (lbm.data.vx_profile)
		{
//			lbm.setBoundaryX(0, BC::GEO_INFLOW); 		// left
			lbm.setBoundaryX(0, BC::GEO_INFLOW_FREE_RHO); 		// left
//			lbm.setBoundaryX(lbm.global_X-1, BC::GEO_OUTFLOW_EQ);		// right
//			lbm.setBoundaryX(lbm.global_X-1, BC::GEO_OUTFLOW_RIGHT);		// right
			lbm.setBoundaryX(lbm.global_X-1, BC::GEO_OUTFLOW_RIGHT_INTERP);		// right
		} else
		{
			lbm.setBoundaryX(0, BC::GEO_PERIODIC); 		// left
			lbm.setBoundaryX(lbm.global_X-1, BC::GEO_PERIODIC);		// right
		}
		lbm.setBoundaryZ(0, BC::GEO_WALL);		// top
		lbm.setBoundaryZ(lbm.global_Z-1, BC::GEO_WALL);	// bottom
		lbm.setBoundaryY(0, BC::GEO_WALL); 		// back
		lbm.setBoundaryY(lbm.global_Y-1, BC::GEO_WALL);		// front
	}

	virtual bool outputData(int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs)
	{
		int k=0;
		if (index==k++) return vtk_helper("lbm_density", lbm.hmacro(MACRO::e_rho,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_delta_density", lbm.hmacro(MACRO::e_rho,x,y,z) - 1.0, 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("lbm_velocity", lbm.hmacro(MACRO::e_vx,x,y,z), 3, desc, value, dofs);
				case 1: return vtk_helper("lbm_velocity", lbm.hmacro(MACRO::e_vy,x,y,z), 3, desc, value, dofs);
				case 2: return vtk_helper("lbm_velocity", lbm.hmacro(MACRO::e_vz,x,y,z), 3, desc, value, dofs);
			}
		}
		if (index==k++) return vtk_helper("lbm_analytical_ux", analytical_ux(y, z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_ux", lbm.hmacro(MACRO::e_vx,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uy", lbm.hmacro(MACRO::e_vy,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uz", lbm.hmacro(MACRO::e_vz,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_error_ux", fabs(lbm.hmacro(MACRO::e_vx,x,y,z) - analytical_ux(y,z)), 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("velocity", lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("velocity", lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("velocity", lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vz,x,y,z)), 3, desc, value, dofs);
			}
		}
		if (index==k++) return vtk_helper("lbm_analytical_ux", lbm.lbm2physVelocity(analytical_ux(y, z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_ux", lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vx,x,y,z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uy", lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vy,x,y,z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uz", lbm.lbm2physVelocity(lbm.hmacro(MACRO::e_vz,x,y,z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_error_ux", lbm.lbm2physVelocity(fabs(lbm.hmacro(MACRO::e_vx,x,y,z) - analytical_ux(y,z))), 1, desc, value, dofs);
		return false;
	}

/*
	// clear Lattice + boundary setup
	virtual void resetLattice(real irho, real ivx, real ivy, real ivz)
	{
		for (int x = 0; x < lbm.X;x++)
		for (int y = 0; y < lbm.Y;y++)
		for (int z = 0; z < lbm.Z;z++)
		{
			dreal rho= no1;
			dreal vx= 0;//(lbmInputVelocityX!=0) ? (dreal)analytical_ux(y,z) : 0;
			dreal vy= 0;
			dreal vz= 0;
			lbm.setEqLat(x,y,z,rho,vx,vy,vz);
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
		for (int i = lbm.offset_X + 1; i < lbm.offset_X + lbm.local_X - 1; i++)
		for (int j = lbm.offset_Y + 1; j < lbm.offset_Y + lbm.local_Y - 1; j++)
		for (int k = lbm.offset_Z + 1; k < lbm.offset_Z + lbm.local_Z - 1; k++)
		{
			an = analytical_ux(j,k);
			diff = fabs(lbm.hmacro(MACRO::e_vx,i,j,k) - an);
			local_la1sum += an;
			local_la2sum += SQ(an);
			local_l1sum += diff;
			local_l2sum += SQ(diff);
		}

		// MPI reduction of the local results
		real l1sum=0;
		real l2sum=0;
		real la1sum=0;
		real la2sum=0;
		TNL::MPI::Allreduce(&local_l1sum, &l1sum, 1, MPI_SUM, TNL::MPI::AllGroup());
		TNL::MPI::Allreduce(&local_l2sum, &l2sum, 1, MPI_SUM, TNL::MPI::AllGroup());
		TNL::MPI::Allreduce(&local_la1sum, &la1sum, 1, MPI_SUM, TNL::MPI::AllGroup());
		TNL::MPI::Allreduce(&local_la2sum, &la2sum, 1, MPI_SUM, TNL::MPI::AllGroup());

		// Chinese version
		real l1error_chinese = l1sum / la1sum;
		real l2error_chinese = l2sum / la2sum;
		l2error_chinese = sqrt(l2error_chinese);
		// considering PHYS_DL, converting to physical units
		real l1error_phys = l1sum*lbm.physDl*lbm.physDl*lbm.physDl;
		real l2error_phys = l2sum*lbm.physDl*lbm.physDl*lbm.physDl;
		l2error_phys = sqrt(l2error_phys);
		l1error_phys = lbm.lbm2physVelocity(l1error_phys);
		l2error_phys = lbm.lbm2physVelocity(l2error_phys);

		// dynamic stopping criterion
//		real threshold = 1e-6;
//		real l1prev = l1errors[(error_idx - errors_count) % errors_count];
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
		if( stopping < threshold && stddev < threshold_stddev ) lbm.terminate = true;

		error_idx = (error_idx + 1) % errors_count;
		l1errors[error_idx] = l1error_phys;

		if (lbm.rank == 0)
			log("at t=%1.2fs, iterations=%d l1error_chinese=%e l2error_chinese=%e l1error_phys=%e l2error_phys=%e\tstopping=%e",
				lbm.physTime(), lbm.iterations, l1error_chinese, l2error_chinese, l1error_phys, l2error_phys, stopping);
	}


	StateLocal(int iX, int iY, int iZ, real iphysViscosity, real iphysDl, real iphysDt, point_t iphysOrigin, int RES)
		: State<LBM_TYPE>(iX, iY, iZ, iphysViscosity, iphysDl, iphysDt, iphysOrigin)
	{
		errors_count = 10 * RES;
		l1errors = new real[errors_count];
		for (int i = 0; i < errors_count; i++) l1errors[i] = 1e18;
	}

	~StateLocal()
	{
		if (an_cache)
		{
			for (int i=0;i<lbm.local_Z;i++) delete [] an_cache[i];
			delete [] an_cache;
		}

		delete[] l1errors;
	}
};

template < typename LBM_TYPE >
int sim02(int RES=1, bool use_forcing=true)
{
	using real = typename LBM_TYPE::TRAITS::real;
	using dreal = typename LBM_TYPE::TRAITS::dreal;
	using point_t = typename LBM_TYPE::TRAITS::point_t;

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

	StateLocal<LBM_TYPE> state(LBM_X, LBM_Y, LBM_Z, PHYS_VISCOSITY, PHYS_DL, PHYS_DT, PHYS_ORIGIN, RES);
	state.lbm.block_size=block_size;
//	state.lbm.use_multiple_gpus = false;
	state.lbm.physCharLength = 0.125; // [m]

// NOTE: this is for LBM_Data_ConstInflow
//	if (use_forcing)
//	{
//		state.lbm.data.fx = state.lbm.phys2lbmForce(1e-4);
//		state.lbm.data.fy = 0;
//		state.lbm.data.fz = 0;
//		state.lbm.data.inflow_vx = 0;
//		state.lbm.data.inflow_vy = 0;
//		state.lbm.data.inflow_vz = 0;
//	} else
//	{
//		state.lbm.data.fx = 0;
//		state.lbm.data.fy = 0;
//		state.lbm.data.fz = 0;
//		state.lbm.data.inflow_vx = state.lbm.phys2lbmVelocity(2e-6);
//		state.lbm.data.inflow_vy = 0;
//		state.lbm.data.inflow_vz = 0;
//	}

// NOTE: this is for LBM_Data_XProfileInflow
	dreal force = 1e-4;
	if (use_forcing)
	{
		state.lbm.data.fx = state.lbm.phys2lbmForce(force);
		state.lbm.data.fy = 0;
		state.lbm.data.fz = 0;
		state.lbm.data.vx_profile = NULL;
	} else
	{
		// calculate analytical solution using forcing just like above
		state.lbm.data.fx = state.lbm.phys2lbmForce(force);
		state.lbm.data.fy = 0;
		state.lbm.data.fz = 0;
		state.cache_analytical();
		// reset the forcing for the LBM simulation
		state.lbm.data.fx = 0;

		// allocate array for the inflow profile
		#ifdef USE_CUDA
			cudaMalloc((void**)&state.lbm.data.vx_profile, state.lbm.local_Y*state.lbm.local_Z*sizeof(dreal));
		#else
			state.lbm.data.vx_profile = new dreal[state.lbm.local_Y*state.lbm.local_Z];
		#endif

		#ifdef USE_CUDA
			// convert analytical solution from double to float
			dreal analytical[state.lbm.local_Y*state.lbm.local_Z];
			for (int j = 0; j < state.lbm.local_Y; j++)
			for (int k = 0; k < state.lbm.local_Z; k++)
				analytical[k*state.lbm.local_Y+j] = state.an_cache[k][j];
			// copy the analytical profile to the GPU
			cudaMemcpy(state.lbm.data.vx_profile, analytical, state.lbm.local_Y*state.lbm.local_Z*sizeof(dreal), cudaMemcpyHostToDevice);
		#else
			for (int j = 0; j < state.lbm.local_Y; j++)
			for (int k = 0; k < state.lbm.local_Z; k++)
				state.lbm.data.vx_profile[k*state.lbm.local_Y+j] = state.an_cache[k][j];
		#endif
		state.lbm.data.size_y = state.lbm.local_Y;
	}

	state.cnt[PRINT].period = 10.0;
	state.cnt[PROBE1].period = 10.0;// / RES;
//	state.lbm.physFinalTime = PHYS_DT * 1e7;
	state.lbm.physFinalTime = 5000;
//	state.vtk.writePeriod = 1.0;

	const char* prec = (std::is_same<dreal,float>::value) ? "float" : "double";
	state.setid("sim_2_%s_%s_%s_res_%d", LBM_TYPE::COLL::id, prec, (use_forcing)?"forcing":"velocity", RES);

	if (state.isMark())
		return 0;

	state.log("PHYS_DL = %e", PHYS_DL);
//	state.log("in lbm units: forcing=%e velocity=%e", state.lbm.data.fx, state.lbm.data.inflow_vx);
	state.log("in lbm units: forcing=%e", force);

	// add cuts
//	state.add2Dcut_X(LBM_X/2,"cut_X");
//	state.add2Dcut_Y(LBM_Y/2,"cut_Y");
//	state.add2Dcut_Z(LBM_Z/2,"cut_Z");
//	state.add1Dcut_Z(LBM_X/2*PHYS_DL, LBM_Y/2*PHYS_DL, "cut_Z");

	execute(state);

	// deallocate inflow data
	if (state.lbm.data.vx_profile)
	{
		#ifdef USE_CUDA
			cudaFree(state.lbm.data.vx_profile);
		#else
			delete[] state.lbm.data.vx_profile;
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
				LBM_Data_XProfileInflow< TRAITS >,
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

int Main(int argc, char **argv)
{
	TNLMPI_INIT mpi(argc, argv);
	run();
	return 0;
}
