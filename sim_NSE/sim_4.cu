#include "lbm3d/core.h"

// ball in 3D
// IBM-LBM

// filament variants
enum { NIC, NORMAL, MIRROR, FLIP, MIRRORFLIP };

//const int DOMAIN_INNER = -1;
//const int DOMAIN_OUTER = -2;
//const int DOMAIN_BNDRY = -3;

template < typename TRAITS >
struct MacroLocal : D3Q27_MACRO_Base< TRAITS >
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum { e_fx, e_fy, e_fz, e_vx, e_vy, e_vz, e_rho, N };

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void outputMacro(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z)  = KS.vx;
		SD.macro(e_vy, x, y, z)  = KS.vy;
		SD.macro(e_vz, x, y, z)  = KS.vz;
	}

	template < typename LBM_DATA >
	CUDA_HOSTDEV static void zeroForces(LBM_DATA &SD, idx x, idx y, idx z)
	{
		SD.macro(e_fx, x, y, z) = 0;
		SD.macro(e_fy, x, y, z) = 0;
		SD.macro(e_fz, x, y, z) = 0;
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.macro(e_fx, x, y, z);
		KS.fy = SD.macro(e_fy, x, y, z);
		KS.fz = SD.macro(e_fz, x, y, z);
	}
};

template < typename NSE >
struct StateLocal : State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK< NSE >;

	using State<NSE>::nse;
	using State<NSE>::vtk_helper;
	using State<NSE>::id;
	using State<NSE>::FF;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	dreal lbm_input_velocity=0.07;
//	dreal start_velocity;
//	dreal target_velocity;
//	real init_time;
	bool firstrun=true;
//	bool firstplot=true;
	int FIL_INDEX=-1;
	real ball_diameter=0.01;
	real ball_c[3];

	virtual bool outputData(const BLOCK& block, int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs)
	{
		int k=0;
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vx,x,y,z), 3, desc, value, dofs);
				case 1: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vy,x,y,z), 3, desc, value, dofs);
				case 2: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vz,x,y,z), 3, desc, value, dofs);
			}
		}
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("lbm_force", block.hmacro(MACRO::e_fx,x,y,z), 3, desc, value, dofs);
				case 1: return vtk_helper("lbm_force", block.hmacro(MACRO::e_fy,x,y,z), 3, desc, value, dofs);
				case 2: return vtk_helper("lbm_force", block.hmacro(MACRO::e_fz,x,y,z), 3, desc, value, dofs);
			}
		}
		if (index==k++) return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_density_fluctuation", block.hmacro(MACRO::e_rho,x,y,z)-1.0, 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vz,x,y,z)), 3, desc, value, dofs);
			}
		}
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fz,x,y,z)), 3, desc, value, dofs);
			}
		}
		//if (index==k++) return vtk_helper("density", block.hmacro(MACRO::e_rho,x,y,z)*nse.physFluidDensity, 1, desc, value, dofs);
		return false;
	}

	virtual void probe1()
	{
		// compute drag
		real Fx=0, Fy=0, Fz=0, dV=nse.lat.physDl*nse.lat.physDl*nse.lat.physDl;
		real rho = 1.0;//nse.physFluidDensity;
		real target_velocity = nse.lat.lbm2physVelocity(lbm_input_velocity);
		spdlog::info("Reynolds = {:f} lbmvel {:f} physvel {:f}", lbm_input_velocity*ball_diameter/nse.lat.physDl/nse.lat.lbmViscosity(), lbm_input_velocity, nse.lat.lbm2physVelocity(lbm_input_velocity));

		// FIXME: MPI !!!
		// todo: compute C_D: integrate over the whole domain
		for (int x=0; x<nse.lat.global.x(); x++)
		for (int y=0; y<nse.lat.global.y(); y++)
		for (int z=0; z<nse.lat.global.z(); z++)
		{
			// test if outside the ball
//			if (NORM(x*nse.lat.physDl - ball_c[0], y*nse.lat.physDl - ball_c[1], z*nse.lat.physDl - ball_c[2]) > 2.0*ball_diameter/2.0)
//			if (NORM(x*nse.lat.physDl - ball_c[0], y*nse.lat.physDl - ball_c[1], z*nse.lat.physDl - ball_c[2]) > ball_diameter/2.0)
			{
				Fx += nse.blocks.front().hmacro(MACRO::e_fx,x,y,z);
				Fy += nse.blocks.front().hmacro(MACRO::e_fy,x,y,z);
				Fz += nse.blocks.front().hmacro(MACRO::e_fz,x,y,z);
			}
		}

		real lbm_cd_full=-Fx*8.0/lbm_input_velocity/lbm_input_velocity/PI/ball_diameter/ball_diameter*nse.lat.physDl*nse.lat.physDl;
		real phys_cd_full=-nse.lat.lbm2physForce(Fx)*dV*8.0/rho/target_velocity/target_velocity/PI/ball_diameter/ball_diameter;
		if (std::isnan(Fx) || std::isnan(Fz) || std::isnan(Fz)) {
			if (!nse.terminate)
				spdlog::error("nan detected");
			nse.terminate=true;
		}
		spdlog::info("FULL: u0 {:e} Fx {:e} Fy {:e} Fz {:e} C_D{{phys}} {:e} C_D{{LB}} {:f}", lbm_input_velocity, Fx, Fy, Fz, phys_cd_full, lbm_cd_full);

// not used for evaluation of the results
//		// FIXME: MPI !!!
//		for (int x=0; x<nse.lat.global.x(); x++)
//		for (int y=0; y<nse.lat.global.y(); y++)
//		for (int z=0; z<nse.lat.global.z(); z++)
//		{
//			// test if outside the ball
////			if (NORM(x*nse.lat.physDl - ball_c[0], y*nse.lat.physDl - ball_c[1], z*nse.lat.physDl - ball_c[2]) < 2.0*ball_diameter/2.0)
//			if (NORM(x*nse.lat.physDl - ball_c[0], y*nse.lat.physDl - ball_c[1], z*nse.lat.physDl - ball_c[2]) > ball_diameter/2.0)
//			{
//				Fx += nse.blocks.front().hmacro(MACRO::e_fx,x,y,z);
//				Fy += nse.blocks.front().hmacro(MACRO::e_fy,x,y,z);
//				Fz += nse.blocks.front().hmacro(MACRO::e_fz,x,y,z);
//			}
//		}
//		real lbm_cd=-Fx*8.0/lbm_input_velocity/lbm_input_velocity/PI/ball_diameter/ball_diameter*nse.lat.physDl*nse.lat.physDl;
//		real phys_cd=-nse.lat.lbm2physForce(Fx)*dV*8.0/rho/target_velocity/target_velocity/PI/ball_diameter/ball_diameter;
//		if (std::isnan(Fx) || std::isnan(Fz) || std::isnan(Fz)) { if (!nse.terminate) spdlog::error("nan detected"); nse.terminate=true; }
//		spdlog::info("INNN: u0 {:e} Fx {:e} Fy {:e} Fz {:e} C_D{{phys}} {:e} C_D{{LB}} {:f}", lbm_input_velocity, Fx, Fy, Fz, phys_cd, lbm_cd);
////		spdlog::info("Reynolds = {:f} lbmvel 0.07 physvel {:f}", 0.07*ball_diameter/nse.lat.physDl/nse.lbmViscosity(), lbm_input_velocity);

// not used for evaluation of the results
////		real fil_fx=0,fil_fy=0,fil_fz=0;
//		Fx=Fy=Fz=0;
////		// FIXME - integrateForce is not implemented - see _stare_verze_/iblbm3d_verze1/filament_3D.h*
////		if (FIL_INDEX>=0) FF[FIL_INDEX].integrateForce(Fx,Fy,Fz, 1.0);//PI*ball_diameter*ball_diameter/(real)FF[FIL_INDEX].LL.size());
//		real lbm_cd_lagr=-Fx*8.0/lbm_input_velocity/lbm_input_velocity/PI/ball_diameter/ball_diameter*nse.lat.physDl*nse.lat.physDl;
//		real phys_cd_lagr=-nse.lat.lbm2physForce(Fx)*dV*8.0/rho/target_velocity/target_velocity/PI/ball_diameter/ball_diameter;
//		if (std::isnan(Fx) || std::isnan(Fz) || std::isnan(Fz)) { if (!nse.terminate) spdlog::error("nan detected"); nse.terminate=true; }
//		spdlog::info("LAGR: u0 {:e} Fx {:e} Fy {:e} Fz {:e} C_D{{phys}} {:e} C_D{{LB}} {:f}", lbm_input_velocity, Fx, Fy, Fz, phys_cd_lagr, lbm_cd_lagr);


		// empty files
		const char* iotype = (firstrun) ? "wt" : "at";
		firstrun=false;
		// output
		FILE* f;
		//real total = (real)(nse.lat.global.x()*nse.lat.global.y()*nse.lat.global.z()), ratio, area;
		const std::string dir = fmt::format("results_{}/probes", id);
		mkdir_p(dir.c_str(), 0755);

		std::string str = fmt::format("{}/probe_cd_full", dir);
		f = fopen(str.c_str(), iotype);
		fprintf(f, "%e\t%e\n", nse.physTime(), lbm_cd_full);
		fclose(f);

//		str = fmt::format("{}/probe_cd", dir);
//		f = fopen(str.c_str(), iotype);
//		fprintf(f, "%e\t%e\n", nse.physTime(), lbm_cd);
//		fclose(f);

//		str = fmt::format("{}/probe_cd_lagr", dir);
//		f = fopen(str.c_str(), iotype);
//		fprintf(f, "%e\t%e\n", nse.physTime(), lbm_cd_lagr);
//		fclose(f);

//		str = fmt::format("{}/probe_cd_all", dir);
//		f = fopen(str.c_str(), iotype);
//		fprintf(f, "%e\t%e\t%e\t%e\n", nse.physTime(), lbm_cd_full, lbm_cd, lbm_cd_lagr);
//		fclose(f);
	}

	virtual void updateKernelVelocities()
	{
		for (auto& block : nse.blocks)
		{
			block.data.inflow_vx = lbm_input_velocity;
//			block.data.inflow_vx = nse.lat.phys2lbmVelocity(target_velocity);
		}
	}

	virtual void setupBoundaries()
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW); // left
		nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_OUTFLOW_EQ);// right
		nse.setBoundaryY(0, BC::GEO_INFLOW); // back
		nse.setBoundaryY(nse.lat.global.y()-1, BC::GEO_INFLOW);// front
		nse.setBoundaryZ(0, BC::GEO_INFLOW);// top
		nse.setBoundaryZ(nse.lat.global.z()-1, BC::GEO_INFLOW);// bottom
	}

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat)
		: State<NSE>(id, communicator, lat)
	{
		for (auto& block : nse.blocks)
		{
			block.data.inflow_rho = no1;
			block.data.inflow_vx = 0;
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}
};

// ball discretization algorithm: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
template < typename STATE >
int drawFixedSphere(STATE &state, double cx, double cy, double cz, double radius, double sigma, int method=0, int dirac_delta=1, int WuShuCompute=ws_computeGPU_CUSPARSE)
{
	using real = typename STATE::TRAITS::real;

	// based on sigma, estimate N
	real surface = 4.0*PI*radius*radius;
	// sigma is diagonal of a "quasi-square" that 4 points on the sphere surface form
	// so the quasi-square has area = b^2 where b^2 + b^2 = sigma^2, i.e., b^2 = 1/2*sigma^2
	real wanted_unit_area = sigma*sigma / 2.0;
	// count how many of these
	real count = surface/wanted_unit_area;
	int N = ceil(count);

	LagrangePoint3D<real> fp;
	real theta, phi;
	int INDEX = state.addLagrange3D();
	int points=0;
//	real a = 4.0*PI*radius*radius/(real)N;
	real a = 4.0*PI/(real)N;
	real d = sqrt(a);
//	int Mtheta = (int)(PI/d);
	int Mtheta = floor(PI/d);
	real dtheta = PI/Mtheta;
	real dphi = a/dtheta;
	for (int m = 0; m < Mtheta; m++)
	{
		// for a given phi and theta:
		theta = PI*(m+0.5)/(real)Mtheta;
//		int Mphi = (int)(2.0*PI*sin(theta)/dphi);
		int Mphi = floor(2.0*PI*sin(theta)/dphi);
		for (int n = 0; n<Mphi; n++)
		{
			phi = 2.0*PI*n/(real)Mphi;
			fp.x = cx + radius * cos( phi ) * sin( theta );
			fp.y = cy + radius * sin( phi ) * sin( theta );
			fp.z = cz + radius * cos( theta );
			fp.x_ref = fp.x;
			fp.y_ref = fp.y;
			fp.z_ref = fp.z;
			state.FF[INDEX].LL.push_back(fp);
			points++;
		}
	}
	state.FF[INDEX].ws_compute = WuShuCompute; // given by the argument
	state.FF[INDEX].diracDeltaType = dirac_delta;
	state.FF[INDEX].ws_regularDirac=(method==0)?true:false;
	state.FIL_INDEX=INDEX;
	spdlog::info("added {} lagrangian points", points);

	real sigma_min = state.FF[INDEX].computeMinDist();
	real sigma_max = state.FF[INDEX].computeMaxDistFromMinDist(sigma_min);

	spdlog::info("Ball surface: wanted sigma {:e} ({:f} i.e. {:d} points), wanted_unit_area {:e}, sigma_min {:e}, sigma_max {:e}", sigma, count, N, wanted_unit_area, sigma_min, sigma_max);
//	spdlog::info("Added {} Lagrangian points (requested {}) partial area {:e}",points, N, a);
//	spdlog::info("Lagrange created: WuShuCompute {} ws_regularDirac {}", state.FF[INDEX].WuShuCompute, (state.FF[INDEX].ws_regularDirac)?"true":"false");
	spdlog::info("h=physdl {:e} sigma min {:e} sigma_ku_h {:e}", state.nse.lat.physDl, sigma_min, sigma_min/state.nse.lat.physDl);
	spdlog::info("h=physdl {:e} sigma max {:e} sigma_ku_h {:e}", state.nse.lat.physDl, sigma_max, sigma_max/state.nse.lat.physDl);

	state.writeVTK_Points("ball.vtk",0,0,state.FF[INDEX]);
	return INDEX;
}

template < typename NSE >
int sim(int RES=2, double i_Re=1000, double nasobek=2.0, int dirac_delta=2, int method=0, int compute=5)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size=32;
	real BALL_DIAMETER = 0.01;
	real real_domain_height= BALL_DIAMETER*11;// [m]
	//real real_domain_length= BALL_DIAMETER*11;// [m] // extra 1cm on both sides
	idx LBM_Y = RES*block_size; // for 4 cm
	idx LBM_Z = LBM_Y;
	real PHYS_DL = real_domain_height/((real)LBM_Y);
	idx LBM_X = LBM_Y;//(int)(real_domain_length/PHYS_DL)+2;//block_size;//16*RESOLUTION;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// zvolit Re + LBM VELOCITY + PHYS_VISCOSITY
//	real i_Re = ;
	real i_LBM_VELOCITY = 0.07; // Geier
	real i_PHYS_VISCOSITY = 0.00001; // proc ne?
	// mam:
	real i_LBM_VISCOSITY = i_LBM_VELOCITY * BALL_DIAMETER / PHYS_DL / i_Re;
	real i_PHYS_VELOCITY = i_PHYS_VISCOSITY * i_Re / BALL_DIAMETER;
	fmt::print("input phys velocity {:f}\ninput lbm velocity {:f}\nRe {:f}\nlbm viscosity{:f}\nphys viscosity {:f}\n", i_PHYS_VELOCITY, i_LBM_VELOCITY, i_Re, i_LBM_VISCOSITY, i_PHYS_VISCOSITY);

	real LBM_VISCOSITY = i_LBM_VISCOSITY;// 0.0001*RES;//*SIT;//1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_VISCOSITY = i_PHYS_VISCOSITY;//0.00001;// [m^2/s] fluid viscosity of water
	real Re=i_Re;//200;

//	real PHYS_TARGET_VELOCITY = i_PHYS_VELOCITY;//2.0*Re*PHYS_VISCOSITY/BALL_DIAMETER; // [m/s]
//	real PHYS_START_VELOCITY = PHYS_TARGET_VELOCITY; // [m/s]
//	real INIT_TIME = 1.0; // [s]
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY*PHYS_DL*PHYS_DL;

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType( LBM_X, LBM_Y, LBM_Z );
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim_4_{}_{}_dirac_{}_res_{}_Re_{}_nas_{:05.4f}_compute_{}", NSE::COLL::id, (method>0)?"original":"modified", dirac_delta, RES, Re, nasobek, compute);
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

	if (state.isMark())
		return 0;

	state.lbm_input_velocity = i_LBM_VELOCITY;
	state.nse.physCharLength = BALL_DIAMETER; // [m]
	state.ball_diameter = BALL_DIAMETER; // [m]
	//state.nse.physFluidDensity = 1000.0; // [kg/m^3]

	state.cnt[PRINT].period = 0.1;
	state.cnt[PROBE1].period = 0.1;
	state.cnt[STAT_RESET].period = 500.0;
	state.nse.physFinalTime = 30.0;

//	state.cnt[VTK3D].period = 1.0;
	state.cnt[VTK2D].period = 1.0;
	state.cnt[VTK1D].period = 1.0;

	// select compute method
	int ws_compute;
	switch (compute)
	{
		case 1: ws_compute = ws_computeCPU; break;
		case 2: ws_compute = ws_computeGPU_CUSPARSE; break;
		case 3: ws_compute = ws_computeHybrid_CUSPARSE; break;
		case 4: ws_compute = ws_computeCPU_TNL; break;
		case 5: ws_compute = ws_computeGPU_TNL; break;
		case 6: ws_compute = ws_computeHybrid_TNL; break;
		case 7: ws_compute = ws_computeHybrid_TNL_zerocopy; break;
		default: spdlog::warn("Unknown parameter compute={}, selecting default ws_computeGPU_TNL.", compute); ws_compute = ws_computeGPU_TNL; break;
	}

	// add cuts
	state.add2Dcut_X(LBM_X/2,"cut_X");
//	state.add2Dcut_X(2*BALL_DIAMETER/PHYS_DL,"cut_Xball");
	state.add2Dcut_Y(LBM_Y/2,"cut_Y");
	state.add2Dcut_Z(LBM_Z/2,"cut_Z");

	state.ball_c[0] = 2*state.ball_diameter;
	state.ball_c[1] = 5.5*state.ball_diameter;
	state.ball_c[2] = 5.5*state.ball_diameter;
	// create a filament
	real sigma = nasobek * PHYS_DL;
	state.FIL_INDEX = drawFixedSphere(state, state.ball_c[0], state.ball_c[1], state.ball_c[2], state.ball_diameter/2.0, sigma, method, dirac_delta, ws_compute);

	// 2nd ball
	state.ball_c[0] = 5.5*state.ball_diameter;
	state.FIL_INDEX = drawFixedSphere(state, state.ball_c[0], state.ball_c[1], state.ball_c[2], state.ball_diameter/2.0, sigma, method, dirac_delta, ws_compute);

	execute(state);

	return 0;
}

template < typename TRAITS=TraitsSP >
void run(int res, double Re, double h, int dirac, int method, int compute)
{
	using COLL = D3Q27_CUM<TRAITS>;
	using NSE_CONFIG = LBM_CONFIG<
				TRAITS,
				D3Q27_KernelStruct,
				NSE_Data_ConstInflow< TRAITS >,
				COLL,
				typename COLL::EQ,
				D3Q27_STREAMING< TRAITS >,
				D3Q27_BC_All,
				MacroLocal< TRAITS >,
				D3Q27_MACRO_Void< TRAITS >
			>;

	sim<NSE_CONFIG>(res, Re, h, dirac, method, compute);
}

int main(int argc, char **argv)
{
	TNLMPI_INIT mpi(argc, argv);

	// here "h" is the Lagrangian/Eulerian spacing ratio
//	const double hvals[] = { 2.0, 1.5, 1.0, 0.75, 0.50, 0.25 };
//	const double hvals[] = { 2.0 };
//	const double hvals[] = { 0.125 };
	const double hvals[] = { 0.25, 0.5, 0.75, 1.0, 1.5, 2.0 };
//	const double hvals[] = { 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.75, 0.50 };
//	const double hvals[] = { 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.75, 0.50, 0.25, 0.125 };
//	const double hvals[] = { 3.0, 2.5, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 }; // ostrava
	int hmax = sizeof(hvals)/sizeof(double);
	double h=1.0;

	bool use_command_line=true;

	if (!use_command_line)
	{
		int dirac=1;
		int res=2; // 3, 6, 12
		int hi=3;
		int method=0; //0 = modified
		int Re=100;
		int compute=5;
//		for (int Re=100;Re<=200; Re+=100)
//		for (hi=0;hi<hmax;hi++)
//		for (method=0;method<=1;method++)
//		for (res=3;res<=5;res++)
//		for (dirac=1;dirac<=4;dirac++)
//		for (compute=1;compute<=6;compute++)
		{
			if (hi<hmax) h=hvals[hi];
			run(res, (double)Re, h, dirac, method, compute);
		}
	} else
	{
		const int pars=6;
		if (argc <= pars)
		{
			fprintf(stderr, "error: %d parameters required:\n %s method{0,1} dirac{1,2,3,4} Re{100,200} hi[0,%d] res[1,22] compute[1,7]\n", pars, argv[0],hmax-1);
			return 1;
		}
		int method = atoi(argv[1]);	// 0=modified 1=original
		int dirac = atoi(argv[2]);
		int Re = atoi(argv[3]);		// type=0,1,2 (geometry selection)
		int hi = atoi(argv[4]);		// index in the hvals
		int res = atoi(argv[5]);	// res=1,2,3
		int compute = atoi(argv[6]); // compute=1,2,3,4,5,6,7

		if (method > 1 || method < 0) { fprintf(stderr, "error: method=%d out of bounds [0, 1]\n",method); return 1; }
		if (dirac < 1 || dirac > 4) { fprintf(stderr, "error: dirac=%d out of bounds [1,4]\n",dirac); return 1; }
		if (hi >= hmax || hi < 0) { fprintf(stderr, "error: hi=%d out of bounds [0, %d]\n",hi,hmax-1); return 1; }
		if (res < 1) { fprintf(stderr, "error: res=%d out of bounds [1, ...]\n",res); return 1; }
		if (compute < 1 || compute > 7) { fprintf(stderr, "error: compute=%d out of bounds [1,7]\n",compute); return 1; }
		if (hi<hmax) h=hvals[hi];
		run(res, (double)Re, h, dirac, method, compute);
	}
	return 0;
}
