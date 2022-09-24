#include "core.h"

// ball in 3D
// ibmlbm

// filament varianty
enum { NIC, NORMAL, MIRROR, FLIP, MIRRORFLIP };

const int DOMAIN_INNER = -1;
const int DOMAIN_OUTER = -2;
const int DOMAIN_BNDRY = -3;

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

template < typename LBM_TYPE >
struct StateLocal : State<LBM_TYPE>
{
	using TRAITS = typename LBM_TYPE::TRAITS;
	using BC = typename LBM_TYPE::BC;
	using MACRO = typename LBM_TYPE::MACRO;
	using BLOCK = LBM_BLOCK< LBM_TYPE >;

	using State<LBM_TYPE>::nse;
	using State<LBM_TYPE>::log;
	using State<LBM_TYPE>::vtk_helper;
	using State<LBM_TYPE>::id;
	using State<LBM_TYPE>::FF;

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

	// virtualize
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
				case 0: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(MACRO::e_vx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(MACRO::e_vy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(MACRO::e_vz,x,y,z)), 3, desc, value, dofs);
			}
		}
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("force", nse.lbm2physForce(block.hmacro(MACRO::e_fx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("force", nse.lbm2physForce(block.hmacro(MACRO::e_fy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("force", nse.lbm2physForce(block.hmacro(MACRO::e_fz,x,y,z)), 3, desc, value, dofs);
			}
		}
		//if (index==k++) return vtk_helper("density", block.hmacro(MACRO::e_rho,x,y,z)*nse.physFluidDensity, 1, desc, value, dofs);
		return false;
	}

	virtual void statReset()
	{
	}

	virtual void probe1()
	{
		// compute drag
		real Fx=0, Fy=0, Fz=0, dV=nse.lat.physDl*nse.lat.physDl*nse.lat.physDl;
		real rho = 1.0;//nse.physFluidDensity;
		real target_velocity = nse.lbm2physVelocity(lbm_input_velocity);
		log("Reynolds = %f lbmvel %f physvel %f",lbm_input_velocity*ball_diameter/nse.lat.physDl/nse.lbmViscosity(), lbm_input_velocity, nse.lbm2physVelocity(lbm_input_velocity));

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
		real phys_cd_full=-nse.lbm2physForce(Fx)*dV*8.0/rho/target_velocity/target_velocity/PI/ball_diameter/ball_diameter;
		if (std::isnan(Fx) || std::isnan(Fz) || std::isnan(Fz)) { if (!nse.terminate) log("nan detected"); nse.terminate=true; }
		log("FULL: u0 %e Fx %e Fy %e Fz %e C_D{phys} %e C_D{LB} %f", lbm_input_velocity, Fx, Fy, Fz, phys_cd_full, lbm_cd_full);

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
//		real phys_cd=-nse.lbm2physForce(Fx)*dV*8.0/rho/target_velocity/target_velocity/PI/ball_diameter/ball_diameter;
//		if (std::isnan(Fx) || std::isnan(Fz) || std::isnan(Fz)) { if (!nse.terminate) log("nan detected"); nse.terminate=true; }
//		log("INNN: u0 %e Fx %e Fy %e Fz %e C_D{phys} %e C_D{LB} %f", lbm_input_velocity, Fx, Fy, Fz, phys_cd, lbm_cd);
////		log("Reynolds = %f lbmvel 0.07 physvel %f",0.07*ball_diameter/nse.lat.physDl/nse.lbmViscosity(), lbm_input_velocity);

// not used for evaluation of the results
////		real fil_fx=0,fil_fy=0,fil_fz=0;
//		Fx=Fy=Fz=0;
////		// FIXME - integrateForce is not implemented - see _stare_verze_/iblbm3d_verze1/filament_3D.h*
////		if (FIL_INDEX>=0) FF[FIL_INDEX].integrateForce(Fx,Fy,Fz, 1.0);//PI*ball_diameter*ball_diameter/(real)FF[FIL_INDEX].LL.size());
//		real lbm_cd_lagr=-Fx*8.0/lbm_input_velocity/lbm_input_velocity/PI/ball_diameter/ball_diameter*nse.lat.physDl*nse.lat.physDl;
//		real phys_cd_lagr=-nse.lbm2physForce(Fx)*dV*8.0/rho/target_velocity/target_velocity/PI/ball_diameter/ball_diameter;
//		if (std::isnan(Fx) || std::isnan(Fz) || std::isnan(Fz)) { if (!nse.terminate) log("nan detected"); nse.terminate=true; }
//		log("LAGR: u0 %e Fx %e Fy %e Fz %e C_D{phys} %e C_D{LB} %f", lbm_input_velocity, Fx, Fy, Fz, phys_cd_lagr, lbm_cd_lagr);


		// empty files
		char iotype[10];
		if (firstrun) sprintf(iotype,"wt"); else sprintf(iotype,"at");
		firstrun=false;
		// output
		FILE*f;
		char str[200], dir[200];
		char txt[200];
		real total = (real)(nse.lat.global.x()*nse.lat.global.y()*nse.lat.global.z()), ratio, area;
		sprintf(dir,"results_%s",id);
		mkdir(dir,0755);
		sprintf(dir,"results_%s/probes",id);
		mkdir(dir,0755);

		sprintf(str,"%s/probe_cd_full",dir);
		f = fopen(str,iotype);
		fprintf(f,"%e\t%e\n",nse.physTime(),lbm_cd_full);
		fclose(f);

//		sprintf(str,"%s/probe_cd",dir);
//		f = fopen(str,iotype);
//		fprintf(f,"%e\t%e\n",nse.physTime(),lbm_cd);
//		fclose(f);

//		sprintf(str,"%s/probe_cd_lagr",dir);
//		f = fopen(str,iotype);
//		fprintf(f,"%e\t%e\n",nse.physTime(),lbm_cd_lagr);
//		fclose(f);

//		sprintf(str,"%s/probe_cd_all",dir);
//		f = fopen(str,iotype);
//		fprintf(f,"%e\t%e\t%e\t%e\n",nse.physTime(),lbm_cd_full, lbm_cd, lbm_cd_lagr);
//		fclose(f);
	}

	virtual void updateKernelVelocities()
	{
		for (auto& block : nse.blocks)
		{
			block.data.inflow_vx = lbm_input_velocity;
//			block.data.inflow_vx = nse.phys2lbmVelocity(target_velocity);
		}
	}

	virtual void setupBoundaries()
	{
		nse.setBoundaryZ(0, BC::GEO_INFLOW);// top
		nse.setBoundaryZ(nse.lat.global.z()-1, BC::GEO_INFLOW);// bottom
		nse.setBoundaryY(0, BC::GEO_INFLOW); // back
		nse.setBoundaryY(nse.lat.global.y()-1, BC::GEO_INFLOW);// front
		nse.setBoundaryX(0, BC::GEO_INFLOW); // left
		nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_OUTFLOW_EQ);// right
	}

	StateLocal(const TNL::MPI::Comm& communicator, lat_t ilat, real iphysViscosity, real iphysDt)
		: State<LBM_TYPE>(communicator, ilat, iphysViscosity, iphysDt)
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
	using idx = typename STATE::TRAITS::idx;
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
	state.log("added %d lagrangian points",points);

	real sigma_min = state.FF[INDEX].computeMinDist();
	real sigma_max = state.FF[INDEX].computeMaxDistFromMinDist(sigma_min);

	state.log("Ball surface: wanted sigma %e (%f i.e. %d points), wanted_unit_area %e, sigma_min %e, sigma_max %e",sigma,count,N,wanted_unit_area,sigma_min, sigma_max);
//	state.log("Added %d Lagrangian points (requested %d) partial area %e",points, N, a);
//	state.log("Lagrange created: WuShuCompute %d ws_regularDirac %s",state.FF[INDEX].WuShuCompute,(state.FF[INDEX].ws_regularDirac)?"true":"false");
	state.log("h=physdl %e sigma min %e sigma_ku_h %e",state.nse.lat.physDl, sigma_min, sigma_min/state.nse.lat.physDl);
	state.log("h=physdl %e sigma max %e sigma_ku_h %e",state.nse.lat.physDl, sigma_max, sigma_max/state.nse.lat.physDl);

	state.writeVTK_Points("ball.vtk",0,0,state.FF[INDEX]);
	return INDEX;
}

template < typename LBM_TYPE >
int sim(int RES=2, double i_Re=1000, double nasobek=2.0, int dirac_delta=2, int method=0, int compute=5)
{
	using idx = typename LBM_TYPE::TRAITS::idx;
	using real = typename LBM_TYPE::TRAITS::real;
	using point_t = typename LBM_TYPE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size=32;
	real BALL_DIAMETER = 0.01;
	real real_domain_height= BALL_DIAMETER*11;// [m]
	real real_domain_length= BALL_DIAMETER*11;// [m] // extra 1cm on both sides
	idx LBM_Y = RES*block_size; // for 4 cm
	idx LBM_Z = LBM_Y;
	real PHYS_DL = real_domain_height/((real)LBM_Y);
	idx LBM_X = LBM_Y;//(int)(real_domain_length/PHYS_DL)+2;//block_size;//16*RESOLUTION;// width in pixels --- product of 128.
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// zvolit Re + LBM VELOCITY + PHYS_VISCOSITY
//	real i_Re = ;
	real i_LBM_VELOCITY = 0.07; // Geier
	real i_PHYS_VISCOSITY = 0.00001; // proc ne?
	// mam:
	real i_LBM_VISCOSITY = i_LBM_VELOCITY * BALL_DIAMETER / PHYS_DL / i_Re;
	real i_PHYS_VELOCITY = i_PHYS_VISCOSITY * i_Re / BALL_DIAMETER;
	printf("input phys velocity %f\ninput lbm velocity %f\nRe %f\nlbm viscosity %f\nphys viscosity %f\n", i_PHYS_VELOCITY, i_LBM_VELOCITY, i_Re, i_LBM_VISCOSITY, i_PHYS_VISCOSITY);

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

	StateLocal<LBM_TYPE> state(MPI_COMM_WORLD, lat, PHYS_VISCOSITY, PHYS_DT);
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

	state.setid("sim_4_%s_%s_dirac_%d_res_%d_Re_%d_nas_%05.4f_compute_%d", LBM_TYPE::COLL::id, (method>0)?"original":"modified", dirac_delta, RES, (int)Re, nasobek, compute);
	if (state.isMark()) return 0;

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
		default: state.log("Unknown parameter compute=%d, selecting default ws_computeGPU_TNL.", compute); ws_compute = ws_computeGPU_TNL; break;
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
	using NSE_TYPE = D3Q27<
				COLL,
				NSE_Data_ConstInflow< TRAITS >,
				D3Q27_BC_All,
				typename COLL::EQ,
				D3Q27_STREAMING< TRAITS >,
				MacroLocal< TRAITS >,
				D3Q27_MACRO_Void< TRAITS >,
				TRAITS
			>;

	sim<NSE_TYPE>(res, Re, h, dirac, method, compute);
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
			printf("error: required %d parameters required:\n %s method{0,1} dirac{1,2,3,4} Re{100,200} hi[0,%d] res[1,22] compute[1,7]\n", pars, argv[0],hmax-1);
			return 1;
		}
		int method = atoi(argv[1]);	// 0=modified 1=original
		int dirac = atoi(argv[2]);
		int Re = atoi(argv[3]);		// type=0,1,2 (geometry selection)
		int hi = atoi(argv[4]);		// index in the hvals
		int res = atoi(argv[5]);	// res=1,2,3
		int compute = atoi(argv[6]); // compute=1,2,3,4,5,6,7

		if (method > 1 || method < 0) { printf("error: method=%d out of bounds [0, 1]\n",method); return 1; }
		if (dirac < 1 || dirac > 4) { printf("error: dirac=%d out of bounds [1,4]\n",dirac); return 1; }
		if (hi >= hmax || hi < 0) { printf("error: hi=%d out of bounds [0, %d]\n",hi,hmax-1); return 1; }
		if (res < 1) { printf("error: res=%d out of bounds [1, ...]\n",res); return 1; }
		if (compute < 1 || compute > 7) { printf("error: compute=%d out of bounds [1,7]\n",compute); return 1; }
		if (hi<hmax) h=hvals[hi];
		run(res, (double)Re, h, dirac, method, compute);
	}
	return 0;
}
