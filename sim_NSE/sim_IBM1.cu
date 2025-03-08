#include <argparse/argparse.hpp>

#include "lbm3d/core.h"
#include "lbm3d/lagrange_3D.h"
#include "lbm3d/obstacles_ibm.h"

// cylinder in 3D - Schafer-Turek problem
// IBM-LBM

template <typename TRAITS>
struct MacroLocal : D3Q27_MACRO_Base<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum
	{
		e_fx,
		e_fy,
		e_fz,
		e_vx,
		e_vy,
		e_vz,
		e_rho,
		N
	};

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z) = KS.vx;
		SD.macro(e_vy, x, y, z) = KS.vy;
		SD.macro(e_vz, x, y, z) = KS.vz;
	}

	template <typename LBM_DATA>
	CUDA_HOSTDEV static void zeroForces(LBM_DATA& SD, idx x, idx y, idx z)
	{
		SD.macro(e_fx, x, y, z) = 0;
		SD.macro(e_fy, x, y, z) = 0;
		SD.macro(e_fz, x, y, z) = 0;
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.macro(e_fx, x, y, z);
		KS.fy = SD.macro(e_fy, x, y, z);
		KS.fz = SD.macro(e_fz, x, y, z);
	}
};

template <typename TRAITS>
struct NSE_Data_SpecialInflow : NSE_Data<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	dreal inflow_vx = 0;
	dreal inflow_vy = 0;
	dreal inflow_vz = 0;
	//dreal inflow_z0=0.00112;
	//dreal inflow_mez=0.0112;
	//dreal inflow_physDl=0.1;
	dreal physDl;
	dreal H;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.vx = 16.0 * inflow_vx
			  * MAX(0, (physDl * (y - 0.5) / H) * (physDl * (z - 0.5) / H) * (1.0 - physDl * (y - 0.5) / H) * (1.0 - physDl * (z - 0.5) / H));
		//(no1 - ((physDl*y-y0)*(physDl*y-y0) + (physDl*z-z0)*(physDl*z-z0))/delta/delta);
		//KS.vx = inflow_vx*(no1 - ((physDl*y-y0)*(physDl*y-y0) + (physDl*z-z0)*(physDl*z-z0))/delta/delta);
		KS.vy = 0;
		KS.vz = 0;
	}
};

template <typename NSE>
struct StateLocal : State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK<NSE>;

	using State<NSE>::nse;
	using State<NSE>::ibm;
	using State<NSE>::vtk_helper;
	using State<NSE>::id;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	//dreal lbm_input_velocity=0.07;
	//dreal start_velocity;
	dreal phys_input_U_max;
	dreal phys_input_U_bar;
	//real init_time;
	bool firstrun = true;
	//bool firstplot=true;
	real cylinder_diameter = 0.01;
	point_t cylinder_c;

	virtual bool outputData(const BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs)
	{
		int k = 0;
		if (index == k++) {
			switch (dof) {
				case 0:
					return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vx, x, y, z), 3, desc, value, dofs);
				case 1:
					return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vy, x, y, z), 3, desc, value, dofs);
				case 2:
					return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vz, x, y, z), 3, desc, value, dofs);
			}
		}
		if (index == k++) {
			switch (dof) {
				case 0:
					return vtk_helper("lbm_force", block.hmacro(MACRO::e_fx, x, y, z), 3, desc, value, dofs);
				case 1:
					return vtk_helper("lbm_force", block.hmacro(MACRO::e_fy, x, y, z), 3, desc, value, dofs);
				case 2:
					return vtk_helper("lbm_force", block.hmacro(MACRO::e_fz, x, y, z), 3, desc, value, dofs);
			}
		}
		if (index == k++)
			return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho, x, y, z), 1, desc, value, dofs);
		if (index == k++)
			return vtk_helper("lbm_density_fluctuation", block.hmacro(MACRO::e_rho, x, y, z) - 1.0, 1, desc, value, dofs);
		if (index == k++) {
			switch (dof) {
				case 0:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, z)), 3, desc, value, dofs);
				case 1:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, z)), 3, desc, value, dofs);
				case 2:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vz, x, y, z)), 3, desc, value, dofs);
			}
		}
		if (index == k++) {
			switch (dof) {
				case 0:
					return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fx, x, y, z)), 3, desc, value, dofs);
				case 1:
					return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fy, x, y, z)), 3, desc, value, dofs);
				case 2:
					return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fz, x, y, z)), 3, desc, value, dofs);
			}
		}
		//if (index==k++) return vtk_helper("density", block.hmacro(MACRO::e_rho,x,y,z)*nse.physFluidDensity, 1, desc, value, dofs);
		return false;
	}

	virtual void probe1()
	{
		// compute drag
		real Fx = 0, Fy = 0, Fz = 0, dV = nse.lat.physDl * nse.lat.physDl * nse.lat.physDl;
		real rho = 1.0;	 //nse.physFluidDensity;
		//real target_velocity = nse.lat.lbm2physVelocity(lbm_input_velocity);
		real lbm_input_velocity = nse.lat.phys2lbmVelocity(phys_input_U_bar);
		spdlog::info(
			"Reynolds = {:f} lbmvel {:f} physvel {:f} (phys_input_U_bar {:f})",
			lbm_input_velocity * cylinder_diameter / nse.lat.physDl / nse.lat.lbmViscosity(),
			lbm_input_velocity,
			nse.lat.lbm2physVelocity(lbm_input_velocity),
			phys_input_U_bar
		);

		// FIXME: MPI !!!
		// todo: compute C_D: integrate over the whole domain
		for (int x = 0; x < nse.lat.global.x(); x++)
			for (int y = 0; y < nse.lat.global.y(); y++)
				for (int z = 0; z < nse.lat.global.z(); z++) {
					// test if outside the ball
					//if (NORM(x*nse.lat.physDl - cylinder_c[0], y*nse.lat.physDl - cylinder_c[1], z*nse.lat.physDl - cylinder_c[2]) > 2.0*cylinder_diameter/2.0)
					//if (NORM(x*nse.lat.physDl - cylinder_c[0], y*nse.lat.physDl - cylinder_c[1], z*nse.lat.physDl -
					//cylinder_c[2]) > cylinder_diameter/2.0)
					{
						Fx += nse.blocks.front().hmacro(MACRO::e_fx, x, y, z);
						Fy += nse.blocks.front().hmacro(MACRO::e_fy, x, y, z);
						Fz += nse.blocks.front().hmacro(MACRO::e_fz, x, y, z);
					}
				}

		real lbm_cd_full =
			-Fx * 2.0 / lbm_input_velocity / lbm_input_velocity / cylinder_diameter / nse.blocks.front().data.H * nse.lat.physDl * nse.lat.physDl;
		real phys_cd_full =
			-nse.lat.lbm2physForce(Fx) * dV * 2.0 / rho / phys_input_U_bar / phys_input_U_bar / cylinder_diameter / nse.blocks.front().data.H;
		real lbm_cl_full =
			-Fz * 2.0 / lbm_input_velocity / lbm_input_velocity / cylinder_diameter / nse.blocks.front().data.H * nse.lat.physDl * nse.lat.physDl;
		real phys_cl_full =
			-nse.lat.lbm2physForce(Fz) * dV * 2.0 / rho / phys_input_U_bar / phys_input_U_bar / cylinder_diameter / nse.blocks.front().data.H;
		spdlog::info(
			"FULL: u0 {:e} Fx {:e} Fy {:e} Fz {:e} C_D{{phys}} {:e} C_D{{LB}} {:f} C_L{{phys}} {:e} C_L{{LB}} {:f}",
			lbm_input_velocity,
			Fx,
			Fy,
			Fz,
			phys_cd_full,
			lbm_cd_full,
			phys_cl_full,
			lbm_cl_full
		);

		// not used for evaluation of the results
		// FIXME: MPI !!!
		//for (int x=0; x<nse.lat.global.x(); x++)
		//for (int y=0; y<nse.lat.global.y(); y++)
		//for (int z=0; z<nse.lat.global.z(); z++)
		//{
		//	// test if outside the ball
		//	//if (NORM(x*nse.lat.physDl - cylinder_c[0], y*nse.lat.physDl - cylinder_c[1], z*nse.lat.physDl - cylinder_c[2])
		//< 2.0*cylinder_diameter/2.0) 	if (NORM(x*nse.lat.physDl - cylinder_c[0], 0, z*nse.lat.physDl - cylinder_c[2]) > cylinder_diameter/2.0)
		//	{
		//		Fx += nse.hmacro(MACRO::e_fx,x,y,z);
		//		Fy += nse.hmacro(MACRO::e_fy,x,y,z);
		//		Fz += nse.hmacro(MACRO::e_fz,x,y,z);
		//	}
		//}
		//real lbm_cd=-Fx*2.0/lbm_input_velocity/lbm_input_velocity/cylinder_diameter/nse.blocks.front().data.H*nse.lat.physDl*nse.lat.physDl;
		//real phys_cd=-nse.lat.lbm2physForce(Fx)*dV*2.0/rho/phys_input_U_bar/phys_input_U_bar/cylinder_diameter/nse.blocks.front().data.H;
		//real lbm_cl=-Fz*2.0/lbm_input_velocity/lbm_input_velocity/cylinder_diameter/nse.blocks.front().data.H*nse.lat.physDl*nse.lat.physDl;
		//real phys_cl=-nse.lat.lbm2physForce(Fz)*dV*2.0/rho/phys_input_U_bar/phys_input_U_bar/cylinder_diameter/nse.blocks.front().data.H;
		//spdlog::info("INNN: u0 {:e} Fx {:e} Fy {:e} Fz {:e} C_D{{phys}} {:e} C_D{{LB}} {:f}", lbm_input_velocity, Fx, Fy, Fz, phys_cd, lbm_cd);
		////spdlog::info("Reynolds = {:f} lbmvel 0.07 physvel {:f}",0.07*cylinder_diameter/nse.lat.physDl/nse.lbmViscosity(), lbm_input_velocity);

		// not used for evaluation of the results
		////real fil_fx=0,fil_fy=0,fil_fz=0;
		//Fx=Fy=Fz=0;
		//		// FIXME - integrateForce is not implemented - see _stare_verze_/iblbm3d_verze1/filament_3D.h*
		////if (FIL_INDEX>=0) ibm.integrateForce(Fx,Fy,Fz, 1.0);//PI*cylinder_diameter*cylinder_diameter/(real)ibm.LL.size());
		//real lbm_cd_lagr=-Fx*2.0/lbm_input_velocity/lbm_input_velocity/cylinder_diameter/nse.blocks.front().data.H*nse.lat.physDl*nse.lat.physDl;
		//real phys_cd_lagr=-nse.lat.lbm2physForce(Fx)*dV*2.0/rho/phys_input_U_bar/phys_input_U_bar/cylinder_diameter/nse.blocks.front().data.H;
		//real lbm_cl_lagr=-Fz*2.0/lbm_input_velocity/lbm_input_velocity/cylinder_diameter/nse.blocks.front().data.H*nse.lat.physDl*nse.lat.physDl;
		//real phys_cl_lagr=-nse.lat.lbm2physForce(Fz)*dV*2.0/rho/phys_input_U_bar/phys_input_U_bar/cylinder_diameter/nse.blocks.front().data.H;
		//spdlog::info("LAGR: u0 {:e} Fx {:e} Fy {:e} Fz {:e} C_D{{phys}} {:e} C_D{{LB}} {:f}", lbm_input_velocity, Fx, Fy, Fz, phys_cd_lagr,
		//lbm_cd_lagr);

		// empty files
		const char* iotype = (firstrun) ? "wt" : "at";
		firstrun = false;
		// output
		FILE* f;
		//real total = (real)(nse.lat.global.x()*nse.lat.global.y()*nse.lat.global.z()), ratio, area;
		const std::string dir = fmt::format("results_{}/probes", id);
		mkdir_p(dir.c_str(), 0755);

		std::string str = fmt::format("{}/probe_cd_full", dir);
		f = fopen(str.c_str(), iotype);
		fprintf(f, "%e\t%e\n", nse.physTime(), lbm_cd_full);
		fclose(f);

		//str = fmt::format("{}/probe_cd", dir);
		//f = fopen(str.c_str(), iotype);
		//fprintf(f, "%e\t%e\n", nse.physTime(), lbm_cd);
		//fclose(f);

		//str = fmt::format("{}/probe_cd_lagr", dir);
		//f = fopen(str.c_str(), iotype);
		//fprintf(f, "%e\t%e\n", nse.physTime(), lbm_cd_lagr);
		//fclose(f);

		//str = fmt::format("{}/probe_cd_all", dir);
		//f = fopen(str.c_str(), iotype);
		//fprintf(f, "%e\t%e\t%e\t%e\n", nse.physTime(), lbm_cd_full, lbm_cd, lbm_cd_lagr);
		//fclose(f);

		str = fmt::format("{}/probe_cl_full", dir);
		f = fopen(str.c_str(), iotype);
		fprintf(f, "%e\t%e\n", nse.physTime(), lbm_cl_full);
		fclose(f);

		//str = fmt::format("{}/probe_cl", dir);
		//f = fopen(str.c_str(), iotype);
		//fprintf(f, "%e\t%e\n", nse.physTime(), lbm_cl);
		//fclose(f);

		//str = fmt::format("{}/probe_cl_lagr", dir);
		//f = fopen(str.c_str(), iotype);
		//fprintf(f, "%e\t%e\n", nse.physTime(), lbm_cl_lagr);
		//fclose(f);

		//str = fmt::format("{}/probe_cl_all", dir);
		//f = fopen(str.c_str(), iotype);
		//fprintf(f, "%e\t%e\t%e\t%e\n", nse.physTime(), lbm_cl_full, lbm_cl, lbm_cl_lagr);
		//fclose(f);
	}

	virtual void updateKernelVelocities()
	{
		for (auto& block : nse.blocks) {
			block.data.inflow_vx = nse.lat.phys2lbmVelocity(phys_input_U_max);
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
			block.data.physDl = nse.lat.physDl;
			block.data.H = (nse.lat.global.y() - 2.0) * nse.lat.physDl;	 // domain width and height
		}
	}

	virtual void setupBoundaries()
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW_LEFT);					   // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_EQ);  // right
		nse.setBoundaryY(0, BC::GEO_WALL);							   // back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_WALL);		   // front
		nse.setBoundaryZ(0, BC::GEO_WALL);							   // top
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_WALL);		   // bottom
	}

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat)
	: State<NSE>(id, communicator, lat)
	{}
};

template <typename NSE>
int sim(int RES, double Re, double discretization_ratio, const std::string& compute, int dirac, const std::string& method)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	real cylinder_diameter = 0.10;	 // [m]
	real real_domain_height = 0.41;	 // [m]
	real real_domain_length = 2.00;	 // [m]
	idx LBM_Y = RES * block_size;	 // for 4 cm
	idx LBM_Z = LBM_Y;
	real PHYS_DL = real_domain_height / ((real) LBM_Y - 2.0);
	idx LBM_X = (int) (real_domain_length / PHYS_DL) + 2;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	real PHYS_VISCOSITY = 0.001;  // [m^2/s]
	//real Umax = 0.45; // [m/s]
	real Ubar = Re * PHYS_VISCOSITY / cylinder_diameter;
	real Umax = 9.0 / 4.0 * Ubar;  // [m/s] // Re=20 --> 0.45m/s

	real LBM_VISCOSITY = 0.001;

	//fmt::print("input phys velocity {:f}\ninput lbm velocity {:f}\nRe {:f}\nlbm viscosity {:f}\nphys viscosity {:f}\n", i_PHYS_VELOCITY,
	//	i_LBM_VELOCITY, i_Re, i_LBM_VISCOSITY, i_PHYS_VISCOSITY);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(LBM_X, LBM_Y, LBM_Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format(
		"sim_IBM1_{}_{}_dirac_{}_res_{}_Re_{}_nas_{:05.4f}_compute_{}", NSE::COLL::id, method, dirac, RES, Re, discretization_ratio, compute
	);
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

	if (! state.canCompute())
		return 0;

	state.phys_input_U_max = Umax;
	state.phys_input_U_bar = Ubar;
	state.nse.physCharLength = cylinder_diameter;  // [m]
	state.cylinder_diameter = cylinder_diameter;   // [m]
	//state.nse.physFluidDensity = 1000.0; // [kg/m^3]

	state.cnt[PRINT].period = 0.1;
	state.cnt[PROBE1].period = 0.1;
	state.nse.physFinalTime = 10.0;

	//state.cnt[VTK3D].period = 1.0;
	state.cnt[VTK2D].period = 1.0;
	state.cnt[VTK1D].period = 1.0;

	// select compute method
	IbmCompute computeVariant;
	if (compute == "GPU")
		computeVariant = IbmCompute::GPU;
	else if (compute == "CPU")
		computeVariant = IbmCompute::CPU;
	else if (compute == "hybrid")
		computeVariant = IbmCompute::Hybrid;
	else if (compute == "hybrid-zero-copy")
		computeVariant = IbmCompute::Hybrid_zerocopy;
	else {
		spdlog::warn("Unknown parameter compute={}, selecting GPU as the default.", compute);
		computeVariant = IbmCompute::GPU;
	}

	// add cuts
	state.add2Dcut_X(LBM_X / 2, "cut_X");
	//state.add2Dcut_X(2*cylinder_diameter/PHYS_DL,"cut_Xcylinder");
	state.add2Dcut_Y(LBM_Y / 2, "cut_Y");
	state.add2Dcut_Z(LBM_Z / 2, "cut_Z");

	// create immersed objects
	state.cylinder_c[0] = 0.50;	 // [m]
	state.cylinder_c[1] = 0;	 // [m]
	state.cylinder_c[2] = 0.20;	 // [m]
	real sigma = discretization_ratio * PHYS_DL;
	ibmSetupCylinder(state.ibm, state.cylinder_c, state.cylinder_diameter, sigma);
	state.writeVTK_Points("cylinder", 0, 0);

	// configure IBM
	state.ibm.computeVariant = computeVariant;
	state.ibm.diracDeltaTypeEL = dirac;
	if (method == "modified")
		state.ibm.methodVariant = IbmMethod::modified;
	else if (method == "original")
		state.ibm.methodVariant = IbmMethod::original;
	else {
		spdlog::warn("Unknown parameter method={}, selecting modified as the default.", method);
		state.ibm.methodVariant = IbmMethod::modified;
	}

	execute(state);

	return 0;
}

template <typename TRAITS = TraitsSP>
void run(int res, double Re, double discretization_ratio, const std::string& compute, int dirac, const std::string& method)
{
	using COLL = D3Q27_CUM<TRAITS>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_SpecialInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		MacroLocal<TRAITS>>;

	sim<NSE_CONFIG>(res, Re, discretization_ratio, compute, dirac, method);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_IBM1");
	program.add_description("IBM-LBM simulation with cylinder in 3D - Schafer-Turek problem.");
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--Re").help("desired Reynolds number (affects the inflow velocity)").scan<'g', double>().default_value(100.0).nargs(1);
	program.add_argument("--discretization-ratio")
		.help("ratio between the Lagrangian spacing step and the Eulerian spacing step")
		.scan<'g', double>()
		.default_value(0.25)
		.nargs(1);
	program.add_argument("--compute").help("IBM compute method").default_value("GPU").choices("GPU", "CPU", "hybrid", "hybrid-zero-copy").nargs(1);
	program.add_argument("--dirac").help("Dirac delta function to use in IBM").scan<'i', int>().default_value(1).choices(1, 2, 3, 4).nargs(1);
	program.add_argument("--method").help("IBM method").default_value("modified").choices("modified", "original").nargs(1);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		return 1;
	}

	const auto resolution = program.get<int>("--resolution");
	const auto Re = program.get<double>("--Re");
	const auto discretization_ratio = program.get<double>("--discretization-ratio");
	const auto compute = program.get<std::string>("--compute");
	const auto dirac = program.get<int>("--dirac");
	const auto method = program.get<std::string>("--method");

	if (resolution < 1)
		throw std::invalid_argument("CLI error: resolution must be at least 1");
	if (Re < 1)
		throw std::invalid_argument("CLI error: Re must be at least 1");
	if (discretization_ratio <= 0)
		throw std::invalid_argument("CLI error: discretization-ratio must be positive");

	run(resolution, Re, discretization_ratio, compute, dirac, method);

	return 0;
}
