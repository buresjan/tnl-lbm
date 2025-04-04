#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/lagrange_3D.h"
#include "lbm3d/obstacles_ibm.h"

// ball in 3D
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

	dreal lbm_inflow_vx = 0;
	bool firstrun = true;
	real ball_diameter = 0.01;
	point_t ball_c;

	bool outputData(const BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) override
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

	void probe1() override
	{
		// compute drag
		real Fx = 0;
		real Fy = 0;
		real Fz = 0;
		real dV = nse.lat.physDl * nse.lat.physDl * nse.lat.physDl;
		real rho = 1.0;	 //nse.physFluidDensity;
		real target_velocity = nse.lat.lbm2physVelocity(lbm_inflow_vx);
		spdlog::info(
			"Reynolds = {:f} lbmvel {:f} physvel {:f}",
			lbm_inflow_vx * ball_diameter / nse.lat.physDl / nse.lat.lbmViscosity(),
			lbm_inflow_vx,
			nse.lat.lbm2physVelocity(lbm_inflow_vx)
		);

		// FIXME: MPI !!!
		// todo: compute C_D: integrate over the whole domain
		for (int x = 0; x < nse.lat.global.x(); x++)
			for (int y = 0; y < nse.lat.global.y(); y++)
				for (int z = 0; z < nse.lat.global.z(); z++) {
					// test if outside the ball
					//if (NORM(x*nse.lat.physDl - ball_c[0], y*nse.lat.physDl - ball_c[1], z*nse.lat.physDl - ball_c[2]) > 2.0*ball_diameter/2.0)
					//if (NORM(x*nse.lat.physDl - ball_c[0], y*nse.lat.physDl - ball_c[1], z*nse.lat.physDl - ball_c[2]) > ball_diameter/2.0)
					{
						Fx += nse.blocks.front().hmacro(MACRO::e_fx, x, y, z);
						Fy += nse.blocks.front().hmacro(MACRO::e_fy, x, y, z);
						Fz += nse.blocks.front().hmacro(MACRO::e_fz, x, y, z);
					}
				}

		real lbm_cd_full = -Fx * 8.0 / lbm_inflow_vx / lbm_inflow_vx / PI / ball_diameter / ball_diameter * nse.lat.physDl * nse.lat.physDl;
		real phys_cd_full = -nse.lat.lbm2physForce(Fx) * dV * 8.0 / rho / target_velocity / target_velocity / PI / ball_diameter / ball_diameter;
		spdlog::info("FULL: u0 {:e} Fx {:e} Fy {:e} Fz {:e} C_D{{phys}} {:e} C_D{{LB}} {:f}", lbm_inflow_vx, Fx, Fy, Fz, phys_cd_full, lbm_cd_full);

		// not used for evaluation of the results
		// FIXME: MPI !!!
		//for (int x=0; x<nse.lat.global.x(); x++)
		//for (int y=0; y<nse.lat.global.y(); y++)
		//for (int z=0; z<nse.lat.global.z(); z++)
		//{
		//	// test if outside the ball
		//	//if (NORM(x*nse.lat.physDl - ball_c[0], y*nse.lat.physDl - ball_c[1], z*nse.lat.physDl - ball_c[2]) < 2.0*ball_diameter/2.0)
		//	if (NORM(x*nse.lat.physDl - ball_c[0], y*nse.lat.physDl - ball_c[1], z*nse.lat.physDl - ball_c[2]) > ball_diameter/2.0)
		//	{
		//		Fx += nse.blocks.front().hmacro(MACRO::e_fx,x,y,z);
		//		Fy += nse.blocks.front().hmacro(MACRO::e_fy,x,y,z);
		//		Fz += nse.blocks.front().hmacro(MACRO::e_fz,x,y,z);
		//	}
		//}
		//real lbm_cd=-Fx*8.0/lbm_input_velocity/lbm_input_velocity/PI/ball_diameter/ball_diameter*nse.lat.physDl*nse.lat.physDl;
		//real phys_cd=-nse.lat.lbm2physForce(Fx)*dV*8.0/rho/target_velocity/target_velocity/PI/ball_diameter/ball_diameter;
		//spdlog::info("INNN: u0 {:e} Fx {:e} Fy {:e} Fz {:e} C_D{{phys}} {:e} C_D{{LB}} {:f}", lbm_input_velocity, Fx, Fy, Fz, phys_cd, lbm_cd);
		////spdlog::info("Reynolds = {:f} lbmvel 0.07 physvel {:f}", 0.07*ball_diameter/nse.lat.physDl/nse.lbmViscosity(), lbm_input_velocity);

		// not used for evaluation of the results
		////real fil_fx=0,fil_fy=0,fil_fz=0;
		//Fx=Fy=Fz=0;
		////		// FIXME - integrateForce is not implemented - see _stare_verze_/iblbm3d_verze1/filament_3D.h*
		////if (FIL_INDEX>=0) ibm.integrateForce(Fx,Fy,Fz, 1.0);//PI*ball_diameter*ball_diameter/(real)ibm.LL.size());
		//real lbm_cd_lagr=-Fx*8.0/lbm_input_velocity/lbm_input_velocity/PI/ball_diameter/ball_diameter*nse.lat.physDl*nse.lat.physDl;
		//real phys_cd_lagr=-nse.lat.lbm2physForce(Fx)*dV*8.0/rho/target_velocity/target_velocity/PI/ball_diameter/ball_diameter;
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
	}

	void updateKernelVelocities() override
	{
		for (auto& block : nse.blocks) {
			block.data.inflow_vx = lbm_inflow_vx;
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}

	void setupBoundaries() override
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW_LEFT);					   // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_EQ);  // right
		nse.setBoundaryY(0, BC::GEO_INFLOW);						   // back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_INFLOW);	   // front
		nse.setBoundaryZ(0, BC::GEO_INFLOW);						   // top
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_INFLOW);	   // bottom
	}

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat)
	: State<NSE>(id, communicator, std::move(lat))
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
	real BALL_DIAMETER = 0.01;
	real real_domain_height = BALL_DIAMETER * 11;  // [m]
	//real real_domain_length= BALL_DIAMETER*11;// [m] // extra 1cm on both sides
	idx LBM_Y = RES * block_size;  // for 4 cm
	idx LBM_Z = LBM_Y;
	real PHYS_DL = real_domain_height / ((real) LBM_Y);
	idx LBM_X = LBM_Y;	//(int)(real_domain_length/PHYS_DL)+2;//block_size;//16*RESOLUTION;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// zvolit Re + LBM VELOCITY + PHYS_VISCOSITY
	real i_LBM_VELOCITY = 0.07;		  // Geier
	real i_PHYS_VISCOSITY = 0.00001;  // proc ne?
	// mam:
	real i_LBM_VISCOSITY = i_LBM_VELOCITY * BALL_DIAMETER / PHYS_DL / Re;
	real i_PHYS_VELOCITY = i_PHYS_VISCOSITY * Re / BALL_DIAMETER;
	fmt::print(
		"input phys velocity {:f}\ninput lbm velocity {:f}\nRe {:f}\nlbm viscosity{:f}\nphys viscosity {:f}\n",
		i_PHYS_VELOCITY,
		i_LBM_VELOCITY,
		Re,
		i_LBM_VISCOSITY,
		i_PHYS_VISCOSITY
	);

	real LBM_VISCOSITY = i_LBM_VISCOSITY;	 //0.0001*RES;//*SIT;//1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_VISCOSITY = i_PHYS_VISCOSITY;	 //0.00001;// [m^2/s] fluid viscosity of water

	//real INIT_TIME = 1.0; // [s]
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(LBM_X, LBM_Y, LBM_Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format(
		"sim_IBM2_{}_{}_dirac_{}_res_{}_Re_{}_nas_{:05.4f}_compute_{}", NSE::COLL::id, method, dirac, RES, Re, discretization_ratio, compute
	);
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

	if (! state.canCompute())
		return 0;

	state.lbm_inflow_vx = i_LBM_VELOCITY;
	state.nse.physCharLength = BALL_DIAMETER;  // [m]
	state.ball_diameter = BALL_DIAMETER;	   // [m]
	//state.nse.physFluidDensity = 1000.0; // [kg/m^3]

	state.cnt[PRINT].period = 0.1;
	state.cnt[PROBE1].period = 0.1;
	state.nse.physFinalTime = 30.0;

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
	//state.add2Dcut_X(2*BALL_DIAMETER/PHYS_DL,"cut_Xball");
	state.add2Dcut_Y(LBM_Y / 2, "cut_Y");
	state.add2Dcut_Z(LBM_Z / 2, "cut_Z");

	// create immersed objects
	state.ball_c[0] = 2 * state.ball_diameter;
	state.ball_c[1] = 5.5 * state.ball_diameter;
	state.ball_c[2] = 5.5 * state.ball_diameter;
	real sigma = discretization_ratio * PHYS_DL;
	ibmDrawSphere(state.ibm, state.ball_c, state.ball_diameter / 2.0, sigma);

	// 2nd ball
	state.ball_c[0] = 5.5 * state.ball_diameter;
	ibmDrawSphere(state.ibm, state.ball_c, state.ball_diameter / 2.0, sigma);

	state.writeVTK_Points("ball", 0, 0);

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
		NSE_Data_ConstInflow<TRAITS>,
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

	argparse::ArgumentParser program("sim_IBM2");
	program.add_description("IBM-LBM simulation with two spheres in 3D.");
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
		std::cerr << err.what() << '\n';
		std::cerr << program;
		return 1;
	}

	const auto resolution = program.get<int>("--resolution");
	const auto Re = program.get<double>("--Re");
	const auto discretization_ratio = program.get<double>("--discretization-ratio");
	const auto compute = program.get<std::string>("--compute");
	const auto dirac = program.get<int>("--dirac");
	const auto method = program.get<std::string>("--method");

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}
	if (Re < 1) {
		fmt::println(stderr, "CLI error: Re must be at least 1");
		return 1;
	}
	if (discretization_ratio <= 0) {
		fmt::println(stderr, "CLI error: discretization-ratio must be positive");
		return 1;
	}

	run(resolution, Re, discretization_ratio, compute, dirac, method);

	return 0;
}
