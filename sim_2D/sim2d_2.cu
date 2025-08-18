#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/lbm_data.h"

#include "lbm3d/d2q9/bc.h"
#include "lbm3d/d2q9/col_srt.h"
#include "lbm3d/d2q9/col_clbm.h"
#include "lbm3d/d2q9/macro.h"

// exactly one streaming header must be included
#ifdef AA_PATTERN
	#include "lbm3d/d2q9/streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "lbm3d/d2q9/streaming_AB.h"
#endif

template <typename TRAITS>
struct NSE2D_Data_ConstInflow : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	dreal inflow_vx = 0;
	dreal inflow_vy = 0;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.vx = inflow_vx;
		KS.vy = inflow_vy;
	}
};

template <typename TRAITS>
struct D2Q9_MACRO_WithMean : D2Q9_MACRO_Base<TRAITS>
{
    using dreal = typename TRAITS::dreal;
    using idx   = typename TRAITS::idx;

    // Keep default fields and extend with running sums for means
    enum
    {
        e_rho,
        e_vx,
        e_vy,
        e_svx,   // sum of vx (LBM units), gated by accumulate_means
        e_svy,   // sum of vy (LBM units), gated by accumulate_means
        N
    };

    // Write per-cell macro values each step
    template <typename LBM_DATA, typename LBM_KS>
    __cuda_callable__ static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
    {
        SD.macro(e_rho, x, y, z) = KS.rho;
        SD.macro(e_vx , x, y, z) = KS.vx;
        SD.macro(e_vy , x, y, z) = KS.vy;

        // Accumulate only after warm-up (host toggles SD.accumulate_means)
        if (SD.accumulate_means) {
            SD.macro(e_svx, x, y, z) += KS.vx;
            SD.macro(e_svy, x, y, z) += KS.vy;
        }
    }

    // Provide KS with viscosity and body forces (2D)
    template <typename LBM_DATA, typename LBM_KS>
    __cuda_callable__ static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
    {
        KS.lbmViscosity = SD.lbmViscosity;
        KS.fx = SD.fx;   // ok even if zero
        KS.fy = SD.fy;
    }
};

template <typename TRAITS>
struct NSE2D_Data_ParabolicInflow : NSE_Data<TRAITS>
{
    using idx   = typename TRAITS::idx;
    using dreal = typename TRAITS::dreal;

    // Inputs (LBM units and indices), set each step from StateLocal::updateKernelVelocities()
    dreal u_max_lbm = 0;     // U_max in LBM units ( = 1.5 * U_mean_phys mapped to LBM )
    idx   y0 = 1;            // first interior fluid row
    idx   y1 = 1;            // last interior fluid row
    dreal inv_den = 1;       // 1.0 / (y1 - y0), precomputed
    bool  accumulate_means = false; // gate for mean accumulation in Macro

    template <typename LBM_KS>
    CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
    {
        // Normalize to s in [0,1] across interior fluid rows
        dreal s = (dreal)(y - y0) * inv_den;
        if (s < 0) s = 0; else if (s > 1) s = 1;
        KS.vx = u_max_lbm * (4.0 * s * (1.0 - s));
        KS.vy = 0;
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
	using State<NSE>::vtk_helper;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	real lbm_inflow_vx = 0;
	real u_max_lbm = 0;

	// Mean settings
    real stats_start_time = 2.5;  // [s] ignore early transients
    int  mean_samples = 0;        // number of steps accumulated (constant dt => time-weighted)

	void setupBoundaries() override
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW);							  // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_RIGHT);  // right

		nse.setBoundaryY(1, BC::GEO_WALL);						 // back
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);	 // front

		// extra layer needed due to A-A pattern
		nse.setBoundaryY(0, BC::GEO_NOTHING);						// back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);	// front

		// draw a wall with a hole
		int cx = floor(0.20 / nse.lat.physDl);
		int width = nse.lat.global.y() / 10;
		for (int px = cx; px <= cx + width; px++)
			for (int py = 1; py <= nse.lat.global.y() - 2; py++)
				if (! (py >= nse.lat.global.y() * 4 / 10 && py <= nse.lat.global.y() * 6 / 10)) {
					nse.setMap(px, py, 0, BC::GEO_WALL);
				}
	}

	bool outputData(const BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) override
	{
		int k = 0;
		if (index == k++)
			return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho, x, y, z), 1, desc, value, dofs);
		if (index == k++) {
			switch (dof) {
				case 0:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, z)), 3, desc, value, dofs);
				case 1:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, z)), 3, desc, value, dofs);
				case 2:
					// VTK does not know 2D vectors
					return vtk_helper("velocity", 0, 3, desc, value, dofs);
			}
		}
		if (index == k++) {
            // mean_vx (physical units) from sums and sample count
            real mean_vx_phys = 0;
            if (mean_samples > 0) {
                real s = block.hmacro(MACRO::e_svx, x, y, z) / (real)mean_samples;
                mean_vx_phys = nse.lat.lbm2physVelocity(s);
            }
            return vtk_helper("mean_vx", mean_vx_phys, 1, desc, value, dofs);
        }
        if (index == k++) {
            // mean_vy (physical units)
            real mean_vy_phys = 0;
            if (mean_samples > 0) {
                real s = block.hmacro(MACRO::e_svy, x, y, z) / (real)mean_samples;
                mean_vy_phys = nse.lat.lbm2physVelocity(s);
            }
            return vtk_helper("mean_vy", mean_vy_phys, 1, desc, value, dofs);
        }

		return false;
	}

	// Push inflow + mean-accumulation gate to device data each step
    void updateKernelVelocities() override
    {
        const bool accumulate_now = (nse.physTime() >= stats_start_time);
        if (accumulate_now) mean_samples++;

        for (auto& block : nse.blocks) {
            // Inflow profile params
            block.data.u_max_lbm = u_max_lbm;
            block.data.y0 = 1;
            block.data.y1 = nse.lat.global.y() - 2;
            const int denom = std::max(1, (int)(block.data.y1 - block.data.y0));
            block.data.inv_den = 1.0 / (real)denom;

            // Toggle mean accumulation in the Macro
            block.data.accumulate_means = accumulate_now;
        }
    }

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat)
	: State<NSE>(id, communicator, std::move(lat))
	{}
};

template <typename NSE>
int sim(int RESOLUTION = 2)
{
    using idx = typename NSE::TRAITS::idx;
    using real = typename NSE::TRAITS::real;
    using point_t = typename NSE::TRAITS::point_t;
    using lat_t = Lattice<3, real, idx>;

    int block_size = 32;
    int X = 128 * RESOLUTION;                 // width in pixels
    int Y = block_size * RESOLUTION;          // height in pixels

    // Schäfer–Turek 2D-2 inspired setup:
    real LBM_VISCOSITY  = 1.0e-3;               // tau = 0.5 + 3*nu = 0.56 (stable)
    real PHYS_HEIGHT    = 0.41;               // [m]
    real PHYS_VISCOSITY = 1.0e-3;             // [m^2/s] (benchmark fluid)  Re ≈ 100 when U_mean = 1.0 m/s
    real PHYS_VELOCITY  = 1.0;      // [m/s] mean inflow; U_max = 1.5 * mean

    real PHYS_DL = PHYS_HEIGHT / ((real)Y - 2);
    real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
    point_t PHYS_ORIGIN = {0., 0., 0.};

    // initialize the lattice
    lat_t lat;
    lat.global        = typename lat_t::CoordinatesType(X, Y, 1);
    lat.physOrigin    = PHYS_ORIGIN;
    lat.physDl        = PHYS_DL;
    lat.physDt        = PHYS_DT;
    lat.physViscosity = PHYS_VISCOSITY;

    const std::string state_id =
        fmt::format("sim2d_2_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
    StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

    if (!state.canCompute())
        return 0;

	// Set U_max (LBM units) for the parabolic profile
    real U_max_phys = 1.5 * PHYS_VELOCITY;
    state.u_max_lbm = lat.phys2lbmVelocity(U_max_phys);

    state.nse.physFinalTime = 6.0;
    state.cnt[PRINT].period = 0.05;

    // 2D = cut in 3D at z=0
    state.cnt[VTK2D].period = 0.05;
    state.add2Dcut_Z(0, "");

    execute(state);
    return 0;
}

template <typename TRAITS = TraitsDP>
void run(int RES)
{
	//using COLL = D2Q9_SRT<TRAITS>;
	using COLL = D2Q9_CLBM<TRAITS>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D2Q9_KernelStruct,
		// NSE2D_Data_ConstInflow<TRAITS>,
		NSE2D_Data_ParabolicInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D2Q9_STREAMING<TRAITS>,
		D2Q9_BC_All,
		// D2Q9_MACRO_Default<TRAITS>>;
		D2Q9_MACRO_WithMean<TRAITS>>;

	sim<NSE_CONFIG>(RES);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim2d_1");
	program.add_description("Simple incompressible Navier-Stokes simulation example.");
	program.add_argument("resolution").help("resolution of the lattice").scan<'i', int>().default_value(1);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << '\n';
		std::cerr << program;
		return 1;
	}

	const auto resolution = program.get<int>("resolution");
	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}

	run(resolution);

	return 0;
}
