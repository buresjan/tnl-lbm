#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/lbm_data.h"

#include "lbm3d/d2q9/bc.h"
#include "lbm3d/d2q9/col_srt.h"
#include "lbm3d/d2q9/col_clbm.h"
#include "lbm3d/d2q9/macro.h"

#include <algorithm>
#include <cmath>     // std::sqrt, std::abs
#include <fstream>
#include <sstream>
#include <filesystem>
#include "lbm_common/fileutils.h"  // create_parent_directories

namespace {
// Toggle whether geometry type 1 cells should be treated as Bouzidi near-wall fluid by default.
constexpr bool kDefaultUseBouzidiForType1 = true;
bool gUseBouzidiForType1 = kDefaultUseBouzidiForType1;
}

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
    // Macro channels (LBM units unless noted). Channels are sized like the distributions (Q x X x Y x Z).
    //
    // e_svx, e_svy: running sums of velocity components in LBM units. Host keeps mean_samples and
    //                divides to obtain running means when needed. Accumulation gated by accumulate_means.
    // e_mean_vx_frozen, e_mean_vy_frozen: per-cell frozen means in LBM units copied to device before
    //                                      fluctuation accumulation starts.
    // e_smag_uprime: running sum of |u - <u>| in LBM units; average by dividing with fluc_samples;
    //                accumulation gated by accumulate_flucs.
    // e_suprime2_sum / e_svprime2_sum: running sums of squared fluctuation components in LBM units.
    enum
    {
        e_rho,
        e_vx,
        e_vy,
        e_svx,   // sum of vx (LBM units), gated by accumulate_means
        e_svy,   // sum of vy (LBM units), gated by accumulate_means
        e_mean_vx_frozen,   // per-cell frozen mean vx (LBM)
        e_mean_vy_frozen,   // per-cell frozen mean vy (LBM)
        e_smag_uprime,      // running sum of |u - <u>| (LBM); filled after freeze
        e_suprime2_sum,     // running sum of (u'_^2) (LBM^2)
        e_svprime2_sum,     // running sum of (v'_^2) (LBM^2)
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


        // After the mean is frozen, accumulate |u - <u>| (LBM units)
        if (SD.accumulate_flucs) {
            const dreal mvx = SD.macro(e_mean_vx_frozen, x, y, z);
            const dreal mvy = SD.macro(e_mean_vy_frozen, x, y, z);
            const dreal dux = KS.vx - mvx;
            const dreal duy = KS.vy - mvy;
            const dreal mag = sqrt(dux * dux + duy * duy);
            SD.macro(e_smag_uprime, x, y, z) += mag;
            SD.macro(e_suprime2_sum, x, y, z) += dux * dux;
            SD.macro(e_svprime2_sum, x, y, z) += duy * duy;
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
    // u_max_lbm defines a Poiseuille-like inflow profile. y0..y1 define interior fluid rows used for
    // normalization to s in [0,1].
    // accumulate_means / accumulate_flucs are device-side gates to toggle accumulation phases.
    // use_bouzidi toggles Bouzidi interpolation at near-wall nodes.
    dreal u_max_lbm = 0;     // U_max in LBM units ( = 1.5 * U_mean_phys mapped to LBM )
    idx   y0 = 1;            // first interior fluid row
    idx   y1 = 1;            // last interior fluid row
    dreal inv_den = 1;       // 1.0 / (y1 - y0), precomputed
    bool  accumulate_means = false; // gate for mean accumulation in Macro
    bool  accumulate_flucs = false;   // gate for |u - <u>| accumulation (after mean is frozen)
    bool  use_bouzidi      = true;    // toggle for near-wall Bouzidi interpolation

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
    bool enable_bouzidi = true;

    // Object file with per-voxel type and Bouzidi coefficients
    std::string object_filename;

    // Mean settings — host-side control of accumulation, stabilization and freeze of <u>
    real stats_start_time = 1.5;  // [s] ignore early transients
    real stats_end_time   = 3.5;  // [s] fallback window end for mean
    int  mean_samples = 0;        // number of steps accumulated (constant dt => time-weighted)

    // --- Mean convergence control (host-side) ---
    real mean_tol               = 1.0e-3;   // [m/s] |Δ(domain-avg |<u>|)| threshold between checks
    real mean_check_period      = 0.05;     // [s]   cadence to evaluate stabilization
    int  mean_stable_required   = 10;        // consecutive passes needed to declare convergence

    bool means_frozen           = false;    // latched once the mean is declared stable
    real mean_freeze_time       = -1;       // [s]   time when mean was frozen

    // rolling bookkeeping for the check
    real next_mean_check_time   = stats_start_time + mean_check_period;
    real prev_domain_mean_speed = -1;       // previous domain-averaged |<u>| [m/s]
    
    // Fluc settings — host-side gate and stabilization for fluctuation RMS speed
    int  fluc_samples             = 0;        // time-weighted sample count for fluctuation statistics

    real fluc_tol                 = 1.0e-3;   // [m/s] |Δ(domain RMS(|u'|))| threshold between checks
    real fluc_check_period        = 0.05;     // [s]
    int  fluc_stable_required     = 10;        // consecutive passes

    bool flucs_frozen             = false;    // latched once RMS(|u'|) declared stable
    real fluc_freeze_time         = -1;       // [s]

    real next_fluc_check_time     = -1;       // [s]
    real prev_domain_fluc_rms     = -1;       // previous domain RMS(|u'|) [m/s]
    
    // Export control
    bool tke_value_written        = false;    // guard to write TKE once



    void setupBoundaries() override
    {
        // Load geometry/object map and Bouzidi coefficients before BCs
        // The file describes per-voxel x y type and Bouzidi thetas (-1 means "no Bouzidi").
        // Types are mapped to GEO_FLUID / GEO_FLUID_NEAR_WALL / GEO_WALL.
        if (!object_filename.empty()) {
            projectObjectFromFile(object_filename);
        }
		nse.setBoundaryX(0, BC::GEO_INFLOW);							  // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_RIGHT);  // right

		nse.setBoundaryY(1, BC::GEO_WALL);						 // back
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);	 // front

		// extra layer needed due to A-A pattern
		nse.setBoundaryY(0, BC::GEO_NOTHING);						// back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);	// front

		// Physical parameters: center (0.2, 0.2), diameter 0.1 m
		real cx_phys = 0.20;           // [m] x-position of cylinder center
		real cy_phys = 0.20;           // [m] y-position of cylinder center
		real radius_phys = 0.05;       // [m] radius = 0.1 / 2

		// // convert to lattice indices
		// int cx = static_cast<int>(cx_phys / nse.lat.physDl + 0.5);
		// int cy = static_cast<int>(cy_phys / nse.lat.physDl + 0.5);
		// int radius = static_cast<int>(radius_phys / nse.lat.physDl + 0.5);

		// // loop over domain and mark all lattice sites inside circle as wall
		// for (int px = 1; px < nse.lat.global.x() - 1; px++) {
		// 	for (int py = 1; py < nse.lat.global.y() - 1; py++) {
		// 		int dx = px - cx;
		// 		int dy = py - cy;
		// 		if (dx*dx + dy*dy <= radius*radius) {
		// 			nse.setMap(px, py, 0, BC::GEO_WALL);
		// 		}
		// 	}
		// }
	}

    // Allocate arrays and load per-voxel map and Bouzidi coefficients
    //
    // File format: one row per lattice cell (x y type theta_E theta_N theta_W theta_S theta_NE theta_NW theta_SW theta_SE)
    // Coeff order is in the cardinal/diagonal order noted above. We later use opposite-direction theta
    // when applying Bouzidi in the BC. Theta -1 means "no Bouzidi"; theta in (0,1] defines sub-grid wall location.
    void projectObjectFromFile(const std::string& fileArg)
    {
        using std::getline;
        namespace fs = std::filesystem;

        // Determine path: if contains path separator use as-is; otherwise prepend sim_2D/ellipses/
        fs::path path = fileArg;
        if (!fileArg.empty() && fileArg.find('/') == std::string::npos && fileArg.find('\\') == std::string::npos) {
            path = fs::path("sim_2D/ellipses") / fileArg;
        }

        // Allocate Bouzidi arrays across blocks
        nse.allocateBouzidiCoeffArrays();

        const auto X = nse.lat.global.x();
        const auto Y = nse.lat.global.y();
        spdlog::info("Loading geometry '{}' (resolved path '{}'), expected domain {} x {}", fileArg, path.string(), X, Y);

        std::ifstream fin(path);
        if (!fin) {
            spdlog::error("Failed to open object file: {}", path.string());
            throw std::runtime_error("Cannot open object file");
        }

        long long count = 0;
        long long out_of_range = 0;
        long long parse_errors = 0;
        long long coeff_sets = 0;
        long long type_sets = 0;
        long long near_boundary_nearwall = 0;
        typename TRAITS::idx max_x = -1, max_y = -1;

        std::string line;
        long long line_no = 0;
        long long last_processed_line = 0;
        long long last_x = -1;
        long long last_y = -1;
        int last_type = -1;
        double last_coeffs[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

        try {
            while (getline(fin, line)) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                long long xi, yi;
                int cell_type;
                double c[8];
                if (!(iss >> xi >> yi >> cell_type >> c[0] >> c[1] >> c[2] >> c[3] >> c[4] >> c[5] >> c[6] >> c[7])) {
                    parse_errors++;
                    continue;
                }
                line_no++;
                last_processed_line = line_no;
                last_x = xi;
                last_y = yi;
                last_type = cell_type;
                for (int d = 0; d < 8; ++d) {
                    last_coeffs[d] = c[d];
                }

                // Validate theta range: allow -1 (sentinel) or [0,1]. If any > 1, abort.
                for (int d = 0; d < 8; ++d) {
                    if (c[d] > 1.0) {
                        spdlog::error("Invalid Bouzidi theta > 1 at line {} (x={}, y={}, dir={}, theta={}) in {}",
                                       (long long)line_no, xi, yi, d, c[d], path.string());
                        throw std::runtime_error("Bouzidi theta out of range (>1)");
                    }
                }
                count++;
                if (xi >= 0 && yi >= 0) {
                    if (xi > max_x) max_x = (typename TRAITS::idx) xi;
                    if (yi > max_y) max_y = (typename TRAITS::idx) yi;
                }
                if (xi < 0 || yi < 0 || xi >= X || yi >= Y) {
                    out_of_range++;
                    continue;
                }

                // Set map according to type
                typename BC::map_t mapval = BC::GEO_FLUID;
                switch (cell_type) {
                    case 0: mapval = BC::GEO_FLUID; break;
                    case 1:
                        mapval = gUseBouzidiForType1 ? BC::GEO_FLUID_NEAR_WALL : BC::GEO_FLUID;
                        break;
                    case 2: mapval = BC::GEO_WALL; break;
                    default: mapval = BC::GEO_FLUID; break;
                }
                nse.setMap((idx)xi, (idx)yi, 0, mapval);
                type_sets++;
                if (mapval == BC::GEO_FLUID_NEAR_WALL) {
                    if (xi <= 0 || xi >= X - 1 || yi <= 0 || yi >= Y - 1)
                        near_boundary_nearwall++;
                }

                // Store Bouzidi coefficients (always store; -1 used as sentinel as provided)
                static int bouzidi_assign_warnings = 0;
                bool assigned = false;
                for (auto& block : nse.blocks) {
                    if (!block.isLocalIndex((idx)xi, (idx)yi, 0)) continue;
                    if (block.hBouzidi.getData() == nullptr && bouzidi_assign_warnings < 5) {
                        spdlog::error(
                            "Bouzidi host array not allocated for block with offset=({}, {}, {})",
                            block.offset.x(),
                            block.offset.y(),
                            block.offset.z()
                        );
                        bouzidi_assign_warnings++;
                    }
                    for (int d = 0; d < 8; ++d) {
                        // Directions order per user spec mapped to indices 0..7
                        block.hBouzidi(d, (idx)xi, (idx)yi, 0) = (typename TRAITS::dreal) c[d];
                    }
                    coeff_sets++;
                    assigned = true;
                    break;
                }
                if (!assigned && bouzidi_assign_warnings < 5) {
                    spdlog::error(
                        "Bouzidi coefficients for global cell ({},{},{}) did not match any block (blocks={} entries).",
                        xi,
                        yi,
                        0,
                        nse.blocks.size()
                    );
                    bouzidi_assign_warnings++;
                }
            }
        } catch (const std::exception& err) {
            spdlog::error(
                "Exception while loading geometry '{}': line={} x={} y={} type={} coeffs=[{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}] message={}",
                path.string(),
                last_processed_line,
                last_x,
                last_y,
                last_type,
                last_coeffs[0],
                last_coeffs[1],
                last_coeffs[2],
                last_coeffs[3],
                last_coeffs[4],
                last_coeffs[5],
                last_coeffs[6],
                last_coeffs[7],
                err.what()
            );
            throw;
        }

        // Basic dimension checks — ensure the file matches XxY domain
        const auto infer_X = max_x + 1;
        const auto infer_Y = max_y + 1;
        bool dims_ok = (infer_X == X && infer_Y == Y);
        bool count_ok = (count == (long long)X * (long long)Y);
        spdlog::info(
            "Geometry load stats: rows={}, inferred dims={} x {}, type sets={}, coeff sets={}, parse errors={}, out-of-range={}, geometry='{}'",
            count,
            (long long)infer_X,
            (long long)infer_Y,
            type_sets,
            coeff_sets,
            parse_errors,
            out_of_range,
            path.string()
        );
        if (!dims_ok || !count_ok) {
            spdlog::error("Object grid mismatch or incomplete: file dim=({},{}) inferred from max coords, count={} vs expected {}x{}={}.",
                          (long long)infer_X, (long long)infer_Y, count, (long long)X, (long long)Y, (long long)X * (long long)Y);
            throw std::runtime_error("Object file dimensions do not match simulation lattice");
        }

        if (parse_errors > 0 || out_of_range > 0 || near_boundary_nearwall > 0) {
            spdlog::warn("While loading object: parse_errors={}, out_of_range={}, near_boundary_nearwall={}",
                         parse_errors, out_of_range, near_boundary_nearwall);
        }

        for (auto& block : nse.blocks) {
            if (block.dBouzidi.getData() != nullptr) {
                block.dBouzidi = block.hBouzidi;
            }
        }
    }

    // Exposed fields in 2D VTK/probes, queried by index in a deterministic order.
    //
    // 0: lbm_density                (LBM units)
    // 1: velocity (vector, 2D embedded in 3) (physical units)
    // 2: mean_vx, 3: mean_vy        (physical units; mean over accumulated window)
    // 4: mean_fluc_mag              (<|u'|>, physical units)
    // 5: mean_vel_mag               (|<u>|, physical units; frozen mean if available)
    // 6..13: bouzidi_* thetas       (raw theta per direction; -1 if "none")
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

        if (index == k++) {
            // mean(|u'|) in physical units
            real mean_fluc_mag_phys = 0;
            if (fluc_samples > 0) {
                const real s = block.hmacro(MACRO::e_smag_uprime, x, y, z) / (real)fluc_samples; // LBM
                mean_fluc_mag_phys = nse.lat.lbm2physVelocity(s);
            }
            return vtk_helper("mean_fluc_mag", mean_fluc_mag_phys, 1, desc, value, dofs);
        }

        if (index == k++) {
            // mean velocity magnitude (physical units)
            real mean_vel_mag_phys = 0;
            if (means_frozen) {
                const real mvx_lbm = block.hmacro(MACRO::e_mean_vx_frozen, x, y, z);
                const real mvy_lbm = block.hmacro(MACRO::e_mean_vy_frozen, x, y, z);
                const real mag_lbm = sqrt(mvx_lbm * mvx_lbm + mvy_lbm * mvy_lbm);
                mean_vel_mag_phys = nse.lat.lbm2physVelocity(mag_lbm);
            } else if (mean_samples > 0) {
                const real mvx_lbm = block.hmacro(MACRO::e_svx, x, y, z) / (real)mean_samples;
                const real mvy_lbm = block.hmacro(MACRO::e_svy, x, y, z) / (real)mean_samples;
                const real mag_lbm = sqrt(mvx_lbm * mvx_lbm + mvy_lbm * mvy_lbm);
                mean_vel_mag_phys = nse.lat.lbm2physVelocity(mag_lbm);
            }
            return vtk_helper("mean_vel_mag", mean_vel_mag_phys, 1, desc, value, dofs);
        }

        if (index == k++) {
            real velocity_xx = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, z)) * nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, z));
            real velocity_yy = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, z)) * nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, z));
            return vtk_helper("velocity_magnitude", std::sqrt(velocity_xx + velocity_yy), 1, desc, value, dofs);
        }

        // Bouzidi coefficients (theta per direction, -1 if none); guard against missing allocation
        const bool have_bouzidi = (block.hBouzidi.getData() != nullptr);
        if (index == k++) {
            real v = (real)-1;
            if (have_bouzidi) v = (real) block.hBouzidi(0, x, y, z);
            return vtk_helper("bouzidi_east", v, 1, desc, value, dofs);
        }
        if (index == k++) {
            real v = (real)-1;
            if (have_bouzidi) v = (real) block.hBouzidi(1, x, y, z);
            return vtk_helper("bouzidi_north", v, 1, desc, value, dofs);
        }
        if (index == k++) {
            real v = (real)-1;
            if (have_bouzidi) v = (real) block.hBouzidi(2, x, y, z);
            return vtk_helper("bouzidi_west", v, 1, desc, value, dofs);
        }
        if (index == k++) {
            real v = (real)-1;
            if (have_bouzidi) v = (real) block.hBouzidi(3, x, y, z);
            return vtk_helper("bouzidi_south", v, 1, desc, value, dofs);
        }
        if (index == k++) {
            real v = (real)-1;
            if (have_bouzidi) v = (real) block.hBouzidi(4, x, y, z);
            return vtk_helper("bouzidi_ne", v, 1, desc, value, dofs);
        }
        if (index == k++) {
            real v = (real)-1;
            if (have_bouzidi) v = (real) block.hBouzidi(5, x, y, z);
            return vtk_helper("bouzidi_nw", v, 1, desc, value, dofs);
        }
        if (index == k++) {
            real v = (real)-1;
            if (have_bouzidi) v = (real) block.hBouzidi(6, x, y, z);
            return vtk_helper("bouzidi_sw", v, 1, desc, value, dofs);
        }
        if (index == k++) {
            real v = (real)-1;
            if (have_bouzidi) v = (real) block.hBouzidi(7, x, y, z);
            return vtk_helper("bouzidi_se", v, 1, desc, value, dofs);
        }

        
			return false;
		}

	// Push inflow + mean-accumulation gate to device data each step
    void updateKernelVelocities() override
    {
        const real t = nse.physTime();

        // 1) Mean accumulation gating (after warm-up, until frozen)
        //    Limit accumulation to [stats_start_time, stats_end_time) so a fallback
        //    mean over that window can be forced if stabilization does not occur.
        const bool do_acc_means = (!means_frozen) && (t >= stats_start_time) && (t < stats_end_time);
        if (do_acc_means) mean_samples++;  // time-weighted since dt is constant

        // 2) Push inflow profile params + gates to device data
        for (auto& block : nse.blocks) {
            // inflow profile
            block.data.u_max_lbm = u_max_lbm;
            block.data.y0 = 1;
            block.data.y1 = nse.lat.global.y() - 2;
            const int denom = std::max(1, (int)(block.data.y1 - block.data.y0));
            block.data.inv_den = 1.0 / (real)denom;
            block.data.use_bouzidi = enable_bouzidi;

            // gates for device-side accumulation
            block.data.accumulate_means = do_acc_means;
            // fluctuations gate handled in section 3 (set below as we freeze means)
        }

        // 3) Host-side stabilization check for the mean — freezes <u> upon stabilization
        if (!means_frozen) {
            checkAndMaybeFreezeMeans();

            // If we just froze here, we also need to snapshot frozen mean & switch to flucts
            if (means_frozen) {
                snapshotFrozenMeansToMacro();    // (Section 2B)
                // start fluctuation accumulation next step; samples reset there
            }
        }

        // 3B) Fallback: if stabilization did not occur by stats_end_time,
        //      force-freeze the mean computed over [stats_start_time, stats_end_time].
        if (!means_frozen && t >= stats_end_time) {
            means_frozen     = true;
            mean_freeze_time = stats_end_time;  // anchor to window end
            snapshotFrozenMeansToMacro();
        }

        // If mean already frozen, maintain fluctuation accumulation and stabilization
        if (means_frozen) {
            // samples advance only while not frozen and gate is on
            const bool gate_on = (!flucs_frozen);
            if (gate_on) fluc_samples++;

            // keep the device gate in sync
            for (auto& block : nse.blocks) {
                block.data.accumulate_flucs = gate_on;
            }

            // stabilization check for RMS(|u'|) (Section 4)
            if (!flucs_frozen) checkAndMaybeFreezeFlucMag();

            // If fluctuations just became frozen (or are already), export TKE once and terminate
            if (flucs_frozen && !tke_value_written) {
                exportThirdQuarterTKE_andTerminate();
            }
        }

    }

    // Also push Bouzidi coeff pointers once arrays are allocated
    void updateKernelData() override
    {
        State<NSE>::updateKernelData();
        for (auto& block : this->nse.blocks) {
            if (block.dBouzidi.getData() != nullptr)
                block.data.bouzidi_coeff_ptr = block.dBouzidi.getData();
            else
                block.data.bouzidi_coeff_ptr = nullptr;
        }
    }

    // Freeze the running mean <u> into MACRO channels and start the fluctuation phase.
    //
    // Steps:
    // 0) Pull latest macro from device to host (to get current sums)
    // 1) Stop mean accumulation on device
    // 2) Compute and write frozen mean to host MACRO; reset e_smag_uprime to 0
    // 3) Push host MACRO to device so kernels see the frozen mean
    // 4) Reset fluc_samples and enable accumulate_flucs
    void snapshotFrozenMeansToMacro()
    {
        // 0) Pull the latest running sums from device so we base the frozen mean on up-to-date data
        nse.copyMacroToHost();

        // 1) Stop the mean accumulation at the device immediately
        for (auto& block : nse.blocks) {
            block.data.accumulate_means = false;
        }

        // 2) Snapshot per-cell frozen mean (in LBM units) into MACRO channels
        const idx Nx = nse.lat.global.x();
        const idx Ny = nse.lat.global.y();

        for (auto& block : nse.blocks) {
            for (idx x = 1; x < Nx - 1; ++x) {
                for (idx y = 1; y < Ny - 1; ++y) {
                    const real mvx_lbm = (mean_samples > 0)
                        ? block.hmacro(MACRO::e_svx, x, y, 0) / (real)mean_samples
                        : (real)0;
                    const real mvy_lbm = (mean_samples > 0)
                        ? block.hmacro(MACRO::e_svy, x, y, 0) / (real)mean_samples
                        : (real)0;

                    block.hmacro(MACRO::e_mean_vx_frozen, x, y, 0) = mvx_lbm;
                    block.hmacro(MACRO::e_mean_vy_frozen, x, y, 0) = mvy_lbm;

                    // also reset fluctuation sum so its average starts clean
                    block.hmacro(MACRO::e_smag_uprime,    x, y, 0) = 0;
                    block.hmacro(MACRO::e_suprime2_sum,   x, y, 0) = 0;
                    block.hmacro(MACRO::e_svprime2_sum,   x, y, 0) = 0;
                }
            }
        }

        // 3) Push frozen means and reset accumulators to device so the kernel sees correct <u>
        nse.copyMacroToDevice();

        // 4) Initialize fluctuation accumulation (Section 3)
        fluc_samples        = 0;
        flucs_frozen        = false;
        prev_domain_fluc_rms = (real)-1;
        next_fluc_check_time = mean_freeze_time + fluc_check_period;

        for (auto& block : nse.blocks) {
            block.data.accumulate_flucs = true;   // device starts accumulating |u - <u>| next step
        }
    }

    // Disable 3D VTK for this 2D simulation (avoid 3D output even on NaN)
    void writeVTKs_3D() override {}

    bool isFluidLocal(const BLOCK& block, idx x, idx y) const
    {
        return BC::isFluid(block.hmap(x, y, 0));
    }

    // Domain-averaged |<u>| over the full interior (physical units)
    real computeDomainAvgMeanSpeed_phys() const
    {
        if (mean_samples <= 0) return 0;

        const idx Nx = nse.lat.global.x();
        const idx Ny = nse.lat.global.y();

        double sum_speed = 0.0;
        int count  = 0;

        for (const auto& block : nse.blocks) {
            const idx x_begin = std::max<idx>(1, block.offset.x());
            const idx x_end   = std::min<idx>(Nx - 1, block.offset.x() + block.local.x());
            const idx y_begin = std::max<idx>(1, block.offset.y());
            const idx y_end   = std::min<idx>(Ny - 1, block.offset.y() + block.local.y());

            if (x_begin >= x_end || y_begin >= y_end) continue;

            for (idx x = x_begin; x < x_end; ++x) {
                for (idx y = y_begin; y < y_end; ++y) {
                    if (!block.isLocalIndex(x, y, 0)) continue;
                    if (!isFluidLocal(block, x, y)) continue;
                    // running mean in LBM units
                    real mvx_lbm = block.hmacro(MACRO::e_svx, x, y, 0) / (real)mean_samples;
                    real mvy_lbm = block.hmacro(MACRO::e_svy, x, y, 0) / (real)mean_samples;

                    // convert to physical [m/s]
                    const real mvx = nse.lat.lbm2physVelocity(mvx_lbm);
                    const real mvy = nse.lat.lbm2physVelocity(mvy_lbm);

                    sum_speed += std::sqrt((double)mvx * mvx + (double)mvy * mvy);
                    ++count;
                }
            }
        }

        if (count == 0) return 0;
        return (real)(sum_speed / (double)count);
    }

    // Domain RMS fluctuation speed over the full interior (physical units)
    real computeDomainRMSFlucSpeed_phys() const
    {
        if (fluc_samples <= 0) return 0;

        const idx Nx = nse.lat.global.x();
        const idx Ny = nse.lat.global.y();

        double sum_sq_lbm = 0.0;
        long long count = 0;

        for (const auto& block : nse.blocks) {
            const idx x_begin = std::max<idx>(1, block.offset.x());
            const idx x_end   = std::min<idx>(Nx - 1, block.offset.x() + block.local.x());
            const idx y_begin = std::max<idx>(1, block.offset.y());
            const idx y_end   = std::min<idx>(Ny - 1, block.offset.y() + block.local.y());

            if (x_begin >= x_end || y_begin >= y_end) continue;

            for (idx x = x_begin; x < x_end; ++x) {
                for (idx y = y_begin; y < y_end; ++y) {
                    if (!block.isLocalIndex(x, y, 0)) continue;
                    if (!isFluidLocal(block, x, y)) continue;
                    const real up2_lbm = block.hmacro(MACRO::e_suprime2_sum, x, y, 0) / (real)fluc_samples;
                    const real vp2_lbm = block.hmacro(MACRO::e_svprime2_sum, x, y, 0) / (real)fluc_samples;
                    sum_sq_lbm += (double)(up2_lbm + vp2_lbm);
                    ++count;
                }
            }
        }

        if (count == 0) return 0;
        const double avg_sq_lbm = sum_sq_lbm / (double)count;
        const real vel_scale = nse.lat.lbm2physVelocity((real)1);
        const double avg_sq_phys = avg_sq_lbm * (double)vel_scale * (double)vel_scale;
        return (real)std::sqrt(std::max(0.0, avg_sq_phys));
    }

    // Turbulent kinetic energy per cell [m^2/s^2],
    // computed from the mean fluctuation magnitude after fluctuations are frozen.
    // Returns 0 until flucs_frozen is true.
    real computeCellTKE_phys(const BLOCK& block, idx x, idx y) const
    {
        // Provide a meaningful value even if fluctuations have not stabilized yet.
        // Use whatever samples are available; if none, return 0.
        if (fluc_samples <= 0) return (real)0;

        const real avg_up2_lbm = block.hmacro(MACRO::e_suprime2_sum, x, y, 0) / (real) fluc_samples;
        const real avg_vp2_lbm = block.hmacro(MACRO::e_svprime2_sum, x, y, 0) / (real) fluc_samples;

        // Convert squared fluctuations to physical units. lbm2physVelocity is linear, so
        // scaling by the velocity conversion factor squared is sufficient.
        const real vel_scale = nse.lat.lbm2physVelocity((real)1);
        const real avg_up2_phys = avg_up2_lbm * vel_scale * vel_scale;
        const real avg_vp2_phys = avg_vp2_lbm * vel_scale * vel_scale;

        return (real)0.5 * (avg_up2_phys + avg_vp2_phys);
    }

    // Integrate TKE over the third quarter of the domain in X
    // (i.e., x in [2/4*X, 3/4*X)), and in Y exclude 3 layers at the
    // bottom and top (y in [3, Ny-3)).
    //
    // Fallbacks:
    //  A) If we have fluctuation samples: integrate TKE from accumulated squared fluctuations.
    //  B1) Else, if running means exist: approximate TKE from last-step deviations vs running mean.
    //  B2) Else, estimate the mean inside the region from the instantaneous field and use that.
    real integrateTKE_ThirdQuarter_phys() const
    {
        const idx Nx = nse.lat.global.x();
        const idx Ny = nse.lat.global.y();

        idx x0 = (2 * Nx) / 4;   // inclusive
        idx x1 = (3 * Nx) / 4;   // exclusive
        // Clamp to interior excluding the outermost layer like other loops
        if (x0 < 1) x0 = 1;
        if (x1 > Nx - 1) x1 = Nx - 1;

        idx y0 = 3;             // inclusive
        idx y1 = Ny - 3;        // exclusive
        if (y0 < 1) y0 = 1;
        if (y1 > Ny - 1) y1 = Ny - 1;

        if (x0 >= x1 || y0 >= y1) return (real)0;

        // Path A: accumulated fluctuation squares are available -> compute TKE directly
        if (fluc_samples > 0) {
            double sum_tke = 0.0; // [m^2/s^2]
            for (const auto& block : nse.blocks) {
                const idx bx0 = std::max<idx>(x0, block.offset.x());
                const idx bx1 = std::min<idx>(x1, block.offset.x() + block.local.x());
                const idx by0 = std::max<idx>(y0, block.offset.y());
                const idx by1 = std::min<idx>(y1, block.offset.y() + block.local.y());

                if (bx0 >= bx1 || by0 >= by1) continue;

                for (idx x = bx0; x < bx1; ++x) {
                    for (idx y = by0; y < by1; ++y) {
                        if (!block.isLocalIndex(x, y, 0)) continue;
                        if (!isFluidLocal(block, x, y)) continue;
                        sum_tke += (double)computeCellTKE_phys(block, x, y);
                    }
                }
            }
            const double cell_area = (double)nse.lat.physDl * (double)nse.lat.physDl; // [m^2]
            return (real)(sum_tke * cell_area); // [m^4/s^2]
        }

        // Path B: fallback if fluctuations never started accumulating
        // B1: If we have a running mean (mean_samples>0), estimate TKE from last-step deviations
        if (mean_samples > 0) {
            double sum_tke = 0.0;
            for (const auto& block : nse.blocks) {
                const idx bx0 = std::max<idx>(x0, block.offset.x());
                const idx bx1 = std::min<idx>(x1, block.offset.x() + block.local.x());
                const idx by0 = std::max<idx>(y0, block.offset.y());
                const idx by1 = std::min<idx>(y1, block.offset.y() + block.local.y());

                if (bx0 >= bx1 || by0 >= by1) continue;

                for (idx x = bx0; x < bx1; ++x) {
                    for (idx y = by0; y < by1; ++y) {
                        if (!block.isLocalIndex(x, y, 0)) continue;
                        if (!isFluidLocal(block, x, y)) continue;
                        const real vx_lbm = block.hmacro(MACRO::e_vx, x, y, 0);
                        const real vy_lbm = block.hmacro(MACRO::e_vy, x, y, 0);
                        const real mvx_lbm = block.hmacro(MACRO::e_svx, x, y, 0) / (real)mean_samples;
                        const real mvy_lbm = block.hmacro(MACRO::e_svy, x, y, 0) / (real)mean_samples;
                        const real dux_phys = nse.lat.lbm2physVelocity(vx_lbm - mvx_lbm);
                        const real duy_phys = nse.lat.lbm2physVelocity(vy_lbm - mvy_lbm);
                        const double mag2 = (double)dux_phys * (double)dux_phys + (double)duy_phys * (double)duy_phys;
                        sum_tke += 0.5 * mag2;
                    }
                }
            }
            const double cell_area = (double)nse.lat.physDl * (double)nse.lat.physDl;
            return (real)(sum_tke * cell_area);
        }

        // B2: As a last resort, estimate mean from instantaneous field over the region
        {
            double sx = 0.0, sy = 0.0; long long cnt = 0;
            for (const auto& block : nse.blocks) {
                const idx bx0 = std::max<idx>(x0, block.offset.x());
                const idx bx1 = std::min<idx>(x1, block.offset.x() + block.local.x());
                const idx by0 = std::max<idx>(y0, block.offset.y());
                const idx by1 = std::min<idx>(y1, block.offset.y() + block.local.y());

                if (bx0 >= bx1 || by0 >= by1) continue;

                for (idx x = bx0; x < bx1; ++x) {
                    for (idx y = by0; y < by1; ++y) {
                        if (!block.isLocalIndex(x, y, 0)) continue;
                        if (!isFluidLocal(block, x, y)) continue;
                        sx += (double)block.hmacro(MACRO::e_vx, x, y, 0);
                        sy += (double)block.hmacro(MACRO::e_vy, x, y, 0);
                        ++cnt;
                    }
                }
            }
            if (cnt == 0) return (real)0;
            const real mvx_lbm = (real)(sx / (double)cnt);
            const real mvy_lbm = (real)(sy / (double)cnt);

            double sum_tke = 0.0;
            for (const auto& block : nse.blocks) {
                const idx bx0 = std::max<idx>(x0, block.offset.x());
                const idx bx1 = std::min<idx>(x1, block.offset.x() + block.local.x());
                const idx by0 = std::max<idx>(y0, block.offset.y());
                const idx by1 = std::min<idx>(y1, block.offset.y() + block.local.y());

                if (bx0 >= bx1 || by0 >= by1) continue;

                for (idx x = bx0; x < bx1; ++x) {
                    for (idx y = by0; y < by1; ++y) {
                        if (!block.isLocalIndex(x, y, 0)) continue;
                        if (!isFluidLocal(block, x, y)) continue;
                        const real dux_phys = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, 0) - mvx_lbm);
                        const real duy_phys = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, 0) - mvy_lbm);
                        const double mag2 = (double)dux_phys * (double)dux_phys + (double)duy_phys * (double)duy_phys;
                        sum_tke += 0.5 * mag2;
                    }
                }
            }
            const double cell_area = (double)nse.lat.physDl * (double)nse.lat.physDl;
            return (real)(sum_tke * cell_area);
        }
    }

    // Compute and export the integral (as above) and request termination.
    // The output file name is suffixed with the object file basename.
    void exportThirdQuarterTKE_andTerminate()
    {
        if (tke_value_written) return;

        // Compute local integral (no MPI aggregation used)
        const double value = (double)integrateTKE_ThirdQuarter_phys();

        const std::string outpath = fmt::format("sim_2D/values/value_{}", std::filesystem::path(object_filename).filename().string());
        create_parent_directories(outpath.c_str());
        FILE* fp = fopen(outpath.c_str(), "wt");
        if (fp) {
            // print with high precision on one line
            fprintf(fp, "%.17g\n", value);
            fclose(fp);
        }

        tke_value_written = true;
        // request graceful termination of the simulation loop
        this->nse.terminate = true;
    }

    void AfterSimFinished() override
    {
        // If we didn’t export mid-run (e.g., flucs never stabilized),
        // export now with whatever value is available (likely zero).
        if (!tke_value_written) {
            const double value = (double)integrateTKE_ThirdQuarter_phys();
            const std::string outpath = fmt::format("sim_2D/values/value_{}", std::filesystem::path(object_filename).filename().string());
            create_parent_directories(outpath.c_str());
            FILE* fp = fopen(outpath.c_str(), "wt");
            if (fp) {
                fprintf(fp, "%.17g\n", value);
                fclose(fp);
            }
            tke_value_written = true;
        }
        // call base finalization (logging)
        State<NSE>::AfterSimFinished();
    }


    void checkAndMaybeFreezeMeans()
    {
        const real t = nse.physTime();
        if (t < stats_start_time || means_frozen) return;
        if (t + (real)1e-12 < next_mean_check_time) return; // not yet time to check

        nse.copyMacroToHost();

        const real curr = computeDomainAvgMeanSpeed_phys();

        static int stable_hits = 0;
        if (prev_domain_mean_speed < (real)0) {
            // first sample
            stable_hits = 0;
        } else {
            const real delta = std::abs(curr - prev_domain_mean_speed);
            stable_hits = (delta <= mean_tol) ? (stable_hits + 1) : 0;
        }

        prev_domain_mean_speed = curr;
        next_mean_check_time  += mean_check_period;

        // reached required consecutive stable checks -> freeze
        if (stable_hits >= mean_stable_required) {
            means_frozen     = true;
            std::cout << means_frozen << std::endl;
            mean_freeze_time = t;
        }
    }

    void checkAndMaybeFreezeFlucMag()
    {
        if (flucs_frozen || fluc_samples <= 0) return;

        const real t = nse.physTime();
        if (t + (real)1e-12 < next_fluc_check_time) return;

        nse.copyMacroToHost();

        const real curr = computeDomainRMSFlucSpeed_phys();

        static int stable_hits = 0;
        if (prev_domain_fluc_rms < (real)0) {
            stable_hits = 0; // first sample
        } else {
            const real delta = std::abs(curr - prev_domain_fluc_rms);
            stable_hits = (delta <= fluc_tol) ? (stable_hits + 1) : 0;
        }

        prev_domain_fluc_rms  = curr;
        next_fluc_check_time += fluc_check_period;

        if (stable_hits >= fluc_stable_required) {
            flucs_frozen  = true;
            std::cout << flucs_frozen << std::endl;
            fluc_freeze_time = t;

            // stop device-side accumulation
            for (auto& block : nse.blocks) {
                block.data.accumulate_flucs = false;
            }
        }
    }

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat)
	: State<NSE>(id, communicator, std::move(lat))
	{}
};

template <typename NSE>
int sim(int RESOLUTION = 2, const std::string& object_file = std::string(), bool enable_bouzidi = true)
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
    real PHYS_VELOCITY  = 1.5;      // [m/s] mean inflow; U_max = 1.5 * mean

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

    // Include object file name without extension in the state ID
    auto obj_path_for_id = std::filesystem::path(object_file.empty() ? std::string("none") : object_file);
    std::string obj_name_no_ext = obj_path_for_id.stem().string();
    const std::string state_id =
        fmt::format("sim2d_2_res{:02d}_np{:03d}_{}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD), obj_name_no_ext);
    StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);
    state.object_filename = object_file;
    state.enable_bouzidi = enable_bouzidi;

    if (!state.canCompute())
        return 0;

	// Set U_max (LBM units) for the parabolic profile
    real U_max_phys = 1.5 * PHYS_VELOCITY;
    state.u_max_lbm = lat.phys2lbmVelocity(U_max_phys);

    state.nse.physFinalTime = 8.0;
    state.cnt[PRINT].period = 0.05;

    // 2D = cut in 3D at z=0
    state.cnt[VTK2D].period = 0.05;
    state.add2Dcut_Z(0, "");

    execute(state);
    return 0;
}

template <typename TRAITS = TraitsDP>
void run(int RES, const std::string& object_file, bool enable_bouzidi)
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

	sim<NSE_CONFIG>(RES, object_file, enable_bouzidi);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim2d_1");
	program.add_description("Simple incompressible Navier-Stokes simulation example.");
	program.add_argument("resolution").help("resolution of the lattice").scan<'i', int>().default_value(1);
    program.add_argument("object_file").help("object file from sim_2D/ellipses (e.g. 8.txt)").default_value(std::string("sim_2D/ellipses/0.txt"));
    program.add_argument("--no-bouzidi").help("disable Bouzidi near-wall interpolation; treat near-wall as normal fluid").default_value(false).implicit_value(true);
    program.add_argument("--type1-bouzidi")
        .help("control whether geometry type 1 cells use Bouzidi ('on', 'off', or 'auto' to use build default)")
        .choices("auto", "on", "off")
        .default_value(std::string("auto"))
        .nargs(1);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << '\n';
		std::cerr << program;
		return 1;
	}

	const auto resolution = program.get<int>("resolution");
    const auto object_file = program.get<std::string>("object_file");
    const bool no_bouzidi = program.get<bool>("no-bouzidi");
    const auto type1_mode = program.get<std::string>("--type1-bouzidi");
	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}

    if (type1_mode == "on") {
        gUseBouzidiForType1 = true;
    } else if (type1_mode == "off") {
        gUseBouzidiForType1 = false;
    } else {
        gUseBouzidiForType1 = kDefaultUseBouzidiForType1;
    }

	run(resolution, object_file, !no_bouzidi);

	return 0;
}
