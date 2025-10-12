#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/lbm_data.h"

#include "lbm3d/d2q9/bc.h"
#include "lbm3d/d2q9/col_srt.h"
#include "lbm3d/d2q9/col_clbm.h"
#include "lbm3d/d2q9/macro.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdio>                 // FILE*, fopen, fprintf
#include "lbm_common/fileutils.h" // create_parent_directories

namespace {
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

// -------------------- inflows and macros --------------------

template <typename TRAITS>
struct NSE2D_Data_ConstInflow : NSE_Data<TRAITS>
{
  using idx = typename TRAITS::idx;
  using dreal = typename TRAITS::dreal;

  dreal inflow_vx = 0;
  dreal inflow_vy = 0;

  template <typename LBM_KS>
  CUDA_HOSTDEV void inflow(LBM_KS& KS, idx, idx, idx)
  {
    KS.vx = inflow_vx;
    KS.vy = inflow_vy;
  }
};

// MACRO channels extended with accumulators for means and fluctuations
template <typename TRAITS>
struct D2Q9_MACRO_WithMean : D2Q9_MACRO_Base<TRAITS>
{
  using dreal = typename TRAITS::dreal;
  using idx   = typename TRAITS::idx;

  enum
  {
    e_rho,
    e_vx,
    e_vy,
    e_svx,              // sum vx (LBM)
    e_svy,              // sum vy (LBM)
    e_mean_vx_frozen,   // frozen <vx> (LBM)
    e_mean_vy_frozen,   // frozen <vy> (LBM)
    e_smag_uprime,      // sum |u'| (LBM)
    e_suprime2_sum,     // sum u'^2 (LBM^2)
    e_svprime2_sum,     // sum v'^2 (LBM^2)
    N
  };

  template <typename LBM_DATA, typename LBM_KS>
  __cuda_callable__ static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
  {
    // instantaneous fields (LBM)
    SD.macro(e_rho, x, y, z) = KS.rho;
    SD.macro(e_vx , x, y, z) = KS.vx;
    SD.macro(e_vy , x, y, z) = KS.vy;

    // accumulate running means when enabled
    if (SD.accumulate_means) {
      SD.macro(e_svx, x, y, z) += KS.vx;
      SD.macro(e_svy, x, y, z) += KS.vy;
    }

    // accumulate fluctuations around frozen mean when enabled
    if (SD.accumulate_flucs) {
      const dreal mvx = SD.macro(e_mean_vx_frozen, x, y, z);
      const dreal mvy = SD.macro(e_mean_vy_frozen, x, y, z);
      const dreal dux = KS.vx - mvx;
      const dreal duy = KS.vy - mvy;
      const dreal mag = sqrt(dux * dux + duy * duy);
      SD.macro(e_smag_uprime,  x, y, z) += mag;
      SD.macro(e_suprime2_sum, x, y, z) += dux * dux;
      SD.macro(e_svprime2_sum, x, y, z) += duy * duy;
    }
  }

  template <typename LBM_DATA, typename LBM_KS>
  __cuda_callable__ static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx, idx, idx)
  {
    KS.lbmViscosity = SD.lbmViscosity;
    KS.fx = SD.fx;
    KS.fy = SD.fy;
  }
};

// Parabolic inflow with device-side gates for accumulators
template <typename TRAITS>
struct NSE2D_Data_ParabolicInflow : NSE_Data<TRAITS>
{
  using idx   = typename TRAITS::idx;
  using dreal = typename TRAITS::dreal;

  dreal u_max_lbm = 0;
  idx   y0 = 1;
  idx   y1 = 1;
  dreal inv_den = 1;
  bool  accumulate_means = false;
  bool  accumulate_flucs = false;
  bool  use_bouzidi      = true;

  template <typename LBM_KS>
  CUDA_HOSTDEV void inflow(LBM_KS& KS, idx, idx y, idx)
  {
    dreal s = (dreal)(y - y0) * inv_den;
    if (s < 0) s = 0; else if (s > 1) s = 1;
    KS.vx = u_max_lbm * (4.0 * s * (1.0 - s)); // Poiseuille profile
    KS.vy = 0;
  }
};

// -------------------- simulation state --------------------

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
  using point_t = typename TRAITS::point_t;
  using lat_t = Lattice<3, real, idx>;

  // inflow controls
  real lbm_inflow_vx = 0;
  real u_max_lbm = 0;
  bool enable_bouzidi = true;

  // geometry file
  std::string object_filename;

  // host-side statistics control
  real stats_start_time = 1.5; // start collecting running mean
  real stats_end_time   = 5.5; // force-freeze deadline
  int  mean_samples = 0;

  // stabilization thresholds and cadence
  real mean_tol             = 1.0e-3; // abs [m/s]
  real mean_check_period    = 0.05;   // [s]
  int  mean_stable_required = 10;
  real mean_rel_tol         = 1.0e-3; // 0.1% relative
  real mean_min_time        = 1.0;    // guard before checking [s]

  bool means_frozen         = false;
  real mean_freeze_time     = -1;

  real next_mean_check_time   = stats_start_time + mean_check_period;
  real prev_domain_mean_speed = -1;

  int  fluc_samples            = 0;
  real fluc_tol                = 1.0e-3; // abs [m/s]
  real fluc_check_period       = 0.05;   // [s]
  int  fluc_stable_required    = 10;
  real fluc_rel_tol            = 1.0e-3; // 0.1% relative
  real fluc_min_time           = 1.0;    // after mean freeze [s]

  bool flucs_frozen            = false;
  real fluc_freeze_time        = -1;

  real next_fluc_check_time    = -1;
  real prev_domain_fluc_rms    = -1;

  bool tke_value_written       = false;

  // ROI expressed as fractions of domain length in x plus wall offsets in y
  real roi_x0_fraction = (real)0.5;  // start at 50% of domain width
  real roi_x1_fraction = (real)0.75; // end at 75% of domain width
  int  roi_y_offset_cells = 3;

  // -------------------- boundaries and geometry --------------------

  void setupBoundaries() override
  {
    if (!object_filename.empty()) {
      projectObjectFromFile(object_filename);
    }

    // channel: inflow left, outflow right, no-slip walls at y=1 and y=Ny-2
    nse.setBoundaryX(0, BC::GEO_INFLOW);
    nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_RIGHT);

    nse.setBoundaryY(1, BC::GEO_WALL);
    nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);

    // ghost layers for AA pattern
    nse.setBoundaryY(0, BC::GEO_NOTHING);
    nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);
  }

  // Load per-voxel type and Bouzidi thetas from file; map to BC types.
  void projectObjectFromFile(const std::string& fileArg)
  {
    using std::getline;
    namespace fs = std::filesystem;

    nse.allocateBouzidiCoeffArrays();

    const auto X = nse.lat.global.x();
    const auto Y = nse.lat.global.y();
    fs::path path = fileArg;
    if (!fileArg.empty() && fileArg.find('/') == std::string::npos && fileArg.find('\\') == std::string::npos) {
      path = fs::path("sim_2D/ellipses") / fileArg;
    }

    spdlog::info("Loading geometry '{}' (resolved '{}') domain {} x {}", fileArg, path.string(), X, Y);
    std::ifstream fin(path);
    if (!fin) {
      spdlog::error("Failed to open object file: {}", path.string());
      throw std::runtime_error("Cannot open object file");
    }

    long long count = 0, out_of_range = 0, parse_errors = 0, coeff_sets = 0, type_sets = 0, near_boundary_nearwall = 0;
    typename TRAITS::idx max_x = -1, max_y = -1;

    std::string line;
    long long line_no = 0, last_processed_line = 0;
    long long last_x = -1, last_y = -1;
    int last_type = -1;
    double last_coeffs[8] = {-1,-1,-1,-1,-1,-1,-1,-1};

    try {
      while (getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        long long xi, yi; int cell_type; double c[8];
        if (!(iss >> xi >> yi >> cell_type >> c[0] >> c[1] >> c[2] >> c[3] >> c[4] >> c[5] >> c[6] >> c[7])) {
          parse_errors++; continue;
        }
        line_no++; last_processed_line = line_no; last_x = xi; last_y = yi; last_type = cell_type;
        for (int d=0; d<8; ++d) last_coeffs[d] = c[d];

        // validate thetas
        for (int d=0; d<8; ++d) {
          if (c[d] > 1.0) {
            throw std::runtime_error(fmt::format(
              "Bouzidi theta out of range (>1) at line {} (x={}, y={}, dir={}, theta={}) in {}",
              (long long)line_no, xi, yi, d, c[d], path.string()
            ));
          }
        }

        count++;
        if (xi >= 0 && yi >= 0) { if (xi > max_x) max_x = (typename TRAITS::idx)xi; if (yi > max_y) max_y = (typename TRAITS::idx)yi; }
        if (xi < 0 || yi < 0 || xi >= X || yi >= Y) { out_of_range++; continue; }

        // type mapping
        typename BC::map_t mapval = BC::GEO_FLUID;
        switch (cell_type) {
          case 0: mapval = BC::GEO_FLUID; break;
          case 1: mapval = gUseBouzidiForType1 ? BC::GEO_FLUID_NEAR_WALL : BC::GEO_FLUID; break;
          case 2: mapval = BC::GEO_WALL; break;
          default: mapval = BC::GEO_FLUID; break;
        }
        nse.setMap((idx)xi, (idx)yi, 0, mapval);
        type_sets++;
        if (mapval == BC::GEO_FLUID_NEAR_WALL) {
          if (xi <= 0 || xi >= X - 1 || yi <= 0 || yi >= Y - 1) near_boundary_nearwall++;
        }

        // store Bouzidi thetas into the block-local host buffer
        for (auto& block : nse.blocks) {
          if (!block.isLocalIndex((idx)xi, (idx)yi, 0)) continue;
          for (int d=0; d<8; ++d) block.hBouzidi(d, (idx)xi, (idx)yi, 0) = (typename TRAITS::dreal)c[d];
          coeff_sets++; break;
        }
      }
    } catch (const std::exception& err) {
      spdlog::error(
        "Exception while loading geometry '{}': line={} x={} y={} type={} coeffs=[{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}] message={}",
        path.string(), last_processed_line, last_x, last_y, last_type,
        last_coeffs[0], last_coeffs[1], last_coeffs[2], last_coeffs[3],
        last_coeffs[4], last_coeffs[5], last_coeffs[6], last_coeffs[7], err.what()
      );
      throw;
    }

    // file-level sanity checks
    const auto infer_X = max_x + 1;
    const auto infer_Y = max_y + 1;
    bool dims_ok = (infer_X == X && infer_Y == Y);
    bool count_ok = (count == (long long)X * (long long)Y);
    spdlog::info(
      "Geometry load stats: rows={}, inferred dims={} x {}, type sets={}, coeff sets={}, parse errors={}, out-of-range={}",
      count, (long long)infer_X, (long long)infer_Y, type_sets, coeff_sets, parse_errors, out_of_range
    );
    if (!dims_ok || !count_ok) {
      spdlog::error("Object grid mismatch or incomplete: file dim=({},{}) inferred, count={} vs expected {}x{}={}.",
                    (long long)infer_X, (long long)infer_Y, count, (long long)X, (long long)Y, (long long)X * (long long)Y);
      throw std::runtime_error("Object file dimensions do not match simulation lattice");
    }

    if (parse_errors > 0 || out_of_range > 0 || near_boundary_nearwall > 0) {
      spdlog::warn("While loading object: parse_errors={}, out_of_range={}, near_boundary_nearwall={}",
                   parse_errors, out_of_range, near_boundary_nearwall);
    }

    // host->device for Bouzidi coeffs
    for (auto& block : nse.blocks) {
      if (block.dBouzidi.getData() != nullptr) block.dBouzidi = block.hBouzidi;
    }
  }

  // -------------------- VTK outputs --------------------

  bool outputData(const BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) override
  {
    int k = 0;
    if (index == k++) return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho, x, y, z), 1, desc, value, dofs);
    if (index == k++) {
      switch (dof) {
        case 0: return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, z)), 3, desc, value, dofs);
        case 1: return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, z)), 3, desc, value, dofs);
        case 2: return vtk_helper("velocity", 0, 3, desc, value, dofs);
      }
    }
    if (index == k++) {
      real mvx = 0; if (mean_samples > 0) mvx = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_svx, x, y, z) / (real)mean_samples);
      return vtk_helper("mean_vx", mvx, 1, desc, value, dofs);
    }
    if (index == k++) {
      real mvy = 0; if (mean_samples > 0) mvy = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_svy, x, y, z) / (real)mean_samples);
      return vtk_helper("mean_vy", mvy, 1, desc, value, dofs);
    }
    if (index == k++) {
      real mf = 0; if (fluc_samples > 0) mf = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_smag_uprime, x, y, z) / (real)fluc_samples);
      return vtk_helper("mean_fluc_mag", mf, 1, desc, value, dofs);
    }
    if (index == k++) {
      real mag = 0;
      if (means_frozen) {
        const real mvx = block.hmacro(MACRO::e_mean_vx_frozen, x, y, z);
        const real mvy = block.hmacro(MACRO::e_mean_vy_frozen, x, y, z);
        mag = nse.lat.lbm2physVelocity(std::sqrt(mvx*mvx + mvy*mvy));
      } else if (mean_samples > 0) {
        const real mvx = block.hmacro(MACRO::e_svx, x, y, z) / (real)mean_samples;
        const real mvy = block.hmacro(MACRO::e_svy, x, y, z) / (real)mean_samples;
        mag = nse.lat.lbm2physVelocity(std::sqrt(mvx*mvx + mvy*mvy));
      }
      return vtk_helper("mean_vel_mag", mag, 1, desc, value, dofs);
    }
    if (index == k++) {
      real vx = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, z));
      real vy = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, z));
      return vtk_helper("velocity_magnitude", std::sqrt(vx*vx + vy*vy), 1, desc, value, dofs);
    }

    // raw Bouzidi thetas (debugging)
    const bool have_bouzidi = (block.hBouzidi.getData() != nullptr);
    if (index == k++) { real v = (real)-1; if (have_bouzidi) v = (real) block.hBouzidi(0, x, y, z); return vtk_helper("bouzidi_east", v, 1, desc, value, dofs); }
    if (index == k++) { real v = (real)-1; if (have_bouzidi) v = (real) block.hBouzidi(1, x, y, z); return vtk_helper("bouzidi_north", v, 1, desc, value, dofs); }
    if (index == k++) { real v = (real)-1; if (have_bouzidi) v = (real) block.hBouzidi(2, x, y, z); return vtk_helper("bouzidi_west", v, 1, desc, value, dofs); }
    if (index == k++) { real v = (real)-1; if (have_bouzidi) v = (real) block.hBouzidi(3, x, y, z); return vtk_helper("bouzidi_south", v, 1, desc, value, dofs); }
    if (index == k++) { real v = (real)-1; if (have_bouzidi) v = (real) block.hBouzidi(4, x, y, z); return vtk_helper("bouzidi_ne", v, 1, desc, value, dofs); }
    if (index == k++) { real v = (real)-1; if (have_bouzidi) v = (real) block.hBouzidi(5, x, y, z); return vtk_helper("bouzidi_nw", v, 1, desc, value, dofs); }
    if (index == k++) { real v = (real)-1; if (have_bouzidi) v = (real) block.hBouzidi(6, x, y, z); return vtk_helper("bouzidi_sw", v, 1, desc, value, dofs); }
    if (index == k++) { real v = (real)-1; if (have_bouzidi) v = (real) block.hBouzidi(7, x, y, z); return vtk_helper("bouzidi_se", v, 1, desc, value, dofs); }

    return false;
  }

  // -------------------- per-step host update --------------------

  void updateKernelVelocities() override
  {
    const real t = nse.physTime();

    // enable mean accumulation within the window
    const bool do_acc_means = (!means_frozen) && (t >= stats_start_time) && (t < stats_end_time);
    if (do_acc_means) mean_samples++;

    // push inflow and gates to device
    for (auto& block : nse.blocks) {
      block.data.u_max_lbm = u_max_lbm;
      block.data.y0 = 1;
      block.data.y1 = nse.lat.global.y() - 2;
      const int denom = std::max(1, (int)(block.data.y1 - block.data.y0));
      block.data.inv_den = 1.0 / (real)denom;
      block.data.use_bouzidi = enable_bouzidi;
      block.data.accumulate_means = do_acc_means;
    }

    // mean stabilization and freeze
    if (!means_frozen) {
      checkAndMaybeFreezeMeans();
      if (means_frozen) snapshotFrozenMeansToMacro();
    }

    // hard stop for mean accumulation if never stabilized
    if (!means_frozen && t >= stats_end_time) {
      means_frozen     = true;
      mean_freeze_time = stats_end_time;
      snapshotFrozenMeansToMacro();
    }

    // after mean freeze, accumulate fluctuations after a guard time
    if (means_frozen) {
      const bool gate_on = (!flucs_frozen) && (t >= mean_freeze_time + fluc_min_time);
      if (gate_on) fluc_samples++;
      for (auto& block : nse.blocks) block.data.accumulate_flucs = gate_on;

      if (!flucs_frozen) checkAndMaybeFreezeFlucMag();

      // when fluctuations stabilize, export TKE once and terminate
      if (flucs_frozen && !tke_value_written) {
        exportROI_TKE_andTerminate();
      }
    }
  }

  void updateKernelData() override
  {
    // also pass Bouzidi pointers to kernels
    State<NSE>::updateKernelData();
    for (auto& block : this->nse.blocks) {
      if (block.dBouzidi.getData() != nullptr) block.data.bouzidi_coeff_ptr = block.dBouzidi.getData();
      else block.data.bouzidi_coeff_ptr = nullptr;
    }
  }

  // -------------------- ROI helpers --------------------

  // convert ROI fractions to lattice indices and clamp to interior
  void roiIndices(idx& x0, idx& x1, idx& y0, idx& y1) const
  {
    const idx Nx = nse.lat.global.x();
    const idx Ny = nse.lat.global.y();
    const real x0_pos = roi_x0_fraction * (real)Nx;
    const real x1_pos = roi_x1_fraction * (real)Nx;

    x0 = std::max<idx>(1, (idx)std::floor(x0_pos));
    x1 = std::min<idx>(Nx - 1, (idx)std::ceil(x1_pos));
    y0 = std::max<idx>(1, (idx)roi_y_offset_cells);
    y1 = std::min<idx>(Ny - 1, (idx)(Ny - roi_y_offset_cells));

    if (x0 >= x1) { x0 = 1; x1 = Nx - 1; } // safety
    if (y0 >= y1) { y0 = 1 + roi_y_offset_cells; y1 = Ny - 1 - roi_y_offset_cells; }
  }

  // -------------------- mean freeze snapshot --------------------

  // write frozen <u> per cell safely over each block's local extent
  void snapshotFrozenMeansToMacro()
  {
    nse.copyMacroToHost(); // get latest sums

    // stop mean accumulation immediately
    for (auto& block : nse.blocks) block.data.accumulate_means = false;

    // compute frozen mean and reset fluct accumulators
    for (auto& block : nse.blocks) {
      const idx Nx = nse.lat.global.x();
      const idx Ny = nse.lat.global.y();
      const idx bx0 = std::max<idx>(1, block.offset.x());
      const idx bx1 = std::min<idx>(Nx - 1, block.offset.x() + block.local.x());
      const idx by0 = std::max<idx>(1, block.offset.y());
      const idx by1 = std::min<idx>(Ny - 1, block.offset.y() + block.local.y());
      for (idx x = bx0; x < bx1; ++x)
        for (idx y = by0; y < by1; ++y) {
          const real mvx = (mean_samples > 0) ? block.hmacro(MACRO::e_svx, x, y, 0) / (real)mean_samples : (real)0;
          const real mvy = (mean_samples > 0) ? block.hmacro(MACRO::e_svy, x, y, 0) / (real)mean_samples : (real)0;
          block.hmacro(MACRO::e_mean_vx_frozen, x, y, 0) = mvx;
          block.hmacro(MACRO::e_mean_vy_frozen, x, y, 0) = mvy;
          block.hmacro(MACRO::e_smag_uprime,    x, y, 0) = 0;
          block.hmacro(MACRO::e_suprime2_sum,   x, y, 0) = 0;
          block.hmacro(MACRO::e_svprime2_sum,   x, y, 0) = 0;
        }
    }

    // push frozen mean to device and start fluc accumulation next step
    nse.copyMacroToDevice();
    fluc_samples = 0;
    flucs_frozen = false;
    prev_domain_fluc_rms = (real)-1;
    next_fluc_check_time = mean_freeze_time + fluc_check_period;
    for (auto& block : nse.blocks) block.data.accumulate_flucs = true;

    // log a snapshot
    writeStatsSnapshot("mean_frozen");
  }

  void writeVTKs_3D() override {}

  // fluid mask helper
  bool isFluidLocal(const BLOCK& block, idx x, idx y) const
  {
    return BC::isFluid(block.hmap(x, y, 0));
  }

  // -------------------- ROI-based stabilization metrics --------------------

  // average |<u>| over ROI [m/s]
  real computeROIAvgMeanSpeed_phys() const
  {
    if (mean_samples <= 0) return 0;
    idx x0,x1,y0,y1; roiIndices(x0,x1,y0,y1);
    double sum = 0.0; long long cnt = 0;
    for (const auto& block : nse.blocks) {
      const idx bx0 = std::max<idx>(x0, block.offset.x());
      const idx bx1 = std::min<idx>(x1, block.offset.x() + block.local.x());
      const idx by0 = std::max<idx>(y0, block.offset.y());
      const idx by1 = std::min<idx>(y1, block.offset.y() + block.local.y());
      if (bx0 >= bx1 || by0 >= by1) continue;
      for (idx x = bx0; x < bx1; ++x)
        for (idx y = by0; y < by1; ++y) {
          if (!block.isLocalIndex(x, y, 0) || !isFluidLocal(block, x, y)) continue;
          const real mvx = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_svx, x, y, 0) / (real)mean_samples);
          const real mvy = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_svy, x, y, 0) / (real)mean_samples);
          sum += std::sqrt((double)mvx*mvx + (double)mvy*mvy);
          ++cnt;
        }
    }
    return (real)(cnt ? sum / (double)cnt : 0);
  }

  // RMS fluctuation speed sqrt(<u'^2+v'^2>) over ROI [m/s]
  real computeROIRMSFlucSpeed_phys() const
  {
    if (fluc_samples <= 0) return 0;
    idx x0,x1,y0,y1; roiIndices(x0,x1,y0,y1);
    double sum_sq_lbm = 0.0; long long cnt = 0;
    for (const auto& block : nse.blocks) {
      const idx bx0 = std::max<idx>(x0, block.offset.x());
      const idx bx1 = std::min<idx>(x1, block.offset.x() + block.local.x());
      const idx by0 = std::max<idx>(y0, block.offset.y());
      const idx by1 = std::min<idx>(y1, block.offset.y() + block.local.y());
      if (bx0 >= bx1 || by0 >= by1) continue;
      for (idx x = bx0; x < bx1; ++x)
        for (idx y = by0; y < by1; ++y) {
          if (!block.isLocalIndex(x, y, 0) || !isFluidLocal(block, x, y)) continue;
          const real up2 = block.hmacro(MACRO::e_suprime2_sum, x, y, 0) / (real)fluc_samples;
          const real vp2 = block.hmacro(MACRO::e_svprime2_sum, x, y, 0) / (real)fluc_samples;
          sum_sq_lbm += (double)(up2 + vp2);
          ++cnt;
        }
    }
    if (!cnt) return 0;
    const real vscale = nse.lat.lbm2physVelocity((real)1);
    return (real)std::sqrt(std::max(0.0, (sum_sq_lbm / (double)cnt) * (double)vscale * (double)vscale));
  }

  // -------------------- TKE in ROI --------------------

  // per-cell TKE from accumulated squares [m^2/s^2]
  real computeCellTKE_phys(const BLOCK& block, idx x, idx y) const
  {
    if (fluc_samples <= 0) return (real)0;
    const real up2 = block.hmacro(MACRO::e_suprime2_sum, x, y, 0) / (real)fluc_samples;
    const real vp2 = block.hmacro(MACRO::e_svprime2_sum, x, y, 0) / (real)fluc_samples;
    const real vscale = nse.lat.lbm2physVelocity((real)1);
    return (real)0.5 * (up2 + vp2) * vscale * vscale;
  }

  // integrate TKE over ROI; includes fallbacks if fluc_samples==0
  real integrateTKE_ROI_phys() const
  {
    idx x0,x1,y0,y1; roiIndices(x0,x1,y0,y1);
    if (x0 >= x1 || y0 >= y1) return (real)0;
    const double cell_area = (double)nse.lat.physDl * (double)nse.lat.physDl;

    // A) preferred: use accumulated squares
    if (fluc_samples > 0) {
      double sum_tke = 0.0;
      for (const auto& block : nse.blocks) {
        const idx bx0 = std::max<idx>(x0, block.offset.x());
        const idx bx1 = std::min<idx>(x1, block.offset.x() + block.local.x());
        const idx by0 = std::max<idx>(y0, block.offset.y());
        const idx by1 = std::min<idx>(y1, block.offset.y() + block.local.y());
        if (bx0 >= bx1 || by0 >= by1) continue;
        for (idx x = bx0; x < bx1; ++x)
          for (idx y = by0; y < by1; ++y) {
            if (!block.isLocalIndex(x, y, 0) || !isFluidLocal(block, x, y)) continue;
            sum_tke += (double)computeCellTKE_phys(block, x, y);
          }
      }
      return (real)(sum_tke * cell_area);
    }

    // B1) deviations vs running mean if available
    if (mean_samples > 0) {
      double sum_tke = 0.0;
      for (const auto& block : nse.blocks) {
        const idx bx0 = std::max<idx>(x0, block.offset.x());
        const idx bx1 = std::min<idx>(x1, block.offset.x() + block.local.x());
        const idx by0 = std::max<idx>(y0, block.offset.y());
        const idx by1 = std::min<idx>(y1, block.offset.y() + block.local.y());
        if (bx0 >= bx1 || by0 >= by1) continue;
        for (idx x = bx0; x < bx1; ++x)
          for (idx y = by0; y < by1; ++y) {
            if (!block.isLocalIndex(x, y, 0) || !isFluidLocal(block, x, y)) continue;
            const real dux = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, 0) - block.hmacro(MACRO::e_svx, x, y, 0) / (real)mean_samples);
            const real duy = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, 0) - block.hmacro(MACRO::e_svy, x, y, 0) / (real)mean_samples);
            sum_tke += 0.5 * ((double)dux*(double)dux + (double)duy*(double)duy);
          }
      }
      return (real)(sum_tke * cell_area);
    }

    // B2) deviations vs instantaneous ROI mean
    double sx = 0.0, sy = 0.0; long long cnt = 0;
    for (const auto& block : nse.blocks) {
      const idx bx0 = std::max<idx>(x0, block.offset.x());
      const idx bx1 = std::min<idx>(x1, block.offset.x() + block.local.x());
      const idx by0 = std::max<idx>(y0, block.offset.y());
      const idx by1 = std::min<idx>(y1, block.offset.y() + block.local.y());
      if (bx0 >= bx1 || by0 >= by1) continue;
      for (idx x = bx0; x < bx1; ++x)
        for (idx y = by0; y < by1; ++y) {
          if (!block.isLocalIndex(x, y, 0) || !isFluidLocal(block, x, y)) continue;
          sx += (double)block.hmacro(MACRO::e_vx, x, y, 0);
          sy += (double)block.hmacro(MACRO::e_vy, x, y, 0);
          ++cnt;
        }
    }
    if (!cnt) return 0;
    const real mvx_lbm = (real)(sx / (double)cnt);
    const real mvy_lbm = (real)(sy / (double)cnt);
    double sum_tke = 0.0;
    for (const auto& block : nse.blocks) {
      const idx bx0 = std::max<idx>(x0, block.offset.x());
      const idx bx1 = std::min<idx>(x1, block.offset.x() + block.local.x());
      const idx by0 = std::max<idx>(y0, block.offset.y());
      const idx by1 = std::min<idx>(y1, block.offset.y() + block.local.y());
      if (bx0 >= bx1 || by0 >= by1) continue;
      for (idx x = bx0; x < bx1; ++x)
        for (idx y = by0; y < by1; ++y) {
          if (!block.isLocalIndex(x, y, 0) || !isFluidLocal(block, x, y)) continue;
          const real dux = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, 0) - mvx_lbm);
          const real duy = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, 0) - mvy_lbm);
          sum_tke += 0.5 * ((double)dux*(double)dux + (double)duy*(double)duy);
        }
    }
    return (real)(sum_tke * cell_area);
  }

  // -------------------- tiny CSV logger --------------------

  // append one row per call in sim_2D/stats/stats_<object>.csv
  void writeStatsSnapshot(const char* tag)
  {
    // derive base name from object file; handle empty gracefully
    std::string base = std::filesystem::path(object_filename).stem().string();
    if (base.empty()) base = "none";
    const std::string out = fmt::format("sim_2D/stats/stats_{}.csv", base);
    create_parent_directories(out.c_str());
    const bool exists = std::filesystem::exists(out);

    // ensure host has latest accumulators
    nse.copyMacroToHost();

    // compute ROI metrics in SI
    const real t_now    = nse.physTime();
    const real roi_mean = computeROIAvgMeanSpeed_phys();  // m/s
    const real roi_rms  = computeROIRMSFlucSpeed_phys();  // m/s

    FILE* fp = fopen(out.c_str(), exists ? "at" : "wt");
    if (!fp) return;
    if (!exists) {
      fprintf(fp, "tag,time_s,mean_frozen,mean_freeze_time_s,fluc_frozen,fluc_freeze_time_s,mean_samples,fluc_samples,roi_avg_mean_speed_mps,roi_rms_fluc_mps\n");
    }
    fprintf(fp, "%s,%.9g,%d,%.9g,%d,%.9g,%d,%d,%.9g,%.9g\n",
            tag,
            (double)t_now,
            means_frozen?1:0,
            (double)mean_freeze_time,
            flucs_frozen?1:0,
            (double)fluc_freeze_time,
            mean_samples,
            fluc_samples,
            (double)roi_mean,
            (double)roi_rms);
    fclose(fp);
  }

  // -------------------- export and finalize --------------------

  // compute and write ROI TKE once; then request graceful termination
  void exportROI_TKE_andTerminate()
  {
    if (tke_value_written) return;

    writeStatsSnapshot("export"); // log before export
    nse.copyMacroToHost();        // sync
    const double value = (double)integrateTKE_ROI_phys();

    const std::string outpath = fmt::format("sim_2D/values/value_{}", std::filesystem::path(object_filename).filename().string());
    create_parent_directories(outpath.c_str());
    FILE* fp = fopen(outpath.c_str(), "wt");
    if (fp) { fprintf(fp, "%.17g\n", value); fclose(fp); }

    tke_value_written = true;
    this->nse.terminate = true;
  }

  void AfterSimFinished() override
  {
    if (!tke_value_written) {
      writeStatsSnapshot("final"); // last log
      nse.copyMacroToHost();
      const double value = (double)integrateTKE_ROI_phys();
      const std::string outpath = fmt::format("sim_2D/values/value_{}", std::filesystem::path(object_filename).filename().string());
      create_parent_directories(outpath.c_str());
      FILE* fp = fopen(outpath.c_str(), "wt");
      if (fp) { fprintf(fp, "%.17g\n", value); fclose(fp); }
      tke_value_written = true;
    }
    State<NSE>::AfterSimFinished();
  }

  // -------------------- stabilization checks --------------------

  // freeze <u> when ROI average |<u>| stabilizes (abs/rel tol + guard time)
  void checkAndMaybeFreezeMeans()
  {
    const real t = nse.physTime();
    if (t < stats_start_time + mean_min_time || means_frozen) return;
    if (t + (real)1e-12 < next_mean_check_time) return;

    nse.copyMacroToHost();
    const real curr = computeROIAvgMeanSpeed_phys();

    static int stable_hits = 0;
    if (prev_domain_mean_speed < (real)0) {
      stable_hits = 0;
    } else {
      const real delta  = std::abs(curr - prev_domain_mean_speed);
      const real thresh = std::max(mean_tol, mean_rel_tol * std::max(curr, (real)1e-6));
      stable_hits = (delta <= thresh) ? (stable_hits + 1) : 0;
    }

    prev_domain_mean_speed = curr;
    next_mean_check_time  += mean_check_period;

    if (stable_hits >= mean_stable_required) {
      means_frozen     = true;
      mean_freeze_time = t;
    }
  }

  // freeze fluc accumulation when ROI RMS(|u'|) stabilizes
  void checkAndMaybeFreezeFlucMag()
  {
    if (flucs_frozen || fluc_samples <= 0) return;
    const real t = nse.physTime();
    if (t + (real)1e-12 < next_fluc_check_time || t < mean_freeze_time + fluc_min_time) return;

    nse.copyMacroToHost();
    const real curr = computeROIRMSFlucSpeed_phys();

    static int stable_hits = 0;
    if (prev_domain_fluc_rms < (real)0) {
      stable_hits = 0;
    } else {
      const real delta  = std::abs(curr - prev_domain_fluc_rms);
      const real thresh = std::max(fluc_tol, fluc_rel_tol * std::max(curr, (real)1e-6));
      stable_hits = (delta <= thresh) ? (stable_hits + 1) : 0;
    }

    prev_domain_fluc_rms  = curr;
    next_fluc_check_time += fluc_check_period;

    if (stable_hits >= fluc_stable_required) {
      flucs_frozen  = true;
      fluc_freeze_time = t;
      for (auto& block : nse.blocks) block.data.accumulate_flucs = false;
      writeStatsSnapshot("fluc_frozen"); // log at freeze
    }
  }

  StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat)
  : State<NSE>(id, communicator, std::move(lat)) {}
};

// --------------------------- driver ---------------------------

template <typename NSE>
int sim(int RESOLUTION = 2, const std::string& object_file = std::string(), bool enable_bouzidi = true)
{
  using idx = typename NSE::TRAITS::idx;
  using real = typename NSE::TRAITS::real;
  using point_t = typename NSE::TRAITS::point_t;
  using lat_t = Lattice<3, real, idx>;

  // grid: width X, height Y; AA pattern has ghost at y=0, y=Ny-1
  int block_size = 32;
  int X = 128 * RESOLUTION;
  int Y = block_size * RESOLUTION;

  // fluid and scaling: set physical height = 0.50 m -> width ≈ 2.0 m with this aspect
  real LBM_VISCOSITY  = 1.0e-3; // lattice nu (gives tau ~ 0.56)
  real PHYS_HEIGHT    = 0.50;   // meters
  real PHYS_VISCOSITY = 1.0e-3; // m^2/s
  real PHYS_VELOCITY  = 1.0;    // mean inflow m/s

  // conversions
  real PHYS_DL = PHYS_HEIGHT / ((real)Y - 2); // exclude walls
  real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
  point_t PHYS_ORIGIN = {0., 0., 0.};

  // build lattice
  lat_t lat;
  lat.global        = typename lat_t::CoordinatesType(X, Y, 1);
  lat.physOrigin    = PHYS_ORIGIN;
  lat.physDl        = PHYS_DL;
  lat.physDt        = PHYS_DT;
  lat.physViscosity = PHYS_VISCOSITY;

  // state id includes object name
  auto obj_path_for_id = std::filesystem::path(object_file.empty() ? std::string("none") : object_file);
  std::string obj_name_no_ext = obj_path_for_id.stem().string();
  const std::string state_id =
      fmt::format("sim2d_2_res{:02d}_np{:03d}_{}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD), obj_name_no_ext);

  StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);
  state.object_filename = object_file;
  state.enable_bouzidi  = enable_bouzidi;

  if (!state.canCompute()) return 0;

  // parabolic inflow: U_max = 1.5 * U_mean
  real U_max_phys = 1.5 * PHYS_VELOCITY;
  state.u_max_lbm = lat.phys2lbmVelocity(U_max_phys);

  // timing and outputs
  state.nse.physFinalTime = 10.0;
  state.cnt[PRINT].period = 0.05;
  state.cnt[VTK2D].period = -0.05;
  state.add2Dcut_Z(0, "");

  execute(state);
  return 0;
}

template <typename TRAITS = TraitsDP>
void run(int RES, const std::string& object_file, bool enable_bouzidi)
{
  using COLL = D2Q9_CLBM<TRAITS>;

  using NSE_CONFIG = LBM_CONFIG<
    TRAITS,
    D2Q9_KernelStruct,
    NSE2D_Data_ParabolicInflow<TRAITS>,
    COLL,
    typename COLL::EQ,
    D2Q9_STREAMING<TRAITS>,
    D2Q9_BC_All,
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
      .help("control whether geometry type 1 cells use Bouzidi ('on', 'off', or 'auto')")
      .choices("auto", "on", "off")
      .default_value(std::string("auto"));

  try { program.parse_args(argc, argv); }
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
