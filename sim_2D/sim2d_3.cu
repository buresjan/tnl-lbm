// Minimal solver modeled after sim2d_2:
// - accepts a geometry file
// - optional Bouzidi near-wall toggle
// - parabolic inflow with the same magnitude
// - outputs a single value: integral of instantaneous KE over the third-quarter ROI

#include <argparse/argparse.hpp>
#include <utility>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdio>

#include "lbm3d/core.h"
#include "lbm3d/lbm_data.h"
#include "lbm3d/d2q9/bc.h"
#include "lbm3d/d2q9/col_clbm.h"
#include "lbm3d/d2q9/macro.h"
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

// Parabolic inflow data; device-side inflow computes Poiseuille profile
template <typename TRAITS>
struct NSE2D_Data_ParabolicInflow : NSE_Data<TRAITS>
{
  using idx   = typename TRAITS::idx;
  using dreal = typename TRAITS::dreal;

  dreal u_max_lbm = 0;
  idx   y0 = 1;
  idx   y1 = 1;
  dreal inv_den = 1;

  template <typename LBM_KS>
  CUDA_HOSTDEV void inflow(LBM_KS& KS, idx, idx y, idx)
  {
    dreal s = (dreal)(y - y0) * inv_den;
    if (s < 0) s = 0; else if (s > 1) s = 1;
    KS.vx = u_max_lbm * (4.0 * s * (1.0 - s)); // Poiseuille profile
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

  using idx = typename TRAITS::idx;
  using real = typename TRAITS::real;
  using point_t = typename TRAITS::point_t;
  using lat_t = Lattice<3, real, idx>;

  // inflow control
  real u_max_lbm = 0;

  // geometry and Bouzidi control
  std::string object_filename;
  bool enable_bouzidi = true;

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

    if (enable_bouzidi) nse.allocateBouzidiCoeffArrays();

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

    long long count = 0, out_of_range = 0, parse_errors = 0, coeff_sets = 0, type_sets = 0;
    typename TRAITS::idx max_x = -1, max_y = -1;

    std::string line;
    long long line_no = 0;

    while (getline(fin, line)) {
      if (line.empty()) continue;
      std::istringstream iss(line);
      long long xi, yi; int cell_type; double c[8];
      if (!(iss >> xi >> yi >> cell_type >> c[0] >> c[1] >> c[2] >> c[3] >> c[4] >> c[5] >> c[6] >> c[7])) {
        parse_errors++; continue;
      }
      line_no++;

      // validate thetas
      for (int d=0; d<8; ++d) { if (c[d] > 1.0) { throw std::runtime_error("Bouzidi theta out of range (>1)"); } }

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

      // store Bouzidi thetas into the block-local host buffer (if enabled)
      if (enable_bouzidi) {
        for (auto& block : nse.blocks) {
          if (!block.isLocalIndex((idx)xi, (idx)yi, 0)) continue;
          for (int d=0; d<8; ++d) block.hBouzidi(d, (idx)xi, (idx)yi, 0) = (typename TRAITS::dreal)c[d];
          coeff_sets++; break;
        }
      }
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
      spdlog::error("Object grid mismatch or incomplete");
      throw std::runtime_error("Object file dimensions do not match simulation lattice");
    }

    // mirror host → device once after load
    if (enable_bouzidi) {
      for (auto& block : nse.blocks) {
        if (block.dBouzidi.getData() != nullptr) block.dBouzidi = block.hBouzidi;
      }
    }
  }

  // -------------------- per-step host update --------------------

  void updateKernelVelocities() override
  {
    // push inflow to device
    for (auto& block : nse.blocks) {
      block.data.u_max_lbm = u_max_lbm;
      block.data.y0 = 1;
      block.data.y1 = nse.lat.global.y() - 2;
      const int denom = std::max(1, (int)(block.data.y1 - block.data.y0));
      block.data.inv_den = 1.0 / (real)denom;
    }
  }

  void updateKernelData() override
  {
    State<NSE>::updateKernelData();
    for (auto& block : this->nse.blocks) {
      if (block.dBouzidi.getData() != nullptr && enable_bouzidi)
        block.data.bouzidi_coeff_ptr = block.dBouzidi.getData();
      else
        block.data.bouzidi_coeff_ptr = nullptr;
    }
  }

  // fluid mask helper
  bool isFluidLocal(const BLOCK& block, idx x, idx y) const
  {
    return BC::isFluid(block.hmap(x, y, 0));
  }

  // -------------------- instantaneous KE over ROI --------------------

  // integrate 0.5*(u^2+v^2) over x in [0.5W,0.75W), y in [1,Ny-2] (interior)
  real integrateInstantaneousKE_ROI_phys() const
  {
    const idx Nx = nse.lat.global.x();
    const idx Ny = nse.lat.global.y();
    const idx x0 = std::max<idx>(1, (idx)std::floor((real)0.5 * (real)Nx));
    const idx x1 = std::min<idx>(Nx - 1, (idx)std::ceil((real)0.75 * (real)Nx));
    const idx y0 = 1;
    const idx y1 = Ny - 1; // exclusive upper bound for iteration helper below
    if (x0 >= x1 || y0 >= y1) return (real)0;

    const double cell_area = (double)nse.lat.physDl * (double)nse.lat.physDl;
    double sum_ke = 0.0;
    for (const auto& block : nse.blocks) {
      const idx bx0 = std::max<idx>(x0, block.offset.x());
      const idx bx1 = std::min<idx>(x1, block.offset.x() + block.local.x());
      const idx by0 = std::max<idx>(y0, block.offset.y());
      const idx by1 = std::min<idx>(y1, block.offset.y() + block.local.y());
      if (bx0 >= bx1 || by0 >= by1) continue;
      for (idx x = bx0; x < bx1; ++x)
        for (idx y = by0; y < by1; ++y) {
          if (!block.isLocalIndex(x, y, 0) || !isFluidLocal(block, x, y)) continue;
          const real ux = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, 0));
          const real uy = nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, 0));
          sum_ke += 0.5 * ((double)ux*(double)ux + (double)uy*(double)uy);
        }
    }
    return (real)(sum_ke * cell_area);
  }

  void AfterSimFinished() override
  {
    // ensure up-to-date macro on host and write single value
    nse.copyMacroToHost();
    const double value = (double)integrateInstantaneousKE_ROI_phys();
    const std::string outpath = fmt::format("sim_2D/values/value_{}", std::filesystem::path(object_filename).filename().string());
    create_parent_directories(outpath.c_str());
    FILE* fp = fopen(outpath.c_str(), "wt");
    if (fp) { fprintf(fp, "%.17g\n", value); fclose(fp); }
    State<NSE>::AfterSimFinished();
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

  // physical scaling to match sim2d_2
  real LBM_VISCOSITY  = 1.0e-3; // lattice nu
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
  const std::string state_id = fmt::format("sim2d_3_res{:02d}_np{:03d}_{}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD), obj_name_no_ext);

  StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);
  state.object_filename = object_file;
  state.enable_bouzidi  = enable_bouzidi;

  if (!state.canCompute()) return 0;

  // parabolic inflow: U_max = 1.5 * U_mean
  real U_max_phys = 1.5 * PHYS_VELOCITY;
  state.u_max_lbm = lat.phys2lbmVelocity(U_max_phys);

  // timing: short and no VTKs; compute result at the end
  state.nse.physFinalTime =  4.0;
  state.cnt[PRINT].period = -0.1;
  state.cnt[VTK2D].period = -0.1;

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
    D2Q9_MACRO_Default<TRAITS>>;

  sim<NSE_CONFIG>(RES, object_file, enable_bouzidi);
}

int main(int argc, char** argv)
{
  TNLMPI_INIT mpi(argc, argv);

  argparse::ArgumentParser program("sim2d_3");
  program.add_description("Minimal 2D incompressible Navier-Stokes with geometry + instantaneous KE output.");
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

