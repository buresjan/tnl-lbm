# LBM3D Header Overview

This directory contains the header-only implementation of the three-dimensional
Lattice Boltzmann Method (LBM) solver that is built on top of the Template
Numerical Library (TNL).  The goal of this README is to provide an exhaustive
map of the files present in this folder and to explain the role that every
component plays in the overall architecture of the solver.  The descriptions
below are intentionally verbose so that newcomers can navigate the code base
without opening every header individually.

## Top-Level Files

### `adios_writer.h` / `adios_writer.hpp`
Implements `ADIOSWriter`, a helper class that wraps the [ADIOS2](https://adios2.readthedocs.io) I/O
library.  Instances are constructed with domain metadata (global size, local
size, offset and physical origin) and optionally an output cycle.  The class
records scalar or field variables, attaches VTK/Fides attributes and writes them
into BP4 files.  It is used by the state classes for high‑performance output of
simulation checkpoints and visualisation data.

### `block_size_optimizer.h`
Contains heuristics for choosing an efficient CUDA thread block geometry for the
LBM kernels.  Given the local subdomain size, permutation of axes and hardware
limits it searches for a block size that maximises occupancy while respecting
warp size multiples and hardware limits.  The function logs the selected block
shape so users can reason about performance.

### `checkpoint.h`
`CheckpointManager` provides a minimal wrapper for storing and restoring
simulation state using ADIOS2.  It opens a file, begins a logical step, allows
variables and attributes to be saved or loaded and then finalises the step.
LBM blocks can use it to dump distribution functions, macroscopic fields or
custom arrays for restart capability.

### `core.h`
A convenience header that includes the default set of D3Q27 components
(macroscopic quantity evaluation, boundary conditions, equilibrium functions,
streaming patterns and collision operators) and defines `execute`, the main
simulation loop.  `execute` performs initialisation, iterates the time stepping
loop, triggers I/O counters and handles termination conditions and MPI
synchronisation.

### `defs.h`
Central definitions used throughout the solver.  It selects the streaming
pattern (A–A or A–B), sets the maximum number of distribution functions
(`DFMAX`), chooses the device type, defines the `Traits` template capturing
precision and index types and provides `KernelStruct` specialisations for the
D2Q9, D3Q7 and D3Q27 stencils.  It also defines the `LBM_CONFIG` aggregate that
binds together traits, data layout, collision operators, equilibrium model and
boundary conditions.

### `dirac.h`
Implements discrete Dirac delta kernels used by the immersed boundary method
(IBM).  It provides several radial basis variants (`phi1`–`phi4`), helper
routines that test whether the support of the kernel is non‑zero and a 3D
product form used when coupling Lagrangian points with the Eulerian lattice.

### `ibm_kernels.h`
CUDA kernels for assembling the sparse matrices that couple Lagrangian points
with lattice nodes in IBM.  The kernels compute row capacities for the coupling
matrices, fill the matrices with weighted Dirac delta values and build auxiliary
matrices used to project velocities and forces between descriptions.  The code
is templated over an `LBM` configuration and heavily uses `Lagrange3D` helpers.

### `kernels.h`
Defines the core CUDA kernels (and host equivalents) that advance the
simulation.  `kernelInitIndices` computes neighbour indices with awareness of
periodic boundaries and MPI overlaps.  `cudaLBMKernel` (and its CPU version)
performs the collision and streaming steps together with macroscopic quantity
updates.  Overloads exist for coupled systems such as NSE–ADE where two LBM
states interact during the update.

### `lagrange_3D.h` / `lagrange_3D.hpp`
Large component managing the IBM Lagrangian mesh.  The `Lagrange3D` structure
holds vectors of Lagrangian points, sparse matrices that project quantities
between Eulerian and Lagrangian descriptions, and iterative solvers for the IBM
force calculation.  It supports CPU, GPU and hybrid assembly paths, provides
facilities to compute point spacing, integrates total forces and exposes accessors to macroscopic fields as flat vectors.  The companion implementation file
defines the algorithms for converting point coordinates, allocating matrices and
performing the matrix‑vector operations required by the IBM correction step.

### `lattice.h`
Represents the metadata of a D‑dimensional equidistant lattice.  Besides the
global size it stores the physical origin, spatial and temporal steps and
provides conversions between lattice units and physical quantities (length,
velocity, viscosity and force).  The class is used by `LBM` to describe the
geometry of each subdomain and to translate between lattice indices and
physical coordinates.

### `lattice_decomposition.h`
Utilities for splitting the global lattice into MPI subdomains.  It offers a
simple 1D decomposition routine (`decomposeLattice_D1Q3`) and generic helpers
for optimal block partitioning with axis permutations.  The file also contains
functions that derive neighbour relationships required for
`DistributedNDArraySynchronizer` based on a chosen synchronisation pattern and
block layout.

### `lbm.h` / `lbm.hpp`
Defines the high‑level `LBM` class which represents the solver state for a
single physical model.  It stores lattice metadata, a vector of `LBM_BLOCK`
subdomains and provides numerous helpers for data management.  Methods exist for
allocating host/device arrays, copying maps and macroscopic fields, updating
kernel data, iterating over local or global lattice sites and computing physical
numbers such as Reynolds.  The implementation file contains the function bodies
and MPI communication routines.

### `lbm_block.h` / `lbm_block.hpp`
`LBM_BLOCK` models a local lattice subdomain.  Each block owns its portion of
map data, macroscopic fields and distribution functions on both host and device.
It keeps track of neighbours, MPI synchronisers, CUDA launch parameters and
provides methods for copying data between host and device or across MPI ranks.
Blocks are created by the decomposition utilities and orchestrated by the `LBM`
class.

### `lbm_data.h`
Collection of POD‑style structures that are copied to the device and used by the
kernels.  `LBM_Data` stores pointers to distribution functions, macroscopic
fields and the map together with lattice dimensions.  `NSE_Data` and
`ADE_Data` extend the base with forcing terms, inflow parameters and pointers to
variable diffusion coefficients or heat‑transfer flags.  These structures allow
the kernels to access simulation data efficiently without touching C++
containers.

### `nonNewtonian.h`
Additional kernels and helpers supporting non‑Newtonian rheology.  The file
contains diagnostic kernels used to validate velocity fields, routines that
calculate strain rate tensors and conditionally compile extended collision
operators for models such as Carreau–Yasuda or Casson fluids.

### `obstacles_ibm.h`
High‑level routines that populate the Lagrangian point cloud for IBM obstacles
(e.g. rectangles, cylinders or spheres).  They estimate point counts from a
target spacing `sigma`, apply geometric transformations and insert the points
into the `Lagrange3D` structure while reporting statistics about point spacing.

### `obstacles_lbm.h`
Procedural geometry generators that operate directly on the Eulerian lattice by
marking cells in the map array.  Helper functions draw cubes, spheres,
cylinders, bounding boxes and more complex shapes such as the CUBI benchmark,
allowing simple solid obstacles to be defined without IBM.

### `state.h` / `state.hpp`
`State` encapsulates a complete simulation instance.  It manages counters for
periodic actions, directories for output, flag files used to control execution
and a `CheckpointManager`.  The class owns an `LBM` object representing the
Navier–Stokes equations and an optional `Lagrange3D` instance for IBM.  The
implementation provides numerous I/O helpers: creation of VTK files, sampling of
1D/2D/3D cuts, PNG projections and asynchronous control via flag files.

### `state_NSE_ADE.h`
Derived state that couples a fluid solver (NSE) with an advection–diffusion
solver (ADE).  It owns two `LBM` instances that share the same traits, forwards
calls to both solvers, synchronises their data and overrides `reset` and
`SimUpdate` to evolve the coupled system.  It highlights how the framework can
host multi‑physics simulations.

### `vtk_writer.h` / `vtk_writer.hpp`
A lightweight VTK file writer used for legacy `.vtk` outputs.  The writer keeps a
large buffer to accumulate floats and ensures big‑endian representation required
by the binary format.  It exposes low‑level routines for writing headers and
primitive types and is used by `State` when exporting point clouds or cut planes
for quick visualisation.

## Subdirectories

### `d2q9`
Contains the specialised implementation for the 2D nine‑velocity (D2Q9) stencil.
Files follow a common pattern:

- `bc.h` – boundary conditions and mapping helpers for typical 2D setups
  (inflow/outflow, symmetry and periodic walls).
- `col_*.h` – collision operators such as BGK or cumulant versions tuned for the
  D2Q9 lattice.
- `common.h` – shared constants (lattice weights, discrete velocities).
- `eq.h` – equilibrium distribution function.
- `macro.h` – macroscopic quantity extraction and output routines.
- `streaming_AA.h` / `streaming_AB.h` – streaming step for A–A or A–B memory
  layouts.

These headers enable running 2D simulations within the otherwise 3D oriented
framework.

### `d3q7`
Holds the seven‑velocity lattice used primarily for scalar transport or
simplified 3D flow.  The structure mirrors the D2Q9 folder:

- `bc.h` – boundary conditions and geometry tags.
- `col_clbm.h`, `col_clbm_RS.h`, `col_mrt.h`, `col_srt.h` – several collision
  models, including regularised and multi‑relaxation‑time variants.
- `common.h` – discrete velocity set and weights for D3Q7.
- `eq.h` – equilibrium distributions.
- `macro.h` – macroscopic field handling.
- `streaming_AA.h`, `streaming_AB.h` – streaming kernels for both memory
  patterns.

### `d3q27`
The primary 3D twenty‑seven‑velocity implementation.  This directory contains a
rich set of models:

- `bc.h` – boundary handling for fluid, walls, inflow/outflow and symmetry.
- `common.h` / `common_well.h` – lattice constants for the standard and
  "well‑balanced" variants.
- `macro.h` – macroscopic field copy/output and force computation hooks.
- `eq.h`, `eq_inv_cum.h`, `eq_well.h`, `eq_entropic.h` – various equilibrium
  formulations including cumulant and entropic models.
- Streaming – `streaming_AA.h` and `streaming_AB.h` implement the two memory
  layouts.
- Collision operators – numerous `col_*.h` files implement BGK, cumulant,
  central‑moment (CLBM), finite‑difference cumulant (FCLBM), multi‑relaxation
  time (MRT), Smagorinsky LES (`col_cum_sgs.h`), entropic KBC (`col_kbc_n.h`
  and `col_kbc_c.h`), force‑modified SRT, and well‑balanced variants for
  compressible flow.  These headers encapsulate the mathematical core of each
  collision scheme and expose a common interface used by the main kernel.

Together, the subdirectory provides building blocks for a wide range of 3D flow
simulations with different physical models and numerical stabilisations.

## Grand Scheme

The files described above form a modular LBM framework.  `LBM` objects own a
lattice and a set of `LBM_BLOCK`s that hold the actual data arrays.  The
simulation `State` orchestrates initialisation, the time‑stepping loop and all
input/output.  During each step the kernels from `kernels.h` read and update the
`LBM_Data` structures using collision operators and equilibrium models defined in
the appropriate `d*qn*` subdirectory.  Optional IBM support is provided by
`lagrange_3D.*`, `dirac.h` and `ibm_kernels.h`, while checkpointing and visual
outputs are handled by `adios_writer.*`, `vtk_writer.*` and `checkpoint.h`.

By decomposing responsibilities across these headers the project allows new
lattice models, collision operators or boundary conditions to be added simply by
contributing additional headers and wiring them into a configuration struct.
