//#define AB_PATTERN

#include "core.h"
#include "d3q7_eq.h"
#include "d3q7_col_srt.h"
#include "d3q7_col_mrt.h"
#include "d3q7_col_clbm.h"
// exactly one streaming header must be included
#ifdef AA_PATTERN
	#include "d3q7_streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "d3q7_streaming_AB.h"
#endif
#include "d3q7_bc.h"
#include "d3q7_macro.h"

template < typename TRAITS >
struct NSE_Data_FreeRhoConstInflow : NSE_Data< TRAITS >
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	dreal inflow_vx = 0;
	dreal inflow_vy = 0;
	dreal inflow_vz = 0;

	template < typename LBM_KS >
	CUDA_HOSTDEV void inflow(LBM_KS &KS, idx x, idx y, idx z)
	{
		KS.vx  = inflow_vx;
		KS.vy  = inflow_vy;
		KS.vz  = inflow_vz;
	}
};

template< typename T_LBM >
struct D3Q27_BC_ADE
{
	using COLL = typename T_LBM::COLL;
	using STREAMING = typename T_LBM::STREAMING;
	using DATA = typename T_LBM::DATA;

	using map_t = typename T_LBM::TRAITS::map_t;
	using idx = typename T_LBM::TRAITS::idx;
	using dreal = typename T_LBM::TRAITS::dreal;

	enum GEO : map_t {
		GEO_FLUID, 		// compulsory
		GEO_WALL, 		// compulsory
		GEO_INFLOW,
		GEO_OUTFLOW_RIGHT,
		GEO_PERIODIC,
		GEO_NOTHING
	};

	CUDA_HOSTDEV static bool isPeriodic(map_t mapgi)
	{
		return (mapgi==GEO_PERIODIC);
	}

	CUDA_HOSTDEV static bool isFluid(map_t mapgi)
	{
		return (mapgi==GEO_FLUID);
	}

	template< typename LBM_KS >
	CUDA_HOSTDEV static void preCollision(DATA &SD, LBM_KS &KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING) {
			// nema zadny vliv na vypocet, jen pro output
			KS.rho = 0;
			return;
		}

		// modify pull location for streaming
		if (mapgi == GEO_OUTFLOW_RIGHT)
			xp = x = xm;

		STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

		// boundary conditions
		switch (mapgi)
		{
		case GEO_INFLOW:
			SD.inflow(KS,x,y,z);
			COLL::setEquilibrium(KS);
			break;
		case GEO_OUTFLOW_RIGHT:
//			COLL::computeDensityAndVelocity(KS);
			// compute only density (ADE!)
			KS.rho= ((((KS.f[ppp]+KS.f[mmm]) + (KS.f[pmp]+KS.f[mpm])) + ((KS.f[ppm]+KS.f[mmp])+(KS.f[mpp]+KS.f[pmm])))
				+(((KS.f[zpp]+KS.f[zmm]) + (KS.f[zpm]+KS.f[zmp])) + ((KS.f[pzp]+KS.f[mzm])+(KS.f[pzm]+KS.f[mzp])) + ((KS.f[ppz]+KS.f[mmz]) + (KS.f[pmz]+KS.f[mpz])))
				+((KS.f[pzz]+KS.f[mzz]) + (KS.f[zpz]+KS.f[zmz]) + (KS.f[zzp]+KS.f[zzm]))) + KS.f[zzz];
			break;
		case GEO_WALL:
			// nema zadny vliv na vypocet, jen pro output
			KS.rho = 0;
			// collision step: bounce-back
			TNL::swap( KS.f[mmm], KS.f[ppp] );
			TNL::swap( KS.f[mmz], KS.f[ppz] );
			TNL::swap( KS.f[mmp], KS.f[ppm] );
			TNL::swap( KS.f[mzm], KS.f[pzp] );
			TNL::swap( KS.f[mzz], KS.f[pzz] );
			TNL::swap( KS.f[mzp], KS.f[pzm] );
			TNL::swap( KS.f[mpm], KS.f[pmp] );
			TNL::swap( KS.f[mpz], KS.f[pmz] );
			TNL::swap( KS.f[mpp], KS.f[pmm] );
			TNL::swap( KS.f[zmm], KS.f[zpp] );
			TNL::swap( KS.f[zzm], KS.f[zzp] );
			TNL::swap( KS.f[zmz], KS.f[zpz] );
			TNL::swap( KS.f[zmp], KS.f[zpm] );
			// anti-bounce-back (recovers zero gradient across the wall boundary, see Kruger section 8.5.2.1)
			for (int q = 0; q < 27; q++)
				KS.f[q] = -KS.f[q];
			// TODO: Kruger's eq (8.54) includes concentration imposed on the wall - does it diffusively propagate into the domain?
			break;
		default:
//			COLL::computeDensityAndVelocity(KS);
			// compute only density (ADE!)
			KS.rho= ((((KS.f[ppp]+KS.f[mmm]) + (KS.f[pmp]+KS.f[mpm])) + ((KS.f[ppm]+KS.f[mmp])+(KS.f[mpp]+KS.f[pmm])))
				+(((KS.f[zpp]+KS.f[zmm]) + (KS.f[zpm]+KS.f[zmp])) + ((KS.f[pzp]+KS.f[mzm])+(KS.f[pzm]+KS.f[mzp])) + ((KS.f[ppz]+KS.f[mmz]) + (KS.f[pmz]+KS.f[mpz])))
				+((KS.f[pzz]+KS.f[mzz]) + (KS.f[zpz]+KS.f[zmz]) + (KS.f[zzp]+KS.f[zzm]))) + KS.f[zzz];
			break;
		}
	}

	CUDA_HOSTDEV static bool doCollision(map_t mapgi)
	{
		// by default, collision is done on non-BC sites only
		// additionally, BCs which include the collision step should be specified here
		return isFluid(mapgi) || isPeriodic(mapgi)
			|| mapgi == GEO_OUTFLOW_RIGHT;
	}

	template< typename LBM_KS >
	CUDA_HOSTDEV static void postCollision(DATA &SD, LBM_KS &KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING)
			return;

		STREAMING::postCollisionStreaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);
	}
};

#if 0
template < typename NSE >
#ifdef USE_CUDA
__global__ void cudaLBMComputeQCriterion(
	typename NSE::DATA SD,
	short int rank,
	short int nproc
)
#else
void LBMComputeQCriterion(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	short int rank,
	short int nproc
)
#endif
{
	using dreal = typename NSE::TRAITS::dreal;
	using idx = typename NSE::TRAITS::idx;
	using map_t = typename NSE::TRAITS::map_t;

	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
	map_t gi_map = SD.map(x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (NSE::BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
		// NOTE: ghost layers of lattice sites are assumed in the x-direction, so x+1 and x-1 always work
		xp = x+1;
		xm = x-1;
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

		struct Tensor
		{
			dreal xx=0,xy=0,xz=0;
			dreal yx=0,yy=0,yz=0;
			dreal zx=0,zy=0,zz=0;
		};
		Tensor G;

		// grad vel tensor
		if (y == 0 || z == 0 || y == SD.Y()-1 || z == SD.Z()-1)
		{
			// do nothing
			G.xx = 0;
			G.xy = 0;
			G.xz = 0;
			G.yx = 0;
			G.yy = 0;
			G.yz = 0;
			G.zx = 0;
			G.zy = 0;
			G.zz = 0;
		}
		else if (x == 0) {
			// forward difference for x
			G.xx = SD.macro(NSE::MACRO::e_vx,xp,y,z) - SD.macro(NSE::MACRO::e_vx,x,y,z);
			G.yx = SD.macro(NSE::MACRO::e_vy,xp,y,z) - SD.macro(NSE::MACRO::e_vy,x,y,z);
			G.zx = SD.macro(NSE::MACRO::e_vz,xp,y,z) - SD.macro(NSE::MACRO::e_vz,x,y,z);
			// central differences for y,z
			G.xy = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,yp,z) - SD.macro(NSE::MACRO::e_vx,x,ym,z));
			G.yy = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,yp,z) - SD.macro(NSE::MACRO::e_vy,x,ym,z));
			G.zy = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,yp,z) - SD.macro(NSE::MACRO::e_vz,x,ym,z));
			G.xz = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,y,zp) - SD.macro(NSE::MACRO::e_vx,x,y,zm));
			G.yz = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,y,zp) - SD.macro(NSE::MACRO::e_vy,x,y,zm));
			G.zz = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,y,zp) - SD.macro(NSE::MACRO::e_vz,x,y,zm));
		}
		else if (x == SD.X()-1) {
			// backward difference for x
			G.xx = SD.macro(NSE::MACRO::e_vx,x,y,z) - SD.macro(NSE::MACRO::e_vx,xm,y,z);
			G.yx = SD.macro(NSE::MACRO::e_vy,x,y,z) - SD.macro(NSE::MACRO::e_vy,xm,y,z);
			G.zx = SD.macro(NSE::MACRO::e_vz,x,y,z) - SD.macro(NSE::MACRO::e_vz,xm,y,z);
			// central differences for y,z
			G.xy = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,yp,z) - SD.macro(NSE::MACRO::e_vx,x,ym,z));
			G.yy = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,yp,z) - SD.macro(NSE::MACRO::e_vy,x,ym,z));
			G.zy = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,yp,z) - SD.macro(NSE::MACRO::e_vz,x,ym,z));
			G.xz = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,y,zp) - SD.macro(NSE::MACRO::e_vx,x,y,zm));
			G.yz = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,y,zp) - SD.macro(NSE::MACRO::e_vy,x,y,zm));
			G.zz = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,y,zp) - SD.macro(NSE::MACRO::e_vz,x,y,zm));
		}
		else {
			// central differences
			G.xx = n1o2 * (SD.macro(NSE::MACRO::e_vx,xp,y,z) - SD.macro(NSE::MACRO::e_vx,xm,y,z));
			G.xy = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,yp,z) - SD.macro(NSE::MACRO::e_vx,x,ym,z));
			G.xz = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,y,zp) - SD.macro(NSE::MACRO::e_vx,x,y,zm));
			G.yx = n1o2 * (SD.macro(NSE::MACRO::e_vy,xp,y,z) - SD.macro(NSE::MACRO::e_vy,xm,y,z));
			G.yy = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,yp,z) - SD.macro(NSE::MACRO::e_vy,x,ym,z));
			G.yz = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,y,zp) - SD.macro(NSE::MACRO::e_vy,x,y,zm));
			G.zx = n1o2 * (SD.macro(NSE::MACRO::e_vz,xp,y,z) - SD.macro(NSE::MACRO::e_vz,xm,y,z));
			G.zy = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,yp,z) - SD.macro(NSE::MACRO::e_vz,x,ym,z));
			G.zz = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,y,zp) - SD.macro(NSE::MACRO::e_vz,x,y,zm));
		}

		// q criterion from definition: Q = - sum_ij d_i u_j d_j u_i
		const dreal q = G.xx*G.yy + G.yy*G.zz + G.xx*G.zz - G.zx*G.xz - G.yz*G.zy - G.xy*G.yx;
		SD.macro(NSE::MACRO::e_qcrit,x,y,z) = q;
}

template < typename ADE >
#ifdef USE_CUDA
__global__ void cudaLBMComputePhiGradMag(
	typename ADE::DATA SD,
	short int rank,
	short int nproc
)
#else
void cudaLBMComputePhiGradMag(
	typename ADE::DATA SD,
	typename ADE::TRAITS::idx x,
	typename ADE::TRAITS::idx y,
	typename ADE::TRAITS::idx z,
	short int rank,
	short int nproc
)
#endif
{
	using dreal = typename ADE::TRAITS::dreal;
	using idx = typename ADE::TRAITS::idx;
	using map_t = typename ADE::TRAITS::map_t;

	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
	map_t gi_map = SD.map(x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (ADE::BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
		// NOTE: ghost layers of lattice sites are assumed in the x-direction, so x+1 and x-1 always work
		xp = x+1;
		xm = x-1;
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

		struct Vector
		{
			dreal x=0;
			dreal y=0;
			dreal z=0;
		};
		Vector G;

		// grad phi vector
		if (y == 0 || z == 0 || y == SD.Y()-1 || z == SD.Z()-1)
		{
			// do nothing
			G.x = 0;
			G.y = 0;
			G.z = 0;
		}
		else if (x == 0) {
			// forward difference for x
			G.x = SD.macro(ADE::MACRO::e_phi,xp,y,z) - SD.macro(ADE::MACRO::e_phi,x,y,z);
			G.y = 0;
			G.z = 0;
		}
		else if (x == SD.X()-1) {
			// backward difference for x
			G.x = SD.macro(ADE::MACRO::e_phi,x,y,z) - SD.macro(ADE::MACRO::e_phi,xm,y,z);
			G.y = 0;
			G.z = 0;
		}
		else {
			// central differences
			G.x = n1o2 * (SD.macro(ADE::MACRO::e_phi,xp,y,z) - SD.macro(ADE::MACRO::e_phi,xm,y,z));
			G.y = n1o2 * (SD.macro(ADE::MACRO::e_phi,x,yp,z) - SD.macro(ADE::MACRO::e_phi,x,ym,z));
			G.z = n1o2 * (SD.macro(ADE::MACRO::e_phi,x,y,zp) - SD.macro(ADE::MACRO::e_phi,x,y,zm));
		}

		SD.macro(ADE::MACRO::e_phigradmag2,x,y,z) = G.x*G.x + G.y*G.y + G.z*G.z;
}

template < typename TRAITS >
struct D3Q27_MACRO_QCriterion : D3Q27_MACRO_Base< TRAITS >
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum { e_rho, e_vx, e_vy, e_vz, e_fx, e_fy, e_fz, e_qcrit, N};

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void outputMacro(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z)  = KS.vx;
		SD.macro(e_vy, x, y, z)  = KS.vy;
		SD.macro(e_vz, x, y, z)  = KS.vz;
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.fx;
		KS.fy = SD.fy;
		KS.fz = SD.fz;
	}
};
#endif

// 3D test domain
template < typename NSE, typename ADE >
struct StateLocal : State<NSE>
{
	// using different TRAITS is not implemented (probably does not make sense...)
	static_assert(std::is_same<typename NSE::TRAITS, typename ADE::TRAITS>::value,
			"TRAITS must be the same type in NSE and ADE.");
	using TRAITS = typename NSE::TRAITS;
	using BLOCK_NSE = LBM_BLOCK< NSE >;
	using BLOCK_ADE = LBM_BLOCK< ADE >;

	using State<NSE>::nse;
	using State<NSE>::cnt;
	using State<NSE>::vtk_helper;
	using State<NSE>::log;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	LBM<ADE> ade;

	real lbmInflowDensity = no1;

	// constructor
	StateLocal(const TNL::MPI::Comm& communicator, lat_t ilat, real iphysViscosity, real iphysVelocity, real iphysDt, real iphysDiffusion)
		: State<NSE>(communicator, ilat, iphysViscosity, iphysDt),
		  ade(communicator, ilat, iphysDiffusion, iphysDt)
	{
		for (auto& block : nse.blocks)
		{
//			block.data.inflow_rho = no1;
			block.data.inflow_vx = nse.phys2lbmVelocity(iphysVelocity);
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}

		// ADE allocation
		ade.allocateHostData();

		for (auto& block : ade.blocks)
		{
			// TODO: phys -> lbm conversion for concentration?
			block.data.inflow_phi = 1e-3;
		}
	}

	// TODO: override estimateMemoryDemands

	void reset() override
	{
		nse.resetMap(NSE::BC::GEO_FLUID);
		ade.resetMap(ADE::BC::GEO_FLUID);
		setupBoundaries();
		nse.projectWall();
		ade.projectWall();

		// reset lattice for NSE and ADE
		// NOTE: it is important to reset *all* lattice sites (i.e. including ghost layers) when using the A-A pattern
		// (because GEO_INFLOW and GEO_OUTFLOW_EQ access the ghost layer in streaming)
		nse.forAllLatticeSites( [&] (BLOCK_NSE& block, idx x, idx y, idx z) {
			block.setEqLat(x,y,z,1,0,0,0);//rho,vx,vy,vz);
		} );
		ade.forAllLatticeSites( [&] (BLOCK_ADE& block, idx x, idx y, idx z) {
			block.setEqLat(x,y,z,0,0,0,0);//phi,vx,vy,vz);
		} );

		//initial time of current simulation
		clock_gettime(CLOCK_REALTIME, &this->t_init);
	}

	void setupBoundaries() override
	{
		nse.setBoundaryX(0, NSE::BC::GEO_INFLOW); 		// left
		nse.setBoundaryX(nse.lat.global.x()-1, NSE::BC::GEO_OUTFLOW_EQ);
//		nse.setBoundaryX(nse.lat.global.x()-1, NSE::BC::GEO_OUTFLOW_RIGHT);

		nse.setBoundaryZ(1, NSE::BC::GEO_WALL);		// top
		nse.setBoundaryZ(nse.lat.global.z()-2, NSE::BC::GEO_WALL);	// bottom
		nse.setBoundaryY(1, NSE::BC::GEO_WALL); 		// back
		nse.setBoundaryY(nse.lat.global.y()-2, NSE::BC::GEO_WALL);		// front

		// extra layer needed due to A-A pattern
		nse.setBoundaryZ(0, NSE::BC::GEO_NOTHING);		// top
		nse.setBoundaryZ(nse.lat.global.z()-1, NSE::BC::GEO_NOTHING);	// bottom
		nse.setBoundaryY(0, NSE::BC::GEO_NOTHING); 		// back
		nse.setBoundaryY(nse.lat.global.y()-1, NSE::BC::GEO_NOTHING);		// front

		// ADE boundaries
		ade.setBoundaryX(0, ADE::BC::GEO_INFLOW); 		// left
		ade.setBoundaryX(ade.lat.global.x()-1, ADE::BC::GEO_OUTFLOW_RIGHT);

		ade.setBoundaryZ(1, ADE::BC::GEO_WALL);		// top
		ade.setBoundaryZ(ade.lat.global.z()-2, ADE::BC::GEO_WALL);	// bottom
		ade.setBoundaryY(1, ADE::BC::GEO_WALL); 		// back
		ade.setBoundaryY(ade.lat.global.y()-2, ADE::BC::GEO_WALL);		// front

		// extra layer needed due to A-A pattern
		ade.setBoundaryZ(0, ADE::BC::GEO_NOTHING);		// top
		ade.setBoundaryZ(ade.lat.global.z()-1, ADE::BC::GEO_NOTHING);	// bottom
		ade.setBoundaryY(0, ADE::BC::GEO_NOTHING); 		// back
		ade.setBoundaryY(ade.lat.global.y()-1, ADE::BC::GEO_NOTHING);		// front
	}

	void SimInit() override
	{
		log("MPI info: rank=%d, nproc=%d, lat.global=[%d,%d,%d]", nse.rank, nse.nproc, nse.lat.global.x(), nse.lat.global.y(), nse.lat.global.z());
		for (auto& block : nse.blocks)
			log("LBM block %d: local=[%d,%d,%d], offset=[%d,%d,%d]", block.id, block.local.x(), block.local.y(), block.local.z(), block.offset.x(), block.offset.y(), block.offset.z());

		log("\nSTART: simulation NSE:%s-ADE:%s lbmViscosity %e lbmDiffusion %e physDl %e physDt %e", NSE::COLL::id, ADE::COLL::id, nse.lbmViscosity(), ade.lbmViscosity(), nse.lat.physDl, nse.physDt);

		// reset counters
		for (int c=0;c<MAX_COUNTER;c++) cnt[c].count = 0;
		cnt[SAVESTATE].count = 1;  // skip initial save of state
		nse.iterations = nse.prevIterations = 0;

		// setup map and DFs in CPU memory
		reset();

		// init NSE
		for (auto& block : nse.blocks)
		{
			// create LBM_DATA with host pointers
			typename NSE::DATA SD;
			for (uint8_t dfty=0;dfty<DFMAX;dfty++)
				SD.dfs[dfty] = block.hfs[dfty].getData();
			#ifdef HAVE_MPI
			SD.indexer = block.hmap.getLocalView().getIndexer();
			#else
			SD.indexer = block.hmap.getIndexer();
			#endif
			SD.XYZ = SD.indexer.getStorageSize();
			SD.dmap = block.hmap.getData();
			SD.dmacro = block.hmacro.getData();

			// initialize macroscopic quantities on CPU
			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				LBMKernelInit<NSE>(SD, x, y, z);
		}

		// init ADE
		for (auto& block : ade.blocks)
		{
			// create LBM_DATA with host pointers
			typename ADE::DATA SD;
			for (uint8_t dfty=0;dfty<DFMAX;dfty++)
				SD.dfs[dfty] = block.hfs[dfty].getData();
			#ifdef HAVE_MPI
			SD.indexer = block.hmap.getLocalView().getIndexer();
			#else
			SD.indexer = block.hmap.getIndexer();
			#endif
			SD.XYZ = SD.indexer.getStorageSize();
			SD.dmap = block.hmap.getData();
			SD.dmacro = block.hmacro.getData();

			// initialize macroscopic quantities on CPU
			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				LBMKernelInit<ADE>(SD, x, y, z);
		}

		nse.allocateDeviceData();
		ade.allocateDeviceData();
		this->copyAllToDevice();

#ifdef HAVE_MPI
		// synchronize overlaps with MPI (initial synchronization can be synchronous)
		nse.synchronizeMapDevice();
		nse.synchronizeDFsDevice(df_cur);
		if (NSE::MACRO::use_syncMacro)
			nse.synchronizeMacroDevice();

		ade.synchronizeMapDevice();
		ade.synchronizeDFsDevice(df_cur);
		if (ADE::MACRO::use_syncMacro)
			ade.synchronizeMacroDevice();
#endif
	}

	void updateKernelData() override
	{
		// general update (even_iter, dfs pointer)
		nse.updateKernelData();
		ade.updateKernelData();

		// update LBM viscosity/diffusivity
		for( auto& block : nse.blocks )
			block.data.lbmViscosity = nse.lbmViscosity();
		for( auto& block : ade.blocks )
			block.data.lbmViscosity = ade.lbmViscosity();
	}

	void updateKernelVelocities() override
	{
//		for (auto& block : nse.blocks)
//			block.data.inflow_rho = lbmInflowDensity;
	}

	virtual void statReset() override
	{
	}

#if 0
	void computeBeforeLBMKernel() override
	{
		#ifdef USE_CUDA
		auto get_grid_size = [] (const auto& block) -> dim3
		{
			dim3 gridSize(block.local.x(), block.local.y()/block.block_size.y, block.local.z());

			// check for PEBKAC problem existing between keyboard and chair
			if (gridSize.y * block.block_size.y != block.local.y())
				throw std::logic_error("error: block.local.y() (which is " + std::to_string(block.local.y()) + ") "
									   "is not aligned to a multiple of the block size (which is " + std::to_string(block.block_size.y) + ")");

			return gridSize;
		};
		#endif

		for (auto& block : nse.blocks)
		{
		#ifdef USE_CUDA
			const dim3 gridSize = get_grid_size(block);
			cudaLBMComputeQCriterion< NSE ><<<gridSize, block.block_size>>>(block.data, nse.rank, nse.nproc);
			cudaStreamSynchronize(0);
			checkCudaDevice;
		#else
			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				LBMComputeQCriterion< NSE >(block.data, nse.rank, nse.nproc, x, y, z);
		#endif
		}

		for (auto& block : ade.blocks)
		{
		#ifdef USE_CUDA
			const dim3 gridSize = get_grid_size(block);
			cudaLBMComputePhiGradMag< ADE ><<<gridSize, block.block_size>>>(block.data, nse.rank, nse.nproc);
			cudaStreamSynchronize(0);
			checkCudaDevice;
		#else
			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				cudaLBMComputePhiGradMag< ADE >(block.data, nse.rank, nse.nproc, x, y, z);
		#endif
		}
	}
#endif

	void SimUpdate() override
	{
		// debug
		for (auto& block : nse.blocks)
		if (block.data.lbmViscosity == 0) {
			log("error: NSE viscosity is 0");
			nse.terminate = true;
			return;
		}
		for (auto& block : ade.blocks)
		if (block.data.lbmViscosity == 0) {
			log("error: ADE diffusion is 0");
			nse.terminate = true;
			return;
		}

		#ifdef USE_CUDA
		auto get_grid_size = [] (const auto& block) -> dim3
		{
			dim3 gridSize(block.local.x(), block.local.y()/block.block_size.y, block.local.z());

			// check for PEBKAC problem existing between keyboard and chair
			if (gridSize.y * block.block_size.y != block.local.y())
				throw std::logic_error("error: block.local.y() (which is " + std::to_string(block.local.y()) + ") "
									   "is not aligned to a multiple of the block size (which is " + std::to_string(block.block_size.y) + ")");

			return gridSize;
		};
		#endif


		// call hook method (used e.g. for extra kernels in the non-Newtonian model)
		this->computeBeforeLBMKernel();


		#ifdef AA_PATTERN
		uint8_t output_df = df_cur;
		#endif
		#ifdef AB_PATTERN
		uint8_t output_df = df_out;
		#endif

		if (nse.blocks.size() != ade.blocks.size())
			throw std::logic_error("vectors of nse.blocks and ade.blocks must have equal sizes");

#ifdef USE_CUDA
		#ifdef HAVE_MPI
		if (nse.nproc == 1)
		{
		#endif
			for (std::size_t b = 0; b < nse.blocks.size(); b++)
			{
				auto& block_nse = nse.blocks[b];
				auto& block_ade = ade.blocks[b];
				// TODO: check that block_nse and block_ade have the same sizes

				const dim3 blockSize = block_nse.block_size;
				const dim3 gridSize = get_grid_size(block_nse);
				cudaLBMKernel< NSE, ADE ><<<gridSize, blockSize>>>(block_nse.data, block_ade.data, nse.rank, nse.nproc, (idx) 0);
			}
			cudaDeviceSynchronize();
			checkCudaDevice;
			// copying of overlaps is not necessary for nproc == 1 (nproc is checked in streaming as well)
		#ifdef HAVE_MPI
		}
		else
		{
			for (std::size_t b = 0; b < nse.blocks.size(); b++)
			{
				auto& block_nse = nse.blocks[b];
				auto& block_ade = ade.blocks[b];
				// TODO: check that block_nse and block_ade have the same sizes

				const dim3 blockSize = block_nse.block_size;
				const dim3 gridSizeForBoundary(block_nse.df_overlap_X(), block_nse.local.y()/block_nse.block_size.y, block_nse.local.z());
				const dim3 gridSizeForInternal(block_nse.local.x() - 2*block_nse.df_overlap_X(), block_nse.local.y()/block_nse.block_size.y, block_nse.local.z());

				// compute on boundaries (NOTE: 1D distribution is assumed)
				cudaLBMKernel< NSE, ADE ><<<gridSizeForBoundary, blockSize, 0, cuda_streams[0]>>>(block_nse.data, block_ade.data, block_nse.id, nse.total_blocks, (idx) 0);
				cudaLBMKernel< NSE, ADE ><<<gridSizeForBoundary, blockSize, 0, cuda_streams[1]>>>(block_nse.data, block_ade.data, block_nse.id, nse.total_blocks, block_nse.local.x() - block_nse.df_overlap_X());

				// compute on internal lattice sites
				cudaLBMKernel< NSE, ADE ><<<gridSizeForInternal, blockSize, 0, cuda_streams[2]>>>(block_nse.data, block_ade.data, block_nse.id, nse.total_blocks, block_nse.df_overlap_X());
			}

			// wait for the computations on boundaries to finish
			cudaStreamSynchronize(cuda_streams[0]);
			cudaStreamSynchronize(cuda_streams[1]);

			// start communication of the latest DFs and dmacro on overlaps
			nse.synchronizeDFsDevice_start(output_df);
			ade.synchronizeDFsDevice_start(output_df);
			if (NSE::MACRO::use_syncMacro)
				nse.synchronizeMacroDevice_start();
			if (ADE::MACRO::use_syncMacro)
				ade.synchronizeMacroDevice_start();

			// wait for the communication to finish
			// (it is important to do this before waiting for the computation, otherwise MPI won't progress)
			for (auto& block : nse.blocks)
				block.waitAllCommunication();
			for (auto& block : ade.blocks)
				block.waitAllCommunication();

			// wait for the computation on the interior to finish
			cudaStreamSynchronize(cuda_streams[2]);

			// synchronize the whole GPU and check errors
			cudaDeviceSynchronize();
			checkCudaDevice;
		}
		#endif
#else
		for (std::size_t b = 0; b < nse.blocks.size(); b++)
		{
			auto& block_nse = nse.blocks[b];
			auto& block_ade = ade.blocks[b];
			// TODO: check that block_nse and block_ade have the same sizes

//			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x=0; x<block_nse.local.x(); x++)
			for (idx z=0; z<block_nse.local.z(); z++)
			for (idx y=0; y<block_nse.local.y(); y++)
			{
				LBMKernel< NSE, ADE >(block_nse.data, block_ade.data, x, y, z, nse.rank, nse.nproc);
			}
		}
		#ifdef HAVE_MPI
		// TODO: overlap computation with synchronization, just like above
		nse.synchronizeDFsDevice(output_df);
		ade.synchronizeDFsDevice(output_df);
		#endif
#endif

		nse.iterations++;
		ade.iterations = nse.iterations;

		bool doit=false;
		for (int c=0;c<MAX_COUNTER;c++) if (c!=PRINT && c!=SAVESTATE) if (cnt[c].action(nse.physTime())) doit = true;
		if (doit)
		{
			// common copy
			nse.copyMacroToHost();
			ade.copyMacroToHost();
			// to be able to compute rho, vx, vy, vz etc... based on DFs on CPU to save GPU memory FIXME may not work with ESOTWIST
			if (NSE::CPU_MACRO::N>0)
				nse.copyDFsToHost(output_df);
			if (ADE::CPU_MACRO::N>0)
				ade.copyDFsToHost(output_df);
			#ifdef USE_CUDA
			checkCudaDevice;
			#endif
		}
	}

	void AfterSimUpdate(timespec& t1, timespec& t2) override
	{
		State< NSE >::AfterSimUpdate(t1, t2);
		// TODO: figure out what should be done for ade here...
	}

	// called from SimInit -- copy the initial state to the GPU
	void copyAllToDevice() override
	{
		nse.copyMapToDevice();
		nse.copyDFsToDevice();
		nse.copyMacroToDevice();
		ade.copyMapToDevice();
		ade.copyDFsToDevice();
		ade.copyMacroToDevice();
	}

	// called from core.h -- inside the time loop before saving state
	void copyAllToHost() override
	{
		nse.copyMapToHost();
		nse.copyDFsToHost();
		nse.copyMacroToHost();
		ade.copyMapToHost();
		ade.copyDFsToHost();
		ade.copyMacroToHost();
	}

	void writeVTKs_3D() override
	{
		char dir[FILENAME_CHARS], filename[FILENAME_CHARS], basename[FILENAME_CHARS];
		sprintf(dir,"results_%s/vtk3D", this->id);
		mkdir_p(dir,0755);

		for (const auto& block : nse.blocks)
		{
			sprintf(basename,"NSE_block%03d_%d.vtk", block.id, cnt[VTK3D].count);
			sprintf(filename,"%s/%s", dir, basename);
			auto outputData = [this] (const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
			{
				return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
			};
			block.writeVTK_3D(nse.lat, outputData, filename, nse.physTime(), cnt[VTK3D].count);
			this->log("[vtk %s written, time %f, cycle %d] ", filename, nse.physTime(), cnt[VTK3D].count);
		}
		for (const auto& block : ade.blocks)
		{
			sprintf(basename,"ADE_block%03d_%d.vtk", block.id, cnt[VTK3D].count);
			sprintf(filename,"%s/%s", dir, basename);
			auto outputData = [this] (const BLOCK_ADE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
			{
				return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
			};
			block.writeVTK_3D(ade.lat, outputData, filename, nse.physTime(), cnt[VTK3D].count);
			this->log("[vtk %s written, time %f, cycle %d] ", filename, nse.physTime(), cnt[VTK3D].count);
		}
	}

	void writeVTKs_3Dcut() override
	{
		if (this->probe3Dvec.size()<=0) return;
		// browse all 3D vtk cuts
		for (auto& probevec : this->probe3Dvec)
		{
			for (const auto& block : nse.blocks)
			{
				char fname[FILENAME_CHARS];
				sprintf(fname,"results_%s/vtk3Dcut/%s_NSE_block%03d_%d.vtk", this->id, probevec.name, block.id, probevec.cycle);
				// create parent directories
				create_file(fname);
				auto outputData = [this] (const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
				{
					return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
				};
				block.writeVTK_3Dcut(
					nse.lat,
					outputData,
					fname,
					nse.physTime(),
					probevec.cycle,
					probevec.ox,
					probevec.oy,
					probevec.oz,
					probevec.lx,
					probevec.ly,
					probevec.lz,
					probevec.step
				);
				this->log("[vtk %s written, time %f, cycle %d] ", fname, nse.physTime(), probevec.cycle);
			}
			for (const auto& block : ade.blocks)
			{
				char fname[FILENAME_CHARS];
				sprintf(fname,"results_%s/vtk3Dcut/%s_ADE_block%03d_%d.vtk", this->id, probevec.name, block.id, probevec.cycle);
				// create parent directories
				create_file(fname);
				auto outputData = [this] (const BLOCK_ADE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
				{
					return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
				};
				block.writeVTK_3Dcut(
					ade.lat,
					outputData,
					fname,
					nse.physTime(),
					probevec.cycle,
					probevec.ox,
					probevec.oy,
					probevec.oz,
					probevec.lx,
					probevec.ly,
					probevec.lz,
					probevec.step
				);
				this->log("[vtk %s written, time %f, cycle %d] ", fname, nse.physTime(), probevec.cycle);
			}
			probevec.cycle++;
		}
	}

	void writeVTKs_2D() override
	{
		if (this->probe2Dvec.size()<=0) return;
		// browse all 2D vtk cuts
		for (auto& probevec : this->probe2Dvec)
		{
			for (const auto& block : nse.blocks)
			{
				char fname[FILENAME_CHARS];
				sprintf(fname,"results_%s/vtk2D/%s_NSE_block%03d_%d.vtk", this->id, probevec.name, block.id, probevec.cycle);
				// create parent directories
				create_file(fname);
				auto outputData = [this] (const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
				{
					return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
				};
				switch (probevec.type)
				{
					case 0: block.writeVTK_2DcutX(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
					case 1: block.writeVTK_2DcutY(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
					case 2: block.writeVTK_2DcutZ(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
				}
				log("[vtk %s written, time %f, cycle %d] ", fname, nse.physTime(), probevec.cycle);
			}
			for (const auto& block : ade.blocks)
			{
				char fname[FILENAME_CHARS];
				sprintf(fname,"results_%s/vtk2D/%s_ADE_block%03d_%d.vtk", this->id, probevec.name, block.id, probevec.cycle);
				// create parent directories
				create_file(fname);
				auto outputData = [this] (const BLOCK_ADE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
				{
					return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
				};
				switch (probevec.type)
				{
					case 0: block.writeVTK_2DcutX(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
					case 1: block.writeVTK_2DcutY(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
					case 2: block.writeVTK_2DcutZ(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
				}
				log("[vtk %s written, time %f, cycle %d] ", fname, nse.physTime(), probevec.cycle);
			}
			probevec.cycle++;
		}
	}

	bool outputData(const BLOCK_NSE& block, int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs) override
	{
		int k=0;
		if (index==k++) return vtk_helper("lbm_density", block.hmacro(NSE::MACRO::e_rho,x,y,z), 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(NSE::MACRO::e_vx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(NSE::MACRO::e_vy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(NSE::MACRO::e_vz,x,y,z)), 3, desc, value, dofs);
			}
		}
//		if (index==k++) return vtk_helper("lbm_qcriterion", block.hmacro(NSE::MACRO::e_qcrit,x,y,z), 1, desc, value, dofs);
		return false;
	}

	virtual bool outputData(const BLOCK_ADE& block, int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs)
	{
		int k=0;
		if (index==k++) return vtk_helper("lbm_phi", block.hmacro(ADE::MACRO::e_phi,x,y,z), 1, desc, value, dofs);
//		if (index==k++) return vtk_helper("lbm_phigradmag2", block.hmacro(ADE::MACRO::e_phigradmag2,x,y,z), 1, desc, value, dofs);
		return false;
	}

	void probe1() override
	{
		if (nse.iterations != 0)
		{
			// inflow density extrapolation
			idx x = 5;
			idx y = nse.lat.global.y()/2;
			idx z = nse.lat.global.z()/2;
			for (auto& block : nse.blocks)
			if (block.isLocalIndex(x, y, z))
			{
				real oldlbmInflowDensity = lbmInflowDensity;
				lbmInflowDensity = block.dmacro.getElement(NSE::MACRO::e_rho, x, y, z);
				log("[probe: lbm inflow density changed from %e to %e", oldlbmInflowDensity, lbmInflowDensity);
			}
		}
	}
};

template < typename NSE, typename ADE >
int simT1_test(int RESOLUTION = 2)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size=32;
	int X = 128*RESOLUTION;// width in pixels --- product of 128.
	//	int Y = 41*RESOLUTION;// height in pixels --- top and bottom walls 1px
	//	int Z = 41*RESOLUTION;// height in pixels --- top and bottom walls 1px
	int Y = block_size*RESOLUTION;// height in pixels --- top and bottom walls 1px
	int Z = Y;// height in pixels --- top and bottom walls 1px
	real LBM_VISCOSITY = 0.001/3.0;//1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_HEIGHT = 0.41; // [m] domain height (physical)
	real PHYS_VISCOSITY = 1.552e-5; // [m^2/s] fluid viscosity of air
	real PHYS_VELOCITY = 1.0;
	real PHYS_DL = PHYS_HEIGHT/((real)Y-2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY*PHYS_DL*PHYS_DL;//PHYS_HEIGHT/(real)LBM_HEIGHT;
	real PHYS_DIFFUSION = 2.552e-05; // [m^2/s] diffusion coeff for the ADE
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType( X, Y, Z );
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;

	StateLocal< NSE, ADE > state(MPI_COMM_WORLD, lat, PHYS_VISCOSITY, PHYS_VELOCITY, PHYS_DT, PHYS_DIFFUSION);
	state.setid("sim_T1_res%02d_np%03d", RESOLUTION, state.nse.nproc);
#ifdef USE_CUDA
	for (auto& block : state.nse.blocks)
		block.block_size.y = block_size;
	for (auto& block : state.ade.blocks)
		block.block_size.y = block_size;
#endif
	state.nse.physCharLength = 0.1; // [m]
//	state.printIter = 100;
//	state.printIter = 100;
	state.nse.physFinalTime = 10.0;
	state.cnt[PRINT].period = 0.01;
//	state.cnt[PROBE1].period = 0.001;
	// test
//	state.cnt[PRINT].period = 100*PHYS_DT;
//	state.nse.physFinalTime = 1000*PHYS_DT;
//	state.cnt[VTK3D].period = 1000*PHYS_DT;
//	state.cnt[SAVESTATE].period = 600;  // save state every [period] of wall time
//	state.check_savestate_flag = false;
//	state.wallTime = 60;
	// RCI
//	state.nse.physFinalTime = 0.5;
//	state.cnt[VTK3D].period = 0.5;
//	state.cnt[SAVESTATE].period = 3600;  // save state every [period] of wall time
//	state.check_savestate_flag = false;
//	state.wallTime = 3600 * 23.5;

	// add cuts
	state.cnt[VTK2D].period = 0.01;
	state.add2Dcut_X(X/2,"cutsX/cut_X");
	state.add2Dcut_Y(Y/2,"cutsY/cut_Y");
	state.add2Dcut_Z(Z/2,"cutsZ/cut_Z");

//	state.cnt[VTK3D].period = 0.001;
//	state.cnt[VTK3DCUT].period = 0.001;
//	state.add3Dcut(X/4,Y/4,Z/4, X/2,Y/2,Z/2, 2, "box");

	// draw a sphere
	if (1)
	{
		int cy=floor(0.2/PHYS_DL);
		int cz=floor(0.2/PHYS_DL);
		int cx=floor(0.45/PHYS_DL);
		real radius=0.05; // 10 cm diameter
		int range=ceil(radius/PHYS_DL)+1;
		for (int py=cy-range;py<=cy+range;py++)
		for (int pz=cz-range;pz<=cz+range;pz++)
		for (int px=cx-range;px<=cx+range;px++)
//			if (NORM( (real)(px-cx)*PHYS_DL, (real)(py-cy)*PHYS_DL, (real)(pz-cz)*PHYS_DL) < radius )
			if ((real)(px-cx)*PHYS_DL < radius && (real)(py-cy)*PHYS_DL < radius && (real)(pz-cz)*PHYS_DL < radius )
			{
				state.nse.defineWall(px,py,pz,true);
				state.ade.defineWall(px,py,pz,true);
			}
	}

	// draw a cylinder
	if (0)
	{
		int cy=floor(0.2/PHYS_DL);
		int cz=floor(0.2/PHYS_DL);
		int cx=floor(0.45/PHYS_DL);
		real radius=0.05; // 10 cm diameter
		int range=ceil(radius/PHYS_DL)+1;
		//		for (int py=cy-range;py<=cy+range;py++)
		for (int pz=cz-range;pz<=cz+range;pz++)
		for (int px=cx-range;px<=cx+range;px++)
		for (int py=0;py<=Y-1;py++)
			if (NORM( (real)(px-cx)*PHYS_DL,0, (real)(pz-cz)*PHYS_DL) < radius )
			{
				state.nse.defineWall(px,py,pz,true);
				state.ade.defineWall(px,py,pz,true);
			}
	}

	// draw a block
	if (0)
	{
		//		int cy=floor(0.2/PHYS_DL);
		int cz=floor(0.20/PHYS_DL);
		int cx=floor(0.20/PHYS_DL);
		//		int range=Z/4;
		int width=Z/10;
		//		for (int py=cy-range;py<=cy+range;py++)
		//		for (int pz=0;pz<=cz;pz++)
		for (int px=cx;px<=cx+width;px++)
		for (int pz=1;pz<=Z-2;pz++)
		for (int py=1;py<=Y-2;py++)
			if (!((pz>=Z*4/10 &&  pz<=Z*6/10) && (py>=Y*4/10 && py<=Y*6/10)))
			{
				state.nse.defineWall(px,py,pz,true);
				state.ade.defineWall(px,py,pz,true);
			}
	}

	execute(state);

	return 0;
}

//template < typename TRAITS=TraitsSP >
template < typename TRAITS=TraitsDP >
void run(int RES)
{
	using NSE_COLL = D3Q27_CUM< TRAITS, D3Q27_EQ_INV_CUM<TRAITS> >;
	using NSE_TYPE = D3Q27<
				NSE_COLL,
//				NSE_Data_ConstInflow< TRAITS >,
				// FIXME: FreeRho inflow condition leads to lower velocity in the domain (approx 70%)
				NSE_Data_FreeRhoConstInflow< TRAITS >,
				D3Q27_BC_All,
				typename NSE_COLL::EQ,
				D3Q27_STREAMING< TRAITS >,
				D3Q27_MACRO_Default< TRAITS >,
//				D3Q27_MACRO_QCriterion< TRAITS >,
				D3Q27_MACRO_Void< TRAITS >,
				TRAITS
			>;

//	using ADE_COLL = D3Q7_SRT< TRAITS >;
//	using ADE_COLL = D3Q7_MRT< TRAITS >;
	using ADE_COLL = D3Q7_CLBM< TRAITS >;
	using ADE_TYPE = D3Q7<
				ADE_COLL,
				ADE_Data_ConstInflow< TRAITS >,
				D3Q7_BC_All,
				typename ADE_COLL::EQ,
				D3Q7_STREAMING< TRAITS >,
				D3Q7_MACRO_Default< TRAITS >,
				D3Q7_MACRO_Void< TRAITS >,
				TRAITS
			>;

//	using ADE_COLL = D3Q27_SRT< TRAITS >;
////	using ADE_COLL = D3Q27_CLBM< TRAITS >;  // TODO: blbost, operátor pro NSE nelze použít jen tak pro ADE (jiný počet zachovávajících se veličin)
//	using ADE_TYPE = D3Q27<
//				ADE_COLL,
//				ADE_Data_ConstInflow< TRAITS >,
//				D3Q27_BC_ADE,
//				typename ADE_COLL::EQ,
//				D3Q27_STREAMING< TRAITS >,
//				D3Q27_MACRO_Default< TRAITS >,
//				D3Q27_MACRO_Void< TRAITS >,
//				TRAITS
//			>;

	simT1_test< NSE_TYPE, ADE_TYPE >(RES);
}

int main(int argc, char **argv)
{
	TNLMPI_INIT mpi(argc, argv);

	const int pars=1;
	if (argc <= pars)
	{
		printf("error: required %d parameters:\n %s res[1,...]\n", pars, argv[0]);
		return 1;
	}
	int res = atoi(argv[1]);
	if (res < 1) { printf("error: res=%d out of bounds [1, ...]\n",res); return 1; }

	run(res);

	return 0;
}
