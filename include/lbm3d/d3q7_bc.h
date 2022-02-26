#pragma once

template< typename T_LBM >
struct D3Q7_BC_All
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
			KS.phi = 0;
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
			COLL::computeDensityAndVelocity(KS);
			break;
		case GEO_WALL:
			// nema zadny vliv na vypocet, jen pro output
			KS.phi = 0;
			// collision step: bounce-back
			TNL::swap( KS.f[mzz], KS.f[pzz] );
			TNL::swap( KS.f[zmz], KS.f[zpz] );
			TNL::swap( KS.f[zzm], KS.f[zzp] );
			// anti-bounce-back (recovers zero gradient across the wall boundary, see Kruger section 8.5.2.1)
			for (int q = 0; q < 7; q++)
				KS.f[q] = -KS.f[q];
			// TODO: Kruger's eq (8.54) includes concentration imposed on the wall - does it diffusively propagate into the domain?
			break;
		default:
			COLL::computeDensityAndVelocity(KS);
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
