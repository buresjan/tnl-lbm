#pragma once

#include "lbm3d/defs.h"

template <typename CONFIG>
struct D2Q9_BC_All
{
	using COLL = typename CONFIG::COLL;
	using STREAMING = typename CONFIG::STREAMING;
	using DATA = typename CONFIG::DATA;

	using map_t = typename CONFIG::TRAITS::map_t;
	using idx = typename CONFIG::TRAITS::idx;
	using dreal = typename CONFIG::TRAITS::dreal;

	enum GEO : map_t
	{
		GEO_FLUID,	// compulsory
		GEO_WALL,	// compulsory
		GEO_INFLOW,
		GEO_OUTFLOW_EQ,
		GEO_OUTFLOW_RIGHT,
		GEO_OUTFLOW_RIGHT_INTERP,
		GEO_PERIODIC,
		GEO_NOTHING,
		GEO_SYM_TOP,
		GEO_SYM_BOTTOM,
		GEO_SYM_LEFT,
		GEO_SYM_RIGHT,
		GEO_FLUID_NEAR_WALL
	};

	__cuda_callable__ static bool isPeriodic(map_t mapgi)
	{
		return mapgi == GEO_PERIODIC;
	}

	__cuda_callable__ static bool isFluid(map_t mapgi)
	{
		return mapgi == GEO_FLUID || mapgi == GEO_FLUID_NEAR_WALL;
	}

	__cuda_callable__ static bool isWall(map_t mapgi)
	{
		return mapgi == GEO_WALL;
	}

	// Bouzidi interpolation helper (only used for D2Q9 GEO_FLUID_NEAR_WALL)
	// Returns the interpolated incoming DF using the coefficient theta for the given direction.
	// If theta < 0, performs ordinary streaming for the direction (using opposite DF from neighbor xs,ys).
	template <typename LBM_DATA>
	__cuda_callable__ static dreal f_bouzidi(LBM_DATA& SD, dreal theta, int k, int k_opposite, idx x, idx y, idx z,
	                                       idx xff, idx yff, idx xs, idx ys)
	{
		const dreal one = (dreal)1;
		const dreal two = (dreal)2;
		const dreal half = (dreal)0.5;

		// No Bouzidi: ordinary streaming value from opposite DF at neighbor (pull scheme equivalent)
		if (theta < (dreal)0)
			return SD.df(df_cur, k_opposite, xs, ys, z);

		if (theta <= half) {
			// (1 - 2*theta) * f_k(x+e, y+e) + (2*theta) * f_k(x, y)
			return (one - two * theta) * SD.df(df_cur, k, xff, yff, z)
			     + (two * theta) * SD.df(df_cur, k, x, y, z);
		}
		else {
			// (1 - 1/(2*theta)) * f_k_opposite(x,y) + (1/(2*theta)) * f_k(x,y)
			const dreal w = half / theta;
			return (one - w) * SD.df(df_cur, k_opposite, x, y, z)
			     + w * SD.df(df_cur, k, x, y, z);
		}
	}

	template <typename LBM_KS>
	__cuda_callable__ static void preCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING) {
			// nema zadny vliv na vypocet, jen pro output
			KS.rho = 1;
			KS.vx = 0;
			KS.vy = 0;
			return;
		}

		// modify pull location for streaming
		if (mapgi == GEO_OUTFLOW_RIGHT)
			xp = x = xm;

		if (mapgi != GEO_OUTFLOW_RIGHT_INTERP)
			STREAMING::streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);

		// boundary conditions
		switch (mapgi) {
			case GEO_INFLOW:
				SD.inflow(KS, x, y, z);
				KS.rho = 1;
				COLL::setEquilibrium(KS);
				break;
			case GEO_OUTFLOW_EQ:
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				COLL::setEquilibrium(KS);
				break;
			case GEO_OUTFLOW_RIGHT:
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				break;
			case GEO_OUTFLOW_RIGHT_INTERP:
				STREAMING::streamingInterpRight(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				COLL::setEquilibriumDecomposition(KS, 1);
				KS.rho = 1;
				break;
			case GEO_WALL:
				// does not affect the computation, only the output
				KS.rho = 1;
				KS.vx = 0;
				KS.vy = 0;
				// collision step: bounce-back
				TNL::swap(KS.f[mm], KS.f[pp]);
				TNL::swap(KS.f[zm], KS.f[zp]);
				TNL::swap(KS.f[mz], KS.f[pz]);
				TNL::swap(KS.f[mp], KS.f[pm]);
				break;
			case GEO_FLUID_NEAR_WALL:
			{
				// Apply Bouzidi interpolation on the 8 non-rest directions using per-voxel coefficients.
				// Direction order for coefficients: 0:east(pz),1:north(zp),2:west(mz),3:south(zm),4:ne(pp),5:nw(mp),6:sw(mm),7:se(pm)
				dreal th_e = SD.bouzidi_coeff_ptr ? SD.bouzidiCoeff(0, x, y, z) : (dreal)-1;
				dreal th_n = SD.bouzidi_coeff_ptr ? SD.bouzidiCoeff(1, x, y, z) : (dreal)-1;
				dreal th_w = SD.bouzidi_coeff_ptr ? SD.bouzidiCoeff(2, x, y, z) : (dreal)-1;
				dreal th_s = SD.bouzidi_coeff_ptr ? SD.bouzidiCoeff(3, x, y, z) : (dreal)-1;
				dreal th_ne = SD.bouzidi_coeff_ptr ? SD.bouzidiCoeff(4, x, y, z) : (dreal)-1;
				dreal th_nw = SD.bouzidi_coeff_ptr ? SD.bouzidiCoeff(5, x, y, z) : (dreal)-1;
				dreal th_sw = SD.bouzidi_coeff_ptr ? SD.bouzidiCoeff(6, x, y, z) : (dreal)-1;
				dreal th_se = SD.bouzidi_coeff_ptr ? SD.bouzidiCoeff(7, x, y, z) : (dreal)-1;

				// Override streamed values with Bouzidi interpolation
				KS.f[pz] = f_bouzidi(SD, th_e, mz, pz, x, y, z, xp, y, xm, y);
				KS.f[zp] = f_bouzidi(SD, th_n, zm, zp, x, y, z, x, yp, x, ym);
				KS.f[mz] = f_bouzidi(SD, th_w, pz, mz, x, y, z, xm, y, xp, y);
				KS.f[zm] = f_bouzidi(SD, th_s, zp, zm, x, y, z, x, ym, x, yp);
				KS.f[pp] = f_bouzidi(SD, th_ne, mm, pp, x, y, z, xp, yp, xm, ym);
				KS.f[mp] = f_bouzidi(SD, th_nw, pm, mp, x, y, z, xm, yp, xp, ym);
				KS.f[mm] = f_bouzidi(SD, th_sw, pp, mm, x, y, z, xm, ym, xp, yp);
				KS.f[pm] = f_bouzidi(SD, th_se, mp, pm, x, y, z, xp, ym, xm, yp);
				
				// Rest particle from ordinary streaming (already loaded), but ensure consistent with df_cur
				KS.f[zz] = SD.df(df_cur, zz, x, y, z);
				COLL::computeDensityAndVelocity(KS);
				break;
			}
			case GEO_SYM_LEFT:
				KS.f[pm] = KS.f[mm];
				KS.f[pz] = KS.f[mz];
				KS.f[pp] = KS.f[mp];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_RIGHT:
				KS.f[mm] = KS.f[pm];
				KS.f[mz] = KS.f[pz];
				KS.f[mp] = KS.f[pp];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_BOTTOM:
				KS.f[mp] = KS.f[mm];
				KS.f[zp] = KS.f[zm];
				KS.f[pp] = KS.f[pm];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_TOP:
				KS.f[mm] = KS.f[mp];
				KS.f[zm] = KS.f[zp];
				KS.f[pm] = KS.f[pp];
				COLL::computeDensityAndVelocity(KS);
				break;
			default:
				COLL::computeDensityAndVelocity(KS);
				break;
		}
	}

	__cuda_callable__ static bool doCollision(map_t mapgi)
	{
		// by default, collision is done on non-BC sites only
		// additionally, BCs which include the collision step should be specified here
		return isFluid(mapgi) || isPeriodic(mapgi) || mapgi == GEO_OUTFLOW_RIGHT || mapgi == GEO_OUTFLOW_RIGHT_INTERP;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void
	postCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING)
			return;

		STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}
};
