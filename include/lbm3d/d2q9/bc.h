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
		GEO_FLUID_NEAR_WALL,
		GEO_TRANSFER_FS,
		GEO_TRANSFER_SF,
		GEO_TRANSFER_SW
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

	// D2Q9 setup does not expose any solid phase, but the generic LBM_BLOCK helper
	// expects this predicate to exist (used for D3Q7 phi-transfer tagging). Return false.
	__cuda_callable__ static bool isSolid(map_t)
	{
		return false;
	}

	// Bouzidi interpolation helper (only used for D2Q9 GEO_FLUID_NEAR_WALL)
	// Returns the interpolated incoming DF using the coefficient theta for the given direction.
	// If theta < 0 or neighbor indices are out of bounds, returns ordinary streaming from the opposite DF.
	template <typename LBM_DATA>
	__cuda_callable__ static dreal f_bouzidi(
		LBM_DATA& SD, dreal theta,
		int k, int kbar,              // k = unknown incoming population, kbar = opposite(k)
		idx xA, idx yA, idx zA,       // fluid boundary node x_fA
		idx xB, idx yB                // interior neighbor x_fB = xA - e_k
	){
		const dreal half = (dreal)0.5;

		// Link does not hit wall → keep normally streamed value.
		if (theta < (dreal)0) return SD.df(df_cur, k, xA, yA, zA);

		const dreal fOppA = SD.df(df_cur, kbar, xA, yA, zA);

		if (theta <= half) {
			const dreal fOppB = SD.df(df_cur, kbar, xB, yB, zA);
			return (dreal)2*theta * fOppA + ((dreal)1 - (dreal)2*theta) * fOppB;
		} else {
			const dreal w = half / theta;
			const dreal fA = SD.df(df_cur, k, xA, yA, zA);
			return ((dreal)1 - w) * fOppA + w * fA;
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
                if (SD.bouzidi_coeff_ptr != nullptr && SD.use_bouzidi) {
                    // Apply Bouzidi interpolation using theta from the OPPOSITE direction (central inversion).
                    // Coefficient order: 0:east(pz),1:north(zp),2:west(mz),3:south(zm),4:ne(pp),5:nw(mp),6:sw(mm),7:se(pm)
                    const dreal th_e  = SD.bouzidiCoeff(0, x, y, z);
                    const dreal th_n  = SD.bouzidiCoeff(1, x, y, z);
                    const dreal th_w  = SD.bouzidiCoeff(2, x, y, z);
                    const dreal th_s  = SD.bouzidiCoeff(3, x, y, z);
                    const dreal th_ne = SD.bouzidiCoeff(4, x, y, z);
                    const dreal th_nw = SD.bouzidiCoeff(5, x, y, z);
                    const dreal th_sw = SD.bouzidiCoeff(6, x, y, z);
                    const dreal th_se = SD.bouzidiCoeff(7, x, y, z);

					KS.f[pz] = f_bouzidi(SD, th_e,  pz, mz, x,y,z,  xm, y);
					KS.f[zp] = f_bouzidi(SD, th_n,  zp, zm, x,y,z,   x, ym);
					KS.f[mz] = f_bouzidi(SD, th_w,  mz, pz, x,y,z,  xp, y);
					KS.f[zm] = f_bouzidi(SD, th_s,  zm, zp, x,y,z,   x, yp);

					KS.f[pp] = f_bouzidi(SD, th_ne, pp, mm, x,y,z,  xm, ym);
					KS.f[mp] = f_bouzidi(SD, th_nw, mp, pm, x,y,z,  xp, ym);
					KS.f[mm] = f_bouzidi(SD, th_sw, mm, pp, x,y,z,  xp, yp);
					KS.f[pm] = f_bouzidi(SD, th_se, pm, mp, x,y,z,  xm, yp);

                    // Rest particle from ordinary streaming (already loaded), but ensure consistent with df_cur
                    KS.f[zz] = SD.df(df_cur, zz, x, y, z);
                }
                else {
                    // Bouzidi disabled or missing coefficients: treat as classic bounce-back boundary
                    TNL::swap(KS.f[mm], KS.f[pp]);
                    TNL::swap(KS.f[zm], KS.f[zp]);
                    TNL::swap(KS.f[mz], KS.f[pz]);
                    TNL::swap(KS.f[mp], KS.f[pm]);
                }
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
