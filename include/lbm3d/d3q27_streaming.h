#pragma once

template < typename TRAITS >
struct D3Q27_STREAMING
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void streaming(uint8_t type, LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		KS.f[mmm] = SD.df(type,mmm,xp,yp,zp);
		KS.f[mmz] = SD.df(type,mmz,xp,yp, z);
		KS.f[mmp] = SD.df(type,mmp,xp,yp,zm);
		KS.f[mzm] = SD.df(type,mzm,xp, y,zp);
		KS.f[mzz] = SD.df(type,mzz,xp, y, z);
		KS.f[mzp] = SD.df(type,mzp,xp, y,zm);
		KS.f[mpm] = SD.df(type,mpm,xp,ym,zp);
		KS.f[mpz] = SD.df(type,mpz,xp,ym, z);
		KS.f[mpp] = SD.df(type,mpp,xp,ym,zm);
		KS.f[zmm] = SD.df(type,zmm, x,yp,zp);
		KS.f[zmz] = SD.df(type,zmz, x,yp, z);
		KS.f[zmp] = SD.df(type,zmp, x,yp,zm);
		KS.f[zzm] = SD.df(type,zzm, x, y,zp);
		KS.f[zzz] = SD.df(type,zzz, x, y, z);
		KS.f[zzp] = SD.df(type,zzp, x, y,zm);
		KS.f[zpm] = SD.df(type,zpm, x,ym,zp);
		KS.f[zpz] = SD.df(type,zpz, x,ym, z);
		KS.f[zpp] = SD.df(type,zpp, x,ym,zm);
		KS.f[pmm] = SD.df(type,pmm,xm,yp,zp);
		KS.f[pmz] = SD.df(type,pmz,xm,yp, z);
		KS.f[pmp] = SD.df(type,pmp,xm,yp,zm);
		KS.f[pzm] = SD.df(type,pzm,xm, y,zp);
		KS.f[pzz] = SD.df(type,pzz,xm, y, z);
		KS.f[pzp] = SD.df(type,pzp,xm, y,zm);
		KS.f[ppm] = SD.df(type,ppm,xm,ym,zp);
		KS.f[ppz] = SD.df(type,ppz,xm,ym, z);
		KS.f[ppp] = SD.df(type,ppp,xm,ym,zm);
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void streaming(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streaming(df_cur, SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
/*
		KS.f[mmm] = SD.df(df_cur,mmm,xp,yp,zp);
		KS.f[mmz] = SD.df(df_cur,mmz,xp,yp, z);
		KS.f[mmp] = SD.df(df_cur,mmp,xp,yp,zm);
		KS.f[mzm] = SD.df(df_cur,mzm,xp, y,zp);
		KS.f[mzz] = SD.df(df_cur,mzz,xp, y, z);
		KS.f[mzp] = SD.df(df_cur,mzp,xp, y,zm);
		KS.f[mpm] = SD.df(df_cur,mpm,xp,ym,zp);
		KS.f[mpz] = SD.df(df_cur,mpz,xp,ym, z);
		KS.f[mpp] = SD.df(df_cur,mpp,xp,ym,zm);
		KS.f[zmm] = SD.df(df_cur,zmm, x,yp,zp);
		KS.f[zmz] = SD.df(df_cur,zmz, x,yp, z);
		KS.f[zmp] = SD.df(df_cur,zmp, x,yp,zm);
		KS.f[zzm] = SD.df(df_cur,zzm, x, y,zp);
		KS.f[zzz] = SD.df(df_cur,zzz, x, y, z);
		KS.f[zzp] = SD.df(df_cur,zzp, x, y,zm);
		KS.f[zpm] = SD.df(df_cur,zpm, x,ym,zp);
		KS.f[zpz] = SD.df(df_cur,zpz, x,ym, z);
		KS.f[zpp] = SD.df(df_cur,zpp, x,ym,zm);
		KS.f[pmm] = SD.df(df_cur,pmm,xm,yp,zp);
		KS.f[pmz] = SD.df(df_cur,pmz,xm,yp, z);
		KS.f[pmp] = SD.df(df_cur,pmp,xm,yp,zm);
		KS.f[pzm] = SD.df(df_cur,pzm,xm, y,zp);
		KS.f[pzz] = SD.df(df_cur,pzz,xm, y, z);
		KS.f[pzp] = SD.df(df_cur,pzp,xm, y,zm);
		KS.f[ppm] = SD.df(df_cur,ppm,xm,ym,zp);
		KS.f[ppz] = SD.df(df_cur,ppz,xm,ym, z);
		KS.f[ppp] = SD.df(df_cur,ppp,xm,ym,zm);
*/
	}

	// streaming with bounce-back rule applied
	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void streamingBounceBack(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		KS.f[ppp] = SD.df(df_cur,mmm,xp,yp,zp);
		KS.f[ppz] = SD.df(df_cur,mmz,xp,yp, z);
		KS.f[ppm] = SD.df(df_cur,mmp,xp,yp,zm);
		KS.f[pzp] = SD.df(df_cur,mzm,xp, y,zp);
		KS.f[pzz] = SD.df(df_cur,mzz,xp, y, z);
		KS.f[pzm] = SD.df(df_cur,mzp,xp, y,zm);
		KS.f[pmp] = SD.df(df_cur,mpm,xp,ym,zp);
		KS.f[pmz] = SD.df(df_cur,mpz,xp,ym, z);
		KS.f[pmm] = SD.df(df_cur,mpp,xp,ym,zm);
		KS.f[zpp] = SD.df(df_cur,zmm, x,yp,zp);
		KS.f[zpz] = SD.df(df_cur,zmz, x,yp, z);
		KS.f[zpm] = SD.df(df_cur,zmp, x,yp,zm);
		KS.f[zzp] = SD.df(df_cur,zzm, x, y,zp);
		KS.f[zzz] = SD.df(df_cur,zzz, x, y, z);
		KS.f[zzm] = SD.df(df_cur,zzp, x, y,zm);
		KS.f[zmp] = SD.df(df_cur,zpm, x,ym,zp);
		KS.f[zmz] = SD.df(df_cur,zpz, x,ym, z);
		KS.f[zmm] = SD.df(df_cur,zpp, x,ym,zm);
		KS.f[mpp] = SD.df(df_cur,pmm,xm,yp,zp);
		KS.f[mpz] = SD.df(df_cur,pmz,xm,yp, z);
		KS.f[mpm] = SD.df(df_cur,pmp,xm,yp,zm);
		KS.f[mzp] = SD.df(df_cur,pzm,xm, y,zp);
		KS.f[mzz] = SD.df(df_cur,pzz,xm, y, z);
		KS.f[mzm] = SD.df(df_cur,pzp,xm, y,zm);
		KS.f[mmp] = SD.df(df_cur,ppm,xm,ym,zp);
		KS.f[mmz] = SD.df(df_cur,ppz,xm,ym, z);
		KS.f[mmm] = SD.df(df_cur,ppp,xm,ym,zm);
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void streamingRho(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		KS.rho =
			  SD.df(df_cur,mmm,xp+1,yp,zp)
			+ SD.df(df_cur,mmz,xp+1,yp,z )
			+ SD.df(df_cur,mmp,xp+1,yp,zm)
			+ SD.df(df_cur,mzm,xp+1,y ,zp)
			+ SD.df(df_cur,mzz,xp+1,y ,z )
			+ SD.df(df_cur,mzp,xp+1,y ,zm)
			+ SD.df(df_cur,mpm,xp+1,ym,zp)
			+ SD.df(df_cur,mpz,xp+1,ym,z )
			+ SD.df(df_cur,mpp,xp+1,ym,zm)
			+ SD.df(df_cur,zmm,xp  ,yp,zp)
			+ SD.df(df_cur,zmz,xp  ,yp,z )
			+ SD.df(df_cur,zmp,xp  ,yp,zm)
			+ SD.df(df_cur,zzm,xp  ,y ,zp)
			+ SD.df(df_cur,zzp,xp  ,y ,zm)
			+ SD.df(df_cur,zzz,xp  ,y ,z )
			+ SD.df(df_cur,zpm,xp  ,ym,zp)
			+ SD.df(df_cur,zpz,xp  ,ym,z )
			+ SD.df(df_cur,zpp,xp  ,ym,zm)
			+ SD.df(df_cur,pmm,x   ,yp,zp)
			+ SD.df(df_cur,pmz,x   ,yp,z )
			+ SD.df(df_cur,pmp,x   ,yp,zm)
			+ SD.df(df_cur,pzm,x   ,y ,zp)
			+ SD.df(df_cur,pzz,x   ,y ,z )
			+ SD.df(df_cur,pzp,x   ,y ,zm)
			+ SD.df(df_cur,ppm,x   ,ym,zp)
			+ SD.df(df_cur,ppz,x   ,ym,z )
			+ SD.df(df_cur,ppp,x   ,ym,zm);
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static dreal streamingVx(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		KS.vx =
			  SD.df(df_cur,pmm,xm-1,yp,zp)
			+ SD.df(df_cur,pmz,xm-1,yp,z )
			+ SD.df(df_cur,pmp,xm-1,yp,zm)
			+ SD.df(df_cur,ppm,xm-1,ym,zp)
			+ SD.df(df_cur,ppz,xm-1,ym,z )
			+ SD.df(df_cur,ppp,xm-1,ym,zm)
			+ SD.df(df_cur,pzm,xm-1,y ,zp)
			+ SD.df(df_cur,pzz,xm-1,y ,z )
			+ SD.df(df_cur,pzp,xm-1,y ,zm)
			- SD.df(df_cur,mzm,x   ,y ,zp)
			- SD.df(df_cur,mzz,x   ,y ,z )
			- SD.df(df_cur,mzp,x   ,y ,zm)
			- SD.df(df_cur,mmm,x   ,yp,zp)
			- SD.df(df_cur,mmz,x   ,yp,z )
			- SD.df(df_cur,mmp,x   ,yp,zm)
			- SD.df(df_cur,mpm,x   ,ym,zp)
			- SD.df(df_cur,mpz,x   ,ym,z )
			- SD.df(df_cur,mpp,x   ,ym,zm);
	}


	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static dreal streamingVy(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		KS.vy =
			  SD.df(df_cur,mpm,x   ,ym,zp)
			+ SD.df(df_cur,mpz,x   ,ym,z )
			+ SD.df(df_cur,mpp,x   ,ym,zm)
			+ SD.df(df_cur,zpm,xm  ,ym,zp)
			+ SD.df(df_cur,zpz,xm  ,ym,z )
			+ SD.df(df_cur,zpp,xm  ,ym,zm)
			+ SD.df(df_cur,ppm,xm-1,ym,zp)
			+ SD.df(df_cur,ppz,xm-1,ym,z )
			+ SD.df(df_cur,ppp,xm-1,ym,zm)
			- SD.df(df_cur,zmm,xm  ,yp,zp)
			- SD.df(df_cur,zmz,xm  ,yp,z )
			- SD.df(df_cur,zmp,xm  ,yp,zm)
			- SD.df(df_cur,pmm,xm-1,yp,zp)
			- SD.df(df_cur,pmz,xm-1,yp,z )
			- SD.df(df_cur,pmp,xm-1,yp,zm)
			- SD.df(df_cur,mmm,x   ,yp,zp)
			- SD.df(df_cur,mmz,x   ,yp,z )
			- SD.df(df_cur,mmp,x   ,yp,zm);
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static dreal streamingVz(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		KS.vz =
			  SD.df(df_cur,mmp,x   ,yp,zm)
			+ SD.df(df_cur,pmp,xm-1,yp,zm)
			+ SD.df(df_cur,zmp,xm  ,yp,zm)
			+ SD.df(df_cur,pzp,xm-1,y ,zm)
			+ SD.df(df_cur,zzp,xm  ,y ,zm)
			+ SD.df(df_cur,mzp,x   ,y ,zm)
			+ SD.df(df_cur,ppp,xm-1,ym,zm)
			+ SD.df(df_cur,zpp,xm  ,ym,zm)
			+ SD.df(df_cur,mpp,x   ,ym,zm)
			- SD.df(df_cur,mmm,x   ,yp,zp)
			- SD.df(df_cur,pmm,xm-1,yp,zp)
			- SD.df(df_cur,zmm,xm  ,yp,zp)
			- SD.df(df_cur,pzm,xm-1,y ,zp)
			- SD.df(df_cur,zzm,xm  ,y ,zp)
			- SD.df(df_cur,mzm,x   ,y ,zp)
			- SD.df(df_cur,ppm,xm-1,ym,zp)
			- SD.df(df_cur,zpm,xm  ,ym,zp)
			- SD.df(df_cur,mpm, x  ,ym,zp);
    }
};
