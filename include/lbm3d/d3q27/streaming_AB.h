#pragma once

// pull-scheme
template < typename TRAITS >
struct D3Q27_STREAMING
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void postCollisionStreaming(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// no streaming actually, write to the (x,y,z) site
		for (int i=0;i<27;i++) SD.df(df_out,i,x,y,z) = KS.f[i];
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void streaming(uint8_t type, LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		KS.f[mmm] = TNL::Backend::ldg(SD.df(type,mmm,xp,yp,zp));
		KS.f[mmz] = TNL::Backend::ldg(SD.df(type,mmz,xp,yp, z));
		KS.f[mmp] = TNL::Backend::ldg(SD.df(type,mmp,xp,yp,zm));
		KS.f[mzm] = TNL::Backend::ldg(SD.df(type,mzm,xp, y,zp));
		KS.f[mzz] = TNL::Backend::ldg(SD.df(type,mzz,xp, y, z));
		KS.f[mzp] = TNL::Backend::ldg(SD.df(type,mzp,xp, y,zm));
		KS.f[mpm] = TNL::Backend::ldg(SD.df(type,mpm,xp,ym,zp));
		KS.f[mpz] = TNL::Backend::ldg(SD.df(type,mpz,xp,ym, z));
		KS.f[mpp] = TNL::Backend::ldg(SD.df(type,mpp,xp,ym,zm));
		KS.f[zmm] = TNL::Backend::ldg(SD.df(type,zmm, x,yp,zp));
		KS.f[zmz] = TNL::Backend::ldg(SD.df(type,zmz, x,yp, z));
		KS.f[zmp] = TNL::Backend::ldg(SD.df(type,zmp, x,yp,zm));
		KS.f[zzm] = TNL::Backend::ldg(SD.df(type,zzm, x, y,zp));
		KS.f[zzz] = TNL::Backend::ldg(SD.df(type,zzz, x, y, z));
		KS.f[zzp] = TNL::Backend::ldg(SD.df(type,zzp, x, y,zm));
		KS.f[zpm] = TNL::Backend::ldg(SD.df(type,zpm, x,ym,zp));
		KS.f[zpz] = TNL::Backend::ldg(SD.df(type,zpz, x,ym, z));
		KS.f[zpp] = TNL::Backend::ldg(SD.df(type,zpp, x,ym,zm));
		KS.f[pmm] = TNL::Backend::ldg(SD.df(type,pmm,xm,yp,zp));
		KS.f[pmz] = TNL::Backend::ldg(SD.df(type,pmz,xm,yp, z));
		KS.f[pmp] = TNL::Backend::ldg(SD.df(type,pmp,xm,yp,zm));
		KS.f[pzm] = TNL::Backend::ldg(SD.df(type,pzm,xm, y,zp));
		KS.f[pzz] = TNL::Backend::ldg(SD.df(type,pzz,xm, y, z));
		KS.f[pzp] = TNL::Backend::ldg(SD.df(type,pzp,xm, y,zm));
		KS.f[ppm] = TNL::Backend::ldg(SD.df(type,ppm,xm,ym,zp));
		KS.f[ppz] = TNL::Backend::ldg(SD.df(type,ppz,xm,ym, z));
		KS.f[ppp] = TNL::Backend::ldg(SD.df(type,ppp,xm,ym,zm));
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void streaming(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streaming(df_cur, SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}

	// streaming with bounce-back rule applied
	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void streamingBounceBack(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		KS.f[ppp] = TNL::Backend::ldg(SD.df(df_cur,mmm,xp,yp,zp));
		KS.f[ppz] = TNL::Backend::ldg(SD.df(df_cur,mmz,xp,yp, z));
		KS.f[ppm] = TNL::Backend::ldg(SD.df(df_cur,mmp,xp,yp,zm));
		KS.f[pzp] = TNL::Backend::ldg(SD.df(df_cur,mzm,xp, y,zp));
		KS.f[pzz] = TNL::Backend::ldg(SD.df(df_cur,mzz,xp, y, z));
		KS.f[pzm] = TNL::Backend::ldg(SD.df(df_cur,mzp,xp, y,zm));
		KS.f[pmp] = TNL::Backend::ldg(SD.df(df_cur,mpm,xp,ym,zp));
		KS.f[pmz] = TNL::Backend::ldg(SD.df(df_cur,mpz,xp,ym, z));
		KS.f[pmm] = TNL::Backend::ldg(SD.df(df_cur,mpp,xp,ym,zm));
		KS.f[zpp] = TNL::Backend::ldg(SD.df(df_cur,zmm, x,yp,zp));
		KS.f[zpz] = TNL::Backend::ldg(SD.df(df_cur,zmz, x,yp, z));
		KS.f[zpm] = TNL::Backend::ldg(SD.df(df_cur,zmp, x,yp,zm));
		KS.f[zzp] = TNL::Backend::ldg(SD.df(df_cur,zzm, x, y,zp));
		KS.f[zzz] = TNL::Backend::ldg(SD.df(df_cur,zzz, x, y, z));
		KS.f[zzm] = TNL::Backend::ldg(SD.df(df_cur,zzp, x, y,zm));
		KS.f[zmp] = TNL::Backend::ldg(SD.df(df_cur,zpm, x,ym,zp));
		KS.f[zmz] = TNL::Backend::ldg(SD.df(df_cur,zpz, x,ym, z));
		KS.f[zmm] = TNL::Backend::ldg(SD.df(df_cur,zpp, x,ym,zm));
		KS.f[mpp] = TNL::Backend::ldg(SD.df(df_cur,pmm,xm,yp,zp));
		KS.f[mpz] = TNL::Backend::ldg(SD.df(df_cur,pmz,xm,yp, z));
		KS.f[mpm] = TNL::Backend::ldg(SD.df(df_cur,pmp,xm,yp,zm));
		KS.f[mzp] = TNL::Backend::ldg(SD.df(df_cur,pzm,xm, y,zp));
		KS.f[mzz] = TNL::Backend::ldg(SD.df(df_cur,pzz,xm, y, z));
		KS.f[mzm] = TNL::Backend::ldg(SD.df(df_cur,pzp,xm, y,zm));
		KS.f[mmp] = TNL::Backend::ldg(SD.df(df_cur,ppm,xm,ym,zp));
		KS.f[mmz] = TNL::Backend::ldg(SD.df(df_cur,ppz,xm,ym, z));
		KS.f[mmm] = TNL::Backend::ldg(SD.df(df_cur,ppp,xm,ym,zm));
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void streamingRho(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		KS.rho =
			  TNL::Backend::ldg(SD.df(df_cur,mmm,xp+1,yp,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,mmz,xp+1,yp,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,mmp,xp+1,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,mzm,xp+1,y ,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,mzz,xp+1,y ,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,mzp,xp+1,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,mpm,xp+1,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,mpz,xp+1,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,mpp,xp+1,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zmm,xp  ,yp,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,zmz,xp  ,yp,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,zmp,xp  ,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zzm,xp  ,y ,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,zzp,xp  ,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zzz,xp  ,y ,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,zpm,xp  ,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,zpz,xp  ,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,zpp,xp  ,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,pmm,x   ,yp,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,pmz,x   ,yp,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,pmp,x   ,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,pzm,x   ,y ,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,pzz,x   ,y ,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,pzp,x   ,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,ppm,x   ,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,ppz,x   ,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,ppp,x   ,ym,zm));
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static dreal streamingVx(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		KS.vx =
			  TNL::Backend::ldg(SD.df(df_cur,pmm,xm-1,yp,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,pmz,xm-1,yp,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,pmp,xm-1,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,ppm,xm-1,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,ppz,xm-1,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,ppp,xm-1,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,pzm,xm-1,y ,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,pzz,xm-1,y ,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,pzp,xm-1,y ,zm))
			- TNL::Backend::ldg(SD.df(df_cur,mzm,x   ,y ,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mzz,x   ,y ,z ))
			- TNL::Backend::ldg(SD.df(df_cur,mzp,x   ,y ,zm))
			- TNL::Backend::ldg(SD.df(df_cur,mmm,x   ,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mmz,x   ,yp,z ))
			- TNL::Backend::ldg(SD.df(df_cur,mmp,x   ,yp,zm))
			- TNL::Backend::ldg(SD.df(df_cur,mpm,x   ,ym,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mpz,x   ,ym,z ))
			- TNL::Backend::ldg(SD.df(df_cur,mpp,x   ,ym,zm));
	}


	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static dreal streamingVy(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		KS.vy =
			  TNL::Backend::ldg(SD.df(df_cur,mpm,x   ,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,mpz,x   ,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,mpp,x   ,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zpm,xm  ,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,zpz,xm  ,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,zpp,xm  ,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,ppm,xm-1,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,ppz,xm-1,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,ppp,xm-1,ym,zm))
			- TNL::Backend::ldg(SD.df(df_cur,zmm,xm  ,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,zmz,xm  ,yp,z ))
			- TNL::Backend::ldg(SD.df(df_cur,zmp,xm  ,yp,zm))
			- TNL::Backend::ldg(SD.df(df_cur,pmm,xm-1,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,pmz,xm-1,yp,z ))
			- TNL::Backend::ldg(SD.df(df_cur,pmp,xm-1,yp,zm))
			- TNL::Backend::ldg(SD.df(df_cur,mmm,x   ,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mmz,x   ,yp,z ))
			- TNL::Backend::ldg(SD.df(df_cur,mmp,x   ,yp,zm));
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static dreal streamingVz(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		KS.vz =
			  TNL::Backend::ldg(SD.df(df_cur,mmp,x   ,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,pmp,xm-1,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zmp,xm  ,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,pzp,xm-1,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zzp,xm  ,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,mzp,x   ,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,ppp,xm-1,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zpp,xm  ,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,mpp,x   ,ym,zm))
			- TNL::Backend::ldg(SD.df(df_cur,mmm,x   ,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,pmm,xm-1,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,zmm,xm  ,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,pzm,xm-1,y ,zp))
			- TNL::Backend::ldg(SD.df(df_cur,zzm,xm  ,y ,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mzm,x   ,y ,zp))
			- TNL::Backend::ldg(SD.df(df_cur,ppm,xm-1,ym,zp))
			- TNL::Backend::ldg(SD.df(df_cur,zpm,xm  ,ym,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mpm, x  ,ym,zp));
    }
};
