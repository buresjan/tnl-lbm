template< typename LBM_TYPE >
void LBM<LBM_TYPE>::setEqLat(uint8_t dfty, idx x, idx y, idx z, real rho, real vx, real vy, real vz)
{
	hfs[dfty](mmm,x,y,z) = LBM_TYPE::EQ::eq_mmm(rho,vx,vy,vz);
	hfs[dfty](zmm,x,y,z) = LBM_TYPE::EQ::eq_zmm(rho,vx,vy,vz);
	hfs[dfty](pmm,x,y,z) = LBM_TYPE::EQ::eq_pmm(rho,vx,vy,vz);
	hfs[dfty](mzm,x,y,z) = LBM_TYPE::EQ::eq_mzm(rho,vx,vy,vz);
	hfs[dfty](zzm,x,y,z) = LBM_TYPE::EQ::eq_zzm(rho,vx,vy,vz);
	hfs[dfty](pzm,x,y,z) = LBM_TYPE::EQ::eq_pzm(rho,vx,vy,vz);
	hfs[dfty](mpm,x,y,z) = LBM_TYPE::EQ::eq_mpm(rho,vx,vy,vz);
	hfs[dfty](zpm,x,y,z) = LBM_TYPE::EQ::eq_zpm(rho,vx,vy,vz);
	hfs[dfty](ppm,x,y,z) = LBM_TYPE::EQ::eq_ppm(rho,vx,vy,vz);

	hfs[dfty](mmz,x,y,z) = LBM_TYPE::EQ::eq_mmz(rho,vx,vy,vz);
	hfs[dfty](zmz,x,y,z) = LBM_TYPE::EQ::eq_zmz(rho,vx,vy,vz);
	hfs[dfty](pmz,x,y,z) = LBM_TYPE::EQ::eq_pmz(rho,vx,vy,vz);
	hfs[dfty](mzz,x,y,z) = LBM_TYPE::EQ::eq_mzz(rho,vx,vy,vz);
	hfs[dfty](zzz,x,y,z) = LBM_TYPE::EQ::eq_zzz(rho,vx,vy,vz);
	hfs[dfty](pzz,x,y,z) = LBM_TYPE::EQ::eq_pzz(rho,vx,vy,vz);
	hfs[dfty](mpz,x,y,z) = LBM_TYPE::EQ::eq_mpz(rho,vx,vy,vz);
	hfs[dfty](zpz,x,y,z) = LBM_TYPE::EQ::eq_zpz(rho,vx,vy,vz);
	hfs[dfty](ppz,x,y,z) = LBM_TYPE::EQ::eq_ppz(rho,vx,vy,vz);

	hfs[dfty](mmp,x,y,z) = LBM_TYPE::EQ::eq_mmp(rho,vx,vy,vz);
	hfs[dfty](zmp,x,y,z) = LBM_TYPE::EQ::eq_zmp(rho,vx,vy,vz);
	hfs[dfty](pmp,x,y,z) = LBM_TYPE::EQ::eq_pmp(rho,vx,vy,vz);
	hfs[dfty](mzp,x,y,z) = LBM_TYPE::EQ::eq_mzp(rho,vx,vy,vz);
	hfs[dfty](zzp,x,y,z) = LBM_TYPE::EQ::eq_zzp(rho,vx,vy,vz);
	hfs[dfty](pzp,x,y,z) = LBM_TYPE::EQ::eq_pzp(rho,vx,vy,vz);
	hfs[dfty](mpp,x,y,z) = LBM_TYPE::EQ::eq_mpp(rho,vx,vy,vz);
	hfs[dfty](zpp,x,y,z) = LBM_TYPE::EQ::eq_zpp(rho,vx,vy,vz);
	hfs[dfty](ppp,x,y,z) = LBM_TYPE::EQ::eq_ppp(rho,vx,vy,vz);
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::setEqLat(idx x, idx y, idx z, real rho, real vx, real vy, real vz)
{
	for (uint8_t dfty=0; dfty<DFMAX; dfty++) setEqLat(dfty, x, y, z, rho, vx, vy, vz);
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::resetForces(real ifx, real ify, real ifz)
{
	/// Reset forces - This is necessary since '+=' is used afterwards.
	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset_X; x < offset_X + local_X; x++)
	for (idx z = offset_Z; z < offset_Z + local_Z; z++)
	for (idx y = offset_Y; y < offset_Y + local_Y; y++)
	{
		hmacro(MACRO::e_fx, x, y, z) = ifx;
		hmacro(MACRO::e_fy, x, y, z) = ify;
		hmacro(MACRO::e_fz, x, y, z) = ifz;
	}
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyForcesToDevice()
{
	// FIXME: overlaps
	#ifdef USE_CUDA
	cudaMemcpy(dfx(), hfx(), local_X*local_Y*local_Z*sizeof(dreal), cudaMemcpyHostToDevice);
	cudaMemcpy(dfy(), hfy(), local_X*local_Y*local_Z*sizeof(dreal), cudaMemcpyHostToDevice);
	cudaMemcpy(dfz(), hfz(), local_X*local_Y*local_Z*sizeof(dreal), cudaMemcpyHostToDevice);
	checkCudaDevice;
	#endif
}


template< typename LBM_TYPE >
bool LBM<LBM_TYPE>::isLocalIndex(idx x, idx y, idx z)
{
	return x >= offset_X && x < offset_X + local_X &&
		y >= offset_Y && y < offset_Y + local_Y &&
		z >= offset_Z && z < offset_Z + local_Z;
}

template< typename LBM_TYPE >
bool LBM<LBM_TYPE>::isLocalX(idx x)
{
	return x >= offset_X && x < offset_X + local_X;
}

template< typename LBM_TYPE >
bool LBM<LBM_TYPE>::isLocalY(idx y)
{
	return y >= offset_Y && y < offset_Y + local_Y;
}

template< typename LBM_TYPE >
bool LBM<LBM_TYPE>::isLocalZ(idx z)
{
	return z >= offset_Z && z < offset_Z + local_Z;
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::defineWall(idx x, idx y, idx z, bool value)
{
//	if (x>0 && x<X-1 && y > 0 && y<Y-1 && z > 0 && z<Z-1) wall(x,y,z) = value;
//	if (x>=0 && x<=X-1 && y >= 0 && y<=Y-1 && z>=0 && z<=Z-1) wall(x,y,z) = value;
	if (isLocalIndex(x, y, z)) wall(x, y, z) = value;
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::setBoundaryX(idx x, map_t value)
{
	if (isLocalX(x))
		for (idx y = offset_Y; y < offset_Y + local_Y; y++)
		for (idx z = offset_Z; z < offset_Z + local_Z; z++)
			map(x, y, z) = value;
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::setBoundaryY(idx y, map_t value)
{
	if (isLocalY(y))
		for (idx x = offset_X; x < offset_X + local_X; x++)
		for (idx z = offset_Z; z < offset_Z + local_Z; z++)
			map(x, y, z) = value;
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::setBoundaryZ(idx z, map_t value)
{
	if (isLocalZ(z))
		for (idx x = offset_X; x < offset_X + local_X; x++)
		for (idx y = offset_Y; y < offset_Y + local_Y; y++)
			map(x, y, z) = value;
}

template< typename LBM_TYPE >
bool LBM<LBM_TYPE>::getWall(idx x, idx y, idx z)
{
	if (!isLocalIndex(x, y, z)) return false;
	return wall(x,y,z);
}

template< typename LBM_TYPE >
bool LBM<LBM_TYPE>::isFluid(idx x, idx y, idx z)
{
	if (!isLocalIndex(x, y, z)) return false;
	return LBM_TYPE::BC::isFluid(map(x,y,z));
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::projectWall()
{
	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset_X; x < offset_X + local_X; x++)
	for (idx z = offset_Y; z < offset_Y + local_Z; z++)
	for (idx y = offset_Z; y < offset_Z + local_Y; y++)
	{
		if (wall(x, y, z))
			map(x, y, z) = LBM_TYPE::BC::GEO_WALL;
	}
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::resetMap(map_t geo_type)
{
	hmap.setValue(geo_type);
}


template< typename LBM_TYPE >
void  LBM<LBM_TYPE>::copyMapToDevice()
{
	dmap = hmap;
}

template< typename LBM_TYPE >
void  LBM<LBM_TYPE>::copyMapToHost()
{
	hmap = dmap;
}


template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyDFsToHost(uint8_t dfty)
{
	dlat_view_t df = dfs[0].getView();
	df.bind(data.dfs[dfty]);
	hfs[dfty] = df;
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyDFsToDevice(uint8_t dfty)
{
	dlat_view_t df = dfs[0].getView();
	df.bind(data.dfs[dfty]);
	df = hfs[dfty];
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyDFsToHost()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
		hfs[dfty] = dfs[dfty];
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyDFsToDevice()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
		dfs[dfty] = hfs[dfty];
}

#ifdef HAVE_MPI

template< typename LBM_TYPE >
	template< typename Array >
void LBM<LBM_TYPE>::startDrealArraySynchronization(Array& array, int sync_offset)
{
	static_assert( Array::getDimension() == 4, "4D array expected" );
	constexpr int N = Array::SizesHolderType::template getStaticSize<0>();
	static_assert( N > 0, "the first dimension must be static" );
	constexpr bool is_df = std::is_same< typename Array::ConstViewType, typename dlat_array_t::ConstViewType >::value;

	// NOTE: threadpool and async require MPI_THREAD_MULTIPLE which is slow
	constexpr auto policy = std::decay_t<decltype(dreal_sync[0])>::AsyncPolicy::deferred;

	// empty view, but with correct sizes
	#ifdef HAVE_MPI
	typename sync_array_t::LocalViewType localView(nullptr, data.indexer);
	typename sync_array_t::ViewType view(localView, dmap.getSizes(), dmap.getLocalBegins(), dmap.getLocalEnds(), dmap.getCommunicator());
	#else
	typename sync_array_t::ViewType view(nullptr, data.indexer);
	#endif

	for (int i = 0; i < N; i++) {
		// rebind just the data pointer
		view.bind(array.getData() + i * data.indexer.getStorageSize());
		// determine sync direction
		const TNL::Containers::SyncDirection sync_direction = (is_df) ? df_sync_directions[i] : TNL::Containers::SyncDirection::All;
		// start the synchronization
		dreal_sync[i + sync_offset].synchronizeAsync(view, policy, sync_direction);
	}
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::synchronizeDFsDevice_start(uint8_t dftype)
{
	auto df = dfs[0].getView();
	df.bind(data.dfs[dftype]);
	startDrealArraySynchronization(df, 0);
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::synchronizeDFsDevice(uint8_t dftype)
{
	synchronizeDFsDevice_start(dftype);
	for (int i = 0; i < 27; i++)
		dreal_sync[i].wait();
	cudaDeviceSynchronize();
	checkCudaDevice;
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::synchronizeMacroDevice_start()
{
	startDrealArraySynchronization(dmacro, 27);
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::synchronizeMacroDevice()
{
	synchronizeMacroDevice_start();
	for (int i = 0; i < MACRO::N; i++)
		dreal_sync[27 + i].wait();
	cudaDeviceSynchronize();
	checkCudaDevice;
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::synchronizeMapDevice()
{
	map_sync.synchronize(dmap);
	cudaDeviceSynchronize();
	checkCudaDevice;
}
#endif

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyMacroToHost()
{
	hmacro = dmacro;
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::copyMacroToDevice()
{
	dmacro = hmacro;
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::computeCPUMacroFromLat()
{
	// take Lat, compute KS and then CPU_MACRO
	if (CPU_MACRO::N > 0)
	{
		typename LBM_TYPE::DATA SD;
		for (uint8_t dfty=0;dfty<DFMAX;dfty++)
			SD.dfs[dfty] = hfs[dfty].getData();
		#ifdef HAVE_MPI
		SD.indexer = hmap.getLocalView().getIndexer();
		#else
		SD.indexer = hmap.getIndexer();
		#endif
		SD.XYZ = SD.indexer.getStorageSize();
		SD.dmap = hmap.getData();
		SD.dmacro = cpumacro.getData();

		#pragma omp parallel for schedule(static) collapse(2)
		for (idx x=0; x<local_X; x++)
		for (idx z=0; z<local_Z; z++)
		for (idx y=0; y<local_Y; y++)
		{
			typename LBM_TYPE::template KernelStruct<dreal> KS;
			KS.fx=0;
			KS.fy=0;
			KS.fz=0;
			LBM_TYPE::COLL::copyDFcur2KS(SD, KS, x, y, z);
			LBM_TYPE::COLL::computeDensityAndVelocity(KS);
			CPU_MACRO::outputMacro(SD, KS, x, y, z);
//			if (x==128 && y==23 && z==103)
//			printf("KS: %e %e %e %e vs. cpumacro %e %e %e %e [at %d %d %d]\n", KS.vx, KS.vy, KS.vz, KS.rho, cpumacro[mpos(CPU_MACRO::e_vx,x,y,z)], cpumacro[mpos(CPU_MACRO::e_vy,x,y,z)], cpumacro[mpos(CPU_MACRO::e_vz,x,y,z)],cpumacro[mpos(CPU_MACRO::e_rho,x,y,z)],x,y,z);
		}
//                printf("computeCPUMAcroFromLat done.\n");
	}
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::allocateHostData()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
	{
		hfs[dfty].setSizes(0, global_X, global_Y, global_Z);
		#ifdef HAVE_MPI
		hfs[dfty].template setDistribution< 1 >(offset_X, offset_X + local_X, MPI_COMM_WORLD);
		hfs[dfty].allocate();
		#endif
	}

	hmap.setSizes(global_X, global_Y, global_Z);
	wall.setSizes(global_X, global_Y, global_Z);
#ifdef HAVE_MPI
	hmap.template setDistribution< 0 >(offset_X, offset_X + local_X, MPI_COMM_WORLD);
	hmap.allocate();
	wall.template setDistribution< 0 >(offset_X, offset_X + local_X, MPI_COMM_WORLD);
	wall.allocate();
#endif
	wall.setValue(false);

	hmacro.setSizes(0, global_X, global_Y, global_Z);
	cpumacro.setSizes(0, global_X, global_Y, global_Z);
#ifdef HAVE_MPI
	hmacro.template setDistribution< 1 >(offset_X, offset_X + local_X, MPI_COMM_WORLD);
	hmacro.allocate();
	cpumacro.template setDistribution< 1 >(offset_X, offset_X + local_X, MPI_COMM_WORLD);
	cpumacro.allocate();
#endif
	hmacro.setValue(0);
	// avoid setting empty array
	if (CPU_MACRO::N)
		cpumacro.setValue(0);
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::allocateDeviceData()
{
#ifdef USE_CUDA
	dmap.setSizes(global_X, global_Y, global_Z);
	#ifdef HAVE_MPI
	dmap.template setDistribution< 0 >(offset_X, offset_X + local_X, MPI_COMM_WORLD);
	dmap.allocate();
	#endif

	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
	{
		dfs[dfty].setSizes(0, global_X, global_Y, global_Z);
		#ifdef HAVE_MPI
		dfs[dfty].template setDistribution< 1 >(offset_X, offset_X + local_X, MPI_COMM_WORLD);
		dfs[dfty].allocate();
		#endif
	}

	dmacro.setSizes(0, global_X, global_Y, global_Z);
	#ifdef HAVE_MPI
	dmacro.template setDistribution< 1 >(offset_X, offset_X + local_X, MPI_COMM_WORLD);
	dmacro.allocate();
	#endif
#else
	// TODO: skip douple allocation !!!
//	dmap=hmap;
//	dmacro=hmacro;
//	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
//		dfs[dfty] = (dreal*)malloc(27*size_dreal);
////	df1 = (dreal*)malloc(27*size_dreal);
////	df2 = (dreal*)malloc(27*size_dreal);
#endif

	// initialize data pointers
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
		data.dfs[dfty] = dfs[dfty].getData();
	#ifdef HAVE_MPI
	data.indexer = dmap.getLocalView().getIndexer();
	#else
	data.indexer = dmap.getIndexer();
	#endif
	data.XYZ = data.indexer.getStorageSize();
	data.dmap = dmap.getData();
	data.dmacro = dmacro.getData();
}

template< typename LBM_TYPE >
void LBM<LBM_TYPE>::updateKernelData()
{
	data.lbmViscosity = (dreal)lbmViscosity();
//	data.lbmInputDensity = (dreal)lbmInputDensity();

	// rotation
	int i = iterations % DFMAX; 			// i = 0, 1, 2, ... DMAX-1

	for (int k=0;k<DFMAX;k++)
	{
		int knew = (k-i)<=0 ? (k-i+DFMAX) % DFMAX : k-i;
//		data.dfs[k] = dfs[knew];
		data.dfs[k] = dfs[knew].getData();
//		printf("updateKernelData:: assigning data.dfs[%d] = dfs[%d]\n",k, knew);
	}
}


template< typename LBM_TYPE >
LBM<LBM_TYPE>::LBM(idx iX, idx iY, idx iZ, real iphysViscosity, real iphysDl, real iphysDt, point_t iphysOrigin)
{
	global_X = iX;
	global_Y = iY;
	global_Z = iZ;

	// initialize MPI info
	rank = TNL::MPI::GetRank();
	nproc = TNL::MPI::GetSize();
	auto local_range = TNL::Containers::Partitioner<idx>::splitRange(global_X, MPI_COMM_WORLD);
	local_X = local_range.getEnd() - local_range.getBegin();
	offset_X = local_range.getBegin();
	local_Y = global_Y;
	offset_Y = 0;
	local_Z = global_Z;
	offset_Z = 0;

	physOrigin = iphysOrigin;
	physDl = iphysDl;
	physDt = iphysDt;

	physCharLength = physDl * (real)global_Y;

	physViscosity = iphysViscosity;
	physFluidDensity = 1000.0;		// override this to your fluid

	iterations = 0;

	physFinalTime = 1e10;
	physStartTime = 0;
	terminate=false;
}

template< typename LBM_TYPE >
LBM<LBM_TYPE>::~LBM()
{
}
