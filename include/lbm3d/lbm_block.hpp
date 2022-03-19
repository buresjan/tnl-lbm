#pragma once

#include "lbm_block.h"
#include "vtk_writer.h"

template< typename LBM_TYPE >
LBM_BLOCK<LBM_TYPE>::LBM_BLOCK(idx3d global, idx3d local, idx3d offset, int neighbour_left, int neighbour_right, int left_id, int this_id, int right_id)
: global(global), local(local), offset(offset), left_id(left_id), id(this_id), right_id(right_id)
{
	// initialize MPI info
	rank = TNL::MPI::GetRank();
	nproc = TNL::MPI::GetSize();

	// initialize neighbours
	if (neighbour_left < 0)
		this->neighbour_left = (rank + nproc - 1) % nproc;
	else
		this->neighbour_left = neighbour_left;
	if (neighbour_right < 0)
		this->neighbour_right = (rank + 1) % nproc;
	else
		this->neighbour_right = neighbour_right;
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::setEqLat(idx x, idx y, idx z, real rho, real vx, real vy, real vz)
{
	for (uint8_t dfty=0; dfty<DFMAX; dfty++)
		LBM_TYPE::COLL::setEquilibriumLat(hfs[dfty], x, y, z, rho, vx, vy, vz);
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::resetForces(real ifx, real ify, real ifz)
{
	/// Reset forces - This is necessary since '+=' is used afterwards.
	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
	{
		hmacro(MACRO::e_fx, x, y, z) = ifx;
		hmacro(MACRO::e_fy, x, y, z) = ify;
		hmacro(MACRO::e_fz, x, y, z) = ifz;
	}
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::copyForcesToDevice()
{
	// FIXME: overlaps
	#ifdef USE_CUDA
	cudaMemcpy(dfx(), hfx(), local.x()*local.y()*local.z()*sizeof(dreal), cudaMemcpyHostToDevice);
	cudaMemcpy(dfy(), hfy(), local.x()*local.y()*local.z()*sizeof(dreal), cudaMemcpyHostToDevice);
	cudaMemcpy(dfz(), hfz(), local.x()*local.y()*local.z()*sizeof(dreal), cudaMemcpyHostToDevice);
	checkCudaDevice;
	#endif
}


template< typename LBM_TYPE >
bool LBM_BLOCK<LBM_TYPE>::isLocalIndex(idx x, idx y, idx z) const
{
	return x >= offset.x() && x < offset.x() + local.x() &&
		y >= offset.y() && y < offset.y() + local.y() &&
		z >= offset.z() && z < offset.z() + local.z();
}

template< typename LBM_TYPE >
bool LBM_BLOCK<LBM_TYPE>::isLocalX(idx x) const
{
	return x >= offset.x() && x < offset.x() + local.x();
}

template< typename LBM_TYPE >
bool LBM_BLOCK<LBM_TYPE>::isLocalY(idx y) const
{
	return y >= offset.y() && y < offset.y() + local.y();
}

template< typename LBM_TYPE >
bool LBM_BLOCK<LBM_TYPE>::isLocalZ(idx z) const
{
	return z >= offset.z() && z < offset.z() + local.z();
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::defineWall(idx x, idx y, idx z, bool value)
{
//	if (x>0 && x<X-1 && y > 0 && y<Y-1 && z > 0 && z<Z-1) wall(x,y,z) = value;
//	if (x>=0 && x<=X-1 && y >= 0 && y<=Y-1 && z>=0 && z<=Z-1) wall(x,y,z) = value;
	if (isLocalIndex(x, y, z)) wall(x, y, z) = value;
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::setBoundaryX(idx x, map_t value)
{
	if (isLocalX(x))
		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		{
			map(x, y, z) = value;
			// undef wall (bool array), otherwise projectWall() would reset it back to GEO_WALL
			if (value != LBM_TYPE::BC::GEO_WALL)
				wall(x, y, z) = false;
		}
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::setBoundaryY(idx y, map_t value)
{
	if (isLocalY(y))
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		{
			map(x, y, z) = value;
			// undef wall (bool array), otherwise projectWall() would reset it back to GEO_WALL
			if (value != LBM_TYPE::BC::GEO_WALL)
				wall(x, y, z) = false;
		}
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::setBoundaryZ(idx z, map_t value)
{
	if (isLocalZ(z))
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		{
			map(x, y, z) = value;
			// undef wall (bool array), otherwise projectWall() would reset it back to GEO_WALL
			if (value != LBM_TYPE::BC::GEO_WALL)
				wall(x, y, z) = false;
		}
}

template< typename LBM_TYPE >
bool LBM_BLOCK<LBM_TYPE>::getWall(idx x, idx y, idx z) const
{
	if (!isLocalIndex(x, y, z)) return false;
	return wall(x,y,z);
}

template< typename LBM_TYPE >
bool LBM_BLOCK<LBM_TYPE>::isFluid(idx x, idx y, idx z) const
{
	if (!isLocalIndex(x, y, z)) return false;
	return LBM_TYPE::BC::isFluid(map(x,y,z));
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::projectWall()
{
	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
	for (idx z = offset.y(); z < offset.y() + local.z(); z++)
	for (idx y = offset.z(); y < offset.z() + local.y(); y++)
	{
		if (wall(x, y, z))
			map(x, y, z) = LBM_TYPE::BC::GEO_WALL;
	}
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::resetMap(map_t geo_type)
{
	hmap.setValue(geo_type);
}


template< typename LBM_TYPE >
void  LBM_BLOCK<LBM_TYPE>::copyMapToHost()
{
	hmap = dmap;
}

template< typename LBM_TYPE >
void  LBM_BLOCK<LBM_TYPE>::copyMapToDevice()
{
	dmap = hmap;
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::copyMacroToHost()
{
	hmacro = dmacro;
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::copyMacroToDevice()
{
	dmacro = hmacro;
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::copyDFsToHost(uint8_t dfty)
{
	dlat_view_t df = dfs[0].getView();
	df.bind(data.dfs[dfty]);
	hfs[dfty] = df;
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::copyDFsToDevice(uint8_t dfty)
{
	dlat_view_t df = dfs[0].getView();
	df.bind(data.dfs[dfty]);
	df = hfs[dfty];
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::copyDFsToHost()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
		hfs[dfty] = dfs[dfty];
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::copyDFsToDevice()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
		dfs[dfty] = hfs[dfty];
}

#ifdef HAVE_MPI

template< typename LBM_TYPE >
	template< typename Array >
void LBM_BLOCK<LBM_TYPE>::startDrealArraySynchronization(Array& array, int sync_offset)
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
		// set neighbors (0 = x-direction)
		dreal_sync[i + sync_offset].template setNeighbors< 0 >( neighbour_left, neighbour_right );
		// TODO: make this a general parameter (for now we set an upper bound)
		constexpr int blocks_per_rank = 32;
		dreal_sync[i + sync_offset].setTags(
			left_id < 0 ? -1 :
				(2 * i + 1) * blocks_per_rank * nproc + left_id,   // from left
			left_id < 0 ? -1 :
				(2 * i + 0) * blocks_per_rank * nproc + id,        // to left
			right_id < 0 ? -1 :
				(2 * i + 0) * blocks_per_rank * nproc + right_id,  // from right
			right_id < 0 ? -1 :
				(2 * i + 1) * blocks_per_rank * nproc + id );      // to right
		// rebind just the data pointer
		view.bind(array.getData() + i * data.indexer.getStorageSize());
		// determine sync direction
		TNL::Containers::SyncDirection sync_direction = (is_df) ? df_sync_directions[i] : TNL::Containers::SyncDirection::All;
		#ifdef AA_PATTERN
		// reset shift of the lattice sites
		dreal_sync[i + sync_offset].template setBuffersShift<0>(0);
		if (is_df) {
			if (data.even_iter) {
				// lattice sites for synchronization are not shifted, but DFs have opposite directions
				if (sync_direction == TNL::Containers::SyncDirection::Right)
					sync_direction = TNL::Containers::SyncDirection::Left;
				else if (sync_direction == TNL::Containers::SyncDirection::Left)
					sync_direction = TNL::Containers::SyncDirection::Right;
			}
			else {
				// DFs have canonical directions, but lattice sites for synchronization are shifted
				// (values to be synchronized were written to the neighboring sites)
				dreal_sync[i + sync_offset].template setBuffersShift<0>(1);
			}
		}
		#endif
		// start the synchronization
		dreal_sync[i + sync_offset].synchronizeAsync(view, policy, sync_direction);
	}
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::synchronizeDFsDevice_start(uint8_t dftype)
{
	auto df = dfs[0].getView();
	df.bind(data.dfs[dftype]);
	startDrealArraySynchronization(df, 0);
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::synchronizeDFsDevice(uint8_t dftype)
{
	synchronizeDFsDevice_start(dftype);
	for (int i = 0; i < LBM_TYPE::Q; i++)
		dreal_sync[i].wait();
#ifdef USE_CUDA
	cudaDeviceSynchronize();
	checkCudaDevice;
#endif
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::synchronizeMacroDevice_start()
{
	startDrealArraySynchronization(dmacro, LBM_TYPE::Q);
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::synchronizeMacroDevice()
{
	synchronizeMacroDevice_start();
	for (int i = 0; i < MACRO::N; i++)
		dreal_sync[LBM_TYPE::Q + i].wait();
#ifdef USE_CUDA
	cudaDeviceSynchronize();
	checkCudaDevice;
#endif
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::synchronizeMapDevice_start()
{
	// NOTE: threadpool and async require MPI_THREAD_MULTIPLE which is slow
	constexpr auto policy = std::decay_t<decltype(map_sync)>::AsyncPolicy::deferred;

	// set neighbors (0 = x-direction)
	map_sync.template setNeighbors< 0 >( neighbour_left, neighbour_right );
	map_sync.setTags(
			left_id < 0 ? -1 : nproc + left_id,  // from left
			left_id < 0 ? -1 : id,               // to left
			right_id < 0 ? -1 : right_id,        // from right
			right_id < 0 ? -1 : nproc + id );    // to right
	map_sync.synchronizeAsync(dmap, policy);
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::waitAllCommunication()
{
	for (int i = 0; i < LBM_TYPE::Q; i++)
		dreal_sync[i].wait();
	if (LBM_TYPE::MACRO::use_syncMacro)
		for (int i = 0; i < LBM_TYPE::MACRO::N; i++)
			dreal_sync[LBM_TYPE::Q + i].wait();
}
#endif  // HAVE_MPI

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::computeCPUMacroFromLat()
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
		for (idx x=0; x<local.x(); x++)
		for (idx z=0; z<local.z(); z++)
		for (idx y=0; y<local.y(); y++)
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
void LBM_BLOCK<LBM_TYPE>::allocateHostData()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
	{
		hfs[dfty].setSizes(0, global.x(), global.y(), global.z());
		#ifdef HAVE_MPI
		hfs[dfty].template setDistribution< 1 >(offset.x(), offset.x() + local.x(), MPI_COMM_WORLD);
		hfs[dfty].allocate();
		#endif
	}

	hmap.setSizes(global.x(), global.y(), global.z());
	wall.setSizes(global.x(), global.y(), global.z());
#ifdef HAVE_MPI
	hmap.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), MPI_COMM_WORLD);
	hmap.allocate();
	wall.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), MPI_COMM_WORLD);
	wall.allocate();
#endif
	wall.setValue(false);

	hmacro.setSizes(0, global.x(), global.y(), global.z());
	cpumacro.setSizes(0, global.x(), global.y(), global.z());
#ifdef HAVE_MPI
	hmacro.template setDistribution< 1 >(offset.x(), offset.x() + local.x(), MPI_COMM_WORLD);
	hmacro.allocate();
	cpumacro.template setDistribution< 1 >(offset.x(), offset.x() + local.x(), MPI_COMM_WORLD);
	cpumacro.allocate();
#endif
	hmacro.setValue(0);
	// avoid setting empty array
	if (CPU_MACRO::N)
		cpumacro.setValue(0);
}

template< typename LBM_TYPE >
void LBM_BLOCK<LBM_TYPE>::allocateDeviceData()
{
//#ifdef USE_CUDA
#if 1
	dmap.setSizes(global.x(), global.y(), global.z());
	#ifdef HAVE_MPI
	dmap.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), MPI_COMM_WORLD);
	dmap.allocate();
	#endif

	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
	{
		dfs[dfty].setSizes(0, global.x(), global.y(), global.z());
		#ifdef HAVE_MPI
		dfs[dfty].template setDistribution< 1 >(offset.x(), offset.x() + local.x(), MPI_COMM_WORLD);
		dfs[dfty].allocate();
		#endif
	}

	dmacro.setSizes(0, global.x(), global.y(), global.z());
	#ifdef HAVE_MPI
	dmacro.template setDistribution< 1 >(offset.x(), offset.x() + local.x(), MPI_COMM_WORLD);
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
	template< typename F >
void LBM_BLOCK<LBM_TYPE>::forLocalLatticeSites(F f)
{
	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		f(*this, x, y, z);
}

template< typename LBM_TYPE >
	template< typename F >
void LBM_BLOCK<LBM_TYPE>::forAllLatticeSites(F f)
{
	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset.x() - 1; x <= offset.x() + local.x(); x++)
	for (idx z = offset.z() - 1; z <= offset.z() + local.z(); z++)
	for (idx y = offset.y() - 1; y <= offset.y() + local.y(); y++)
		f(*this, x, y, z);
}

template< typename LBM_TYPE >
	template< typename Output >
void LBM_BLOCK<LBM_TYPE>::writeVTK_3D(lat_t lat, Output&& outputData, const char* filename, real time, int cycle) const
{
	VTKWriter vtk;

	FILE* fp = fopen(filename, "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)local.x(), (int)local.y(), (int)local.z());
	fprintf(fp,"X_COORDINATES %d float\n", (int)local.x());
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeFloat(fp, lat.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)local.y());
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		vtk.writeFloat(fp, lat.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)local.z());
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		vtk.writeFloat(fp, lat.lbm2physZ(z));
	vtk.writeBuffer(fp);

	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(local.x()*local.y()*local.z()));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeInt(fp, map(x,y,z));

	char idd[500];
	real value;
	int dofs;
	int count=0, index=0;
	while (outputData(*this, index++, 0, idd, offset.x(), offset.y(), offset.z(), value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		}
		else
			fprintf(fp,"VECTORS %s float\n",idd);

		for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(*this, index-1, dof, idd, x, y, z, value, dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
		count++;
	}

	fclose(fp);
}

template< typename LBM_TYPE >
	template< typename Output >
void LBM_BLOCK<LBM_TYPE>::writeVTK_3Dcut(lat_t lat, Output&& outputData, const char* filename, real time, int cycle, idx ox, idx oy, idx oz, idx lx, idx ly, idx lz, idx step) const
{
	if (!isLocalIndex(ox, oy, oz)) return;

	VTKWriter vtk;

	// intersection of the local domain with the box
	lx = MIN(ox + lx, offset.x() + local.x()) - MAX(ox, offset.x());
	ly = MIN(oy + ly, offset.y() + local.y()) - MAX(oy, offset.y());
	lz = MIN(oz + lz, offset.z() + local.z()) - MAX(oz, offset.z());
	ox = MAX(ox, offset.x());
	oy = MAX(oy, offset.y());
	oz = MAX(oz, offset.z());

	// box dimensions (round-up integer division)
	idx X = lx / step + (lx % step != 0);
	idx Y = ly / step + (ly % step != 0);
	idx Z = lz / step + (lz % step != 0);

	FILE* fp = fopen(filename, "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)X, (int)Y, (int)Z);
	fprintf(fp,"X_COORDINATES %d float\n", (int)X);
	for (idx x = ox; x < ox + lx; x += step)
		vtk.writeFloat(fp, lat.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)Y);
	for (idx y = oy; y < oy + ly; y += step)
		vtk.writeFloat(fp, lat.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)Z);
	for (idx z = oz; z < oz + lz; z += step)
		vtk.writeFloat(fp, lat.lbm2physZ(z));
	vtk.writeBuffer(fp);

	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(X*Y*Z));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	for (idx z = oz; z < oz + lz; z += step)
	for (idx y = oy; y < oy + ly; y += step)
	for (idx x = ox; x < ox + lx; x += step)
		vtk.writeInt(fp, map(x,y,z));

	char idd[500];
	real value;
	int dofs;
	int count=0, index=0;
	while (outputData(*this, index++, 0, idd, ox, oy, oz, value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		}
		else
			fprintf(fp,"VECTORS %s float\n",idd);

		for (idx z = oz; z < oz + lz; z += step)
		for (idx y = oy; y < oy + ly; y += step)
		for (idx x = ox; x < ox + lx; x += step)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(*this, index-1, dof, idd, x, y, z, value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
		count++;
	}

	fclose(fp);
}

template< typename LBM_TYPE >
	template< typename Output >
void LBM_BLOCK<LBM_TYPE>::writeVTK_2DcutX(lat_t lat, Output&& outputData, const char* name, real time, int cycle, idx XPOS) const
{
	if (!isLocalX(XPOS)) return;

	VTKWriter vtk;

	FILE* fp = fopen(name, "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n",1, (int)local.y(), (int)local.z());

	fprintf(fp,"X_COORDINATES %d float\n", 1);
	vtk.writeFloat(fp, lat.lbm2physX(XPOS));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)local.y());
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		vtk.writeFloat(fp, lat.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)local.z());
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		vtk.writeFloat(fp, lat.lbm2physZ(z));
	vtk.writeBuffer(fp);


	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(1*local.y()*local.z()));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	idx x=XPOS;
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		vtk.writeInt(fp, map(x,y,z));

	int count=0, index=0;
	char idd[500];
	real value;
	int dofs;
	while (outputData(*this, index++, 0, idd, offset.x(),offset.y(),offset.z(), value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		} else
		{
			fprintf(fp,"VECTORS %s float\n",idd);
		}
		for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(*this, index-1,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
		count++;
	}

	fclose(fp);
}

template< typename LBM_TYPE >
	template< typename Output >
void LBM_BLOCK<LBM_TYPE>::writeVTK_2DcutY(lat_t lat, Output&& outputData, const char* name, real time, int cycle, idx YPOS) const
{
	if (!isLocalY(YPOS)) return;

	VTKWriter vtk;

	FILE* fp = fopen(name, "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)local.x(), 1, (int)local.z());
	fprintf(fp,"X_COORDINATES %d float\n", (int)local.x());
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeFloat(fp, lat.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES 1 float\n");
	vtk.writeFloat(fp, lat.lbm2physY(YPOS));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)local.z());
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		vtk.writeFloat(fp, lat.lbm2physZ(z));
	vtk.writeBuffer(fp);


	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(1*local.x()*local.z()));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	idx y=YPOS;
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeInt(fp, map(x,y,z));

	int count=0, index=0;
	char idd[500];
	real value;
	int dofs;
	while (outputData(*this, index++, 0, idd, offset.x(),offset.y(),offset.z(), value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		} else
			fprintf(fp,"VECTORS %s float\n",idd);

		for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(*this, index-1,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
		count++;

	}

	fclose(fp);
}

template< typename LBM_TYPE >
	template< typename Output >
void LBM_BLOCK<LBM_TYPE>::writeVTK_2DcutZ(lat_t lat, Output&& outputData, const char* name, real time, int cycle, idx ZPOS) const
{
	if (!isLocalZ(ZPOS)) return;

	VTKWriter vtk;

	FILE* fp = fopen(name, "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)local.x(), (int)local.y(), 1);
	fprintf(fp,"X_COORDINATES %d float\n", (int)local.x());
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeFloat(fp, lat.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)local.y());
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		vtk.writeFloat(fp, lat.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", 1);
	vtk.writeFloat(fp, lat.lbm2physZ(ZPOS));
	vtk.writeBuffer(fp);

	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(1*local.x()*local.y()));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	idx z=ZPOS;
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeInt(fp, map(x,y,z));

	int count=0, index=0;
	char idd[500];
	real value;
	int dofs;
	while (outputData(*this, index++, 0, idd, offset.x(),offset.y(),offset.z(), value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		} else
			fprintf(fp,"VECTORS %s float\n",idd);

		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(*this, index-1,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
		count++;
	}

	fclose(fp);
}
