#include "state.h"

#include "fileutils.h"
#include "timeutils.h"

template< typename LBM_TYPE >
int State<LBM_TYPE>::addLagrange3D()
{
	char dir[FILENAME_CHARS];
	sprintf(dir,"results_%s",id);
	FF.emplace_back(lbm, dir);
	return FF.size()-1;
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::computeAllLagrangeForces()
{
	for (int i=0;i<FF.size();i++)
		if (FF[i].implicitWuShuForcing)
			FF[i].computeWuShuForcesSparse(lbm.physTime());
}

template< typename LBM_TYPE >
bool State<LBM_TYPE>::getPNGdimensions(const char * filename, int &w, int &h)
{
	if (!fileExists(filename)) { printf("file %s does not exist\n",filename); return false; }
	FILE *fp = fopen(filename, "rb");

	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(!png) { printf("file %s png read error\n",filename); return false; }

	png_infop info = png_create_info_struct(png);
	if(!png) { printf("file %s png read error\n",filename); return false; }

	if(setjmp(png_jmpbuf(png))) { printf("file %s png read error\n",filename); return false; }

	png_init_io(png, fp);

	png_read_info(png, info);

	w = png_get_image_width(png, info);
	h = png_get_image_height(png, info);
	//  color_type = png_get_color_type(png, info);
	//  bit_depth  = png_get_bit_depth(png, info);
	fclose(fp);
	if (w>0 && h>0) return true;
	return false;
}


template< typename LBM_TYPE >
template< typename... ARGS >
void State<LBM_TYPE>::log(const char* fmt, ARGS... args)
{
	char dir[FILENAME_CHARS];
	sprintf(dir,"results_%s",id);
	mkdir(dir,0777);
	char fname[FILENAME_CHARS];
	sprintf(fname,"%s/log_rank%03d",dir,lbm.rank);

	FILE*f = fopen(fname,"at"); // append information
	if (f==0)
	{
		printf("unable to create/access file %s",fname);
		return;
	}
	// insert time stamp
	char tname[FILENAME_CHARS];
	timestamp(tname);
	fprintf(f, "%s ", tname);
	fprintf(f,fmt, args...);
	fprintf(f,"\n");
	fclose(f);

	printf(fmt, args...);
	printf("\n");

}

/// outputs information into log file "type"
template< typename LBM_TYPE >
template< typename... ARGS >
void State<LBM_TYPE>::setid(const char* fmt, ARGS... args)
{
	sprintf(id, fmt, args...);
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::flagCreate(const char*flagname)
{
	if (lbm.rank != 0) return;

	char fname[FILENAME_CHARS];
	sprintf(fname,"results_%s/%s",id,flagname);
	create_file(fname);

	FILE*f = fopen(fname,"at"); // append information
	if (f==0)
	{
		printf("unable to create/access file %s",fname);
		return;
	}
	// insert time stamp
	char tname[FILENAME_CHARS];
	timestamp(tname);
	fprintf(f, "%s\n", tname);
	fclose(f);
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::flagDelete(const char*flagname)
{
	if (lbm.rank != 0) return;

	char fname[FILENAME_CHARS];
	sprintf(fname,"results_%s/%s",id,flagname);
	if (fileExists(fname)) remove(fname);
}

template< typename LBM_TYPE >
bool State<LBM_TYPE>::flagExists(const char*flagname)
{
	char fname[FILENAME_CHARS];
	sprintf(fname,"results_%s/%s",id,flagname);
	return fileExists(fname);
}

template< typename LBM_TYPE >
template< typename... ARGS >
void State<LBM_TYPE>::mark(const char* fmt, ARGS... args)
{
	if (lbm.rank != 0) return;

	char fname[FILENAME_CHARS];
	sprintf(fname,"results_%s/mark",id);
	create_file(fname);

	FILE*f = fopen(fname,"at"); // append information
	if (f==0)
	{
		printf("unable to create/access file %s",fname);
		return;
	}
	// insert time stamp
	char tname[FILENAME_CHARS];
	timestamp(tname);
	fprintf(f, "%s ", tname);
	fprintf(f,fmt, args...);
	fprintf(f,"\n");
	fclose(f);
}

/// checks/creates mark and return status
template< typename LBM_TYPE >
bool State<LBM_TYPE>::isMark()
{
	bool result;
	if (lbm.rank == 0)
	{
		char fname[FILENAME_CHARS];
		sprintf(fname,"results_%s/mark", id);
		if (fileExists(fname))
		{
			log("Mark %s already exists.",fname);
			result = true;
		}
		else
		{
			log("Mark %s does not exist. Creating new mark.",fname);
			mark("");
			result = false;
		}
	}
	TNL::MPI::Bcast(&result, 1, 0, TNL::MPI::AllGroup());
	return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK SURFACE
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTK_Surface(const char* name, real time, int cycle, Lagrange3D &fil)
{
	VTKWriter vtk;

	char fname[FILENAME_CHARS];
	sprintf(fname,"results_%s/vtk3D/rank%03d_%s.vtk",id,lbm.rank,name);
	create_file(fname);

	FILE* fp = fopen(fname, "w+");
	vtk.writeHeader(fp);

	fprintf(fp, "DATASET POLYDATA\n");

	fprintf(fp, "POINTS %d float\n", fil.LL.size());
	for (int i=0;i<fil.LL.size();i++)
	{
		vtk.writeFloat(fp, fil.LL[i].x);
		vtk.writeFloat(fp, fil.LL[i].y);
		vtk.writeFloat(fp, fil.LL[i].z);
	}
	vtk.writeBuffer(fp);

	fprintf(fp, "POLYGONS %d %d\n", (fil.lag_X-1)*fil.lag_Y , 5*(fil.lag_X-1)*fil.lag_Y ); // first number: number of polygons, second number: total integers describing the list
	for (int i=0;i<fil.lag_X-1;i++)
	for (int j=0;j<fil.lag_Y;j++)
	{
		int ip = i+1;
		int jp = (j==fil.lag_Y-1) ? 0 : j+1;
		vtk.writeInt(fp,4);
		vtk.writeInt(fp,fil.findIndex(i,j));
		vtk.writeInt(fp,fil.findIndex(ip,j));
		vtk.writeInt(fp,fil.findIndex(ip,jp));
		vtk.writeInt(fp,fil.findIndex(i,jp));
	}
	fclose(fp);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK POINTS
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTK_Points(const char* name, real time, int cycle, Lagrange3D &fil)
{
	VTKWriter vtk;

	char fname[FILENAME_CHARS];
	sprintf(fname,"results_%s/vtk3D/rank%03d_%s.vtk",id,lbm.rank,name);
	create_file(fname);

	FILE* fp = fopen(fname, "w+");
	vtk.writeHeader(fp);

	fprintf(fp, "DATASET POLYDATA\n");

	fprintf(fp, "POINTS %d float\n", fil.LL.size());
	for (int i=0;i<fil.LL.size();i++)
	{
		vtk.writeFloat(fp, fil.LL[i].x);
		vtk.writeFloat(fp, fil.LL[i].y);
		vtk.writeFloat(fp, fil.LL[i].z);
	}
	vtk.writeBuffer(fp);
	fclose(fp);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK 1D CUT
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename LBM_TYPE >
template< typename... ARGS >
void State<LBM_TYPE>::add1Dcut(real fromx, real fromy, real fromz, real tox, real toy, real toz, const char* fmt, ARGS... args)
{
	if (lbm.rank > 0) log("TODO: add1Dcut is not implemented for MPI.");

	probe1Dlinevec.push_back( T_PROBE1DLINECUT() );
	int last = probe1Dlinevec.size()-1;
	sprintf(probe1Dlinevec[last].name, fmt, args...);
	probe1Dlinevec[last].from[0] = fromx;
	probe1Dlinevec[last].from[1] = fromy;
	probe1Dlinevec[last].from[2] = fromz;

	probe1Dlinevec[last].to[0] = tox;
	probe1Dlinevec[last].to[1] = toy;
	probe1Dlinevec[last].to[2] = toz;

	probe1Dlinevec[last].cycle = 0;
}

template< typename LBM_TYPE >
template< typename... ARGS >
void State<LBM_TYPE>::add1Dcut_X(real y, real z, const char* fmt, ARGS... args)
{
	if (!lbm.isLocalY(lbm.phys2lbmY(y)) || !lbm.isLocalZ(lbm.phys2lbmZ(z))) return;

	probe1Dvec.push_back( T_PROBE1DCUT() );
	int last = probe1Dvec.size()-1;
	sprintf(probe1Dvec[last].name, fmt, args...);
	probe1Dvec[last].type = 0;
	probe1Dvec[last].pos1 = lbm.phys2lbmY(y);
	probe1Dvec[last].pos2 = lbm.phys2lbmZ(z);
	probe1Dvec[last].cycle = 0;
}

template< typename LBM_TYPE >
template< typename... ARGS >
void State<LBM_TYPE>::add1Dcut_Y(real x, real z, const char* fmt, ARGS... args)
{
	if (!lbm.isLocalX(lbm.phys2lbmX(x)) || !lbm.isLocalZ(lbm.phys2lbmZ(z))) return;

	probe1Dvec.push_back( T_PROBE1DCUT() );
	int last = probe1Dvec.size()-1;
	sprintf(probe1Dvec[last].name, fmt, args...);
	probe1Dvec[last].type = 1;
	probe1Dvec[last].pos1 = lbm.phys2lbmX(x);
	probe1Dvec[last].pos2 = lbm.phys2lbmZ(z);
	probe1Dvec[last].cycle = 0;
}

template< typename LBM_TYPE >
template< typename... ARGS >
void State<LBM_TYPE>::add1Dcut_Z(real x, real y, const char* fmt, ARGS... args)
{
	if (!lbm.isLocalX(lbm.phys2lbmX(x)) || !lbm.isLocalY(lbm.phys2lbmY(y))) return;

	probe1Dvec.push_back( T_PROBE1DCUT() );
	int last = probe1Dvec.size()-1;
	sprintf(probe1Dvec[last].name, fmt, args...);
	probe1Dvec[last].type = 2;
	probe1Dvec[last].pos1 = lbm.phys2lbmX(x);
	probe1Dvec[last].pos2 = lbm.phys2lbmY(y);
	probe1Dvec[last].cycle = 0;
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTKs_1D()
{
	if (probe1Dvec.size()>0)
	{
		// browse all 1D probeline cuts
		for (int i=0;i<probe1Dvec.size(); i++)
		{
			char fname[FILENAME_CHARS];
//			sprintf(fname,"results_%s/probes1D/%s_%06d_t%f", id, probe1Dvec[i].name, probe1Dvec[i].cycle, lbm.physTime());
			sprintf(fname,"results_%s/probes1D/%s_rank%03d_%06d", id, probe1Dvec[i].name, lbm.rank, probe1Dvec[i].cycle);
			// create parent directories
			create_file(fname);
//			probeLine(probe1Dvec[i].from[0],probe1Dvec[i].from[1],probe1Dvec[i].from[2],probe1Dvec[i].to[0],probe1Dvec[i].to[1],probe1Dvec[i].to[2],fname);
			switch (probe1Dvec[i].type)
			{
				case 0: write1Dcut_X(probe1Dvec[i].pos1, probe1Dvec[i].pos2, fname);
					break;
				case 1: write1Dcut_Y(probe1Dvec[i].pos1, probe1Dvec[i].pos2, fname);
					break;
				case 2: write1Dcut_Z(probe1Dvec[i].pos1, probe1Dvec[i].pos2, fname);
					break;
			}
			probe1Dvec[i].cycle++;
		}
	}

	// browse all 1D probe cuts
	for (int i=0;i<probe1Dlinevec.size(); i++)
	{
		char fname[FILENAME_CHARS];
//		sprintf(fname,"results_%s/probes1D/%s_%06d_t%f", id, probe1Dvec[i].name, probe1Dvec[i].cycle, lbm.physTime());
		sprintf(fname,"results_%s/probes1D/%s_rank%03d_%06d", id, probe1Dlinevec[i].name, lbm.rank, probe1Dlinevec[i].cycle);
		// create parent directories
		create_file(fname);
		write1Dcut(probe1Dlinevec[i].from[0],probe1Dlinevec[i].from[1],probe1Dlinevec[i].from[2],probe1Dlinevec[i].to[0],probe1Dlinevec[i].to[1],probe1Dlinevec[i].to[2],fname);
		probe1Dlinevec[i].cycle++;
	}
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::write1Dcut_X(idx y, idx z, const char * fname)
{
	FILE*fout = fopen(fname,"wt"); // append information
	log("[probe %s]",fname);
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout,"#time %f s\n", lbm.physTime());
	fprintf(fout,"#1:x");
	int count=2, index=0;
	while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (idx i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,(int)i);
	}
	fprintf(fout,"\n");

	for (idx i = lbm.offset_X; i < lbm.offset_X + lbm.local_X; i++)
	{
		if (lbm.isFluid(i,y,z))
		{
			fprintf(fout, "%e", lbm.lbm2physX(i));
			index=0;
			if (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
			{
				for (int dof=0;dof<dofs;dof++)
				{
					outputData(index-1,dof,idd,i,y,z,value,dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	}
	fclose(fout);
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::write1Dcut_Y(idx x, idx z, const char * fname)
{
	FILE*fout = fopen(fname,"wt"); // append information
	log("[probe %s]",fname);
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout,"#time %f s\n", lbm.physTime());
	fprintf(fout,"#1:y");
	int count=2, index=0;
	while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (idx i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,(int)i);
	}
	fprintf(fout,"\n");

	for (idx i=0;i<lbm.local_Y;i++)
	{
		if (lbm.isFluid(x,i,z))
		{
			fprintf(fout, "%e", lbm.lbm2physY(i));
			int index=0;
			while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
			{
				for (int dof=0;dof<dofs;dof++)
				{
					outputData(index-1,dof,idd,x,i,z,value,dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	}
	fclose(fout);
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::write1Dcut_Z(idx x, idx y, const char * fname)
{
	FILE*fout = fopen(fname,"wt"); // append information
	log("[probe %s]",fname);
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout,"#time %f s\n", lbm.physTime());
	fprintf(fout,"#1:z");
	int count=2, index=0;
	while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (idx i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,(int)i);
	}
	fprintf(fout,"\n");

	for (idx i = lbm.offset_Z; i < lbm.offset_Z + lbm.local_Z; i++)
	{
		if (lbm.isFluid(x,y,i))
		{
			fprintf(fout, "%e", lbm.lbm2physZ(i));
			index=0;
			while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
			{
				for (int dof=0;dof<dofs;dof++)
				{
					outputData(index-1,dof,idd,x,y,i,value,dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	}
	fclose(fout);
}

// line projection from[3] to[3]
template< typename LBM_TYPE >
void State<LBM_TYPE>::write1Dcut(real fromx, real fromy, real fromz, real tox, real toy, real toz, const char * fname)
{
	if (lbm.rank > 0) log("TODO: write1Dcut is not implemented for MPI.");

//		char dir[FILENAME_CHARS];
//		sprintf(dir,"results_%s/probes1D",id);
//		mkdir_p(dir,0777);
//		char fname[FILENAME_CHARS];
//		sprintf(fname,"%s/%s_it%08d_t%f",dir,desc,lbm.iterations,lbm.physTime());
	FILE*fout = fopen(fname,"wt"); // append information
	log("[probe %s]",fname);
	// probe vertical profile at x_m
	real i[3],f[3],p[3];
	i[0]=fromx/lbm.physDl;
	i[1]=fromy/lbm.physDl;
	i[2]=fromz/lbm.physDl;
	f[0]=tox/lbm.physDl;
	f[1]=toy/lbm.physDl;
	f[2]=toz/lbm.physDl;
	real dist = NORM(i[0]-f[0],i[1]-f[1],i[2]-f[2]);
	real ds = 1.0/(dist*2.0); // rozliseni najit
	// special case: sampling along an axis
	if( (i[0] == f[0] && i[1] == f[1]) ||
		(i[1] == f[1] && i[2] == f[2]) ||
		(i[0] == f[0] && i[2] == f[2]) )
		ds = 1.0/dist;

	char idd[500];
	real value;
	int dofs;
	fprintf(fout,"#time %f s\n", lbm.physTime());
	fprintf(fout,"#1:rel_pos");

	int count=2, index=0;
	while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (int i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,i);
	}
	fprintf(fout,"\n");

	for (real s=0;s<=1.0;s+=ds)
	{
		for (int k=0;k<3;k++) p[k] = i[k] + s*(f[k]-i[k]);
		// FIXME: isFluid checks idx, not real !!!
		if (lbm.isFluid(p[0],p[1],p[2]))
		{
			fprintf(fout, "%e", (s*dist-0.5)*lbm.physDl);
			index=0;
			while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
			{
				for (int dof=0;dof<dofs;dof++)
				{
					outputData(index-1,dof,idd,p[0],p[1],p[2],value,dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	}
	fclose(fout);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK 3D
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTKs_3D()
{
	// only one 3D vtk is written now
	char dir[FILENAME_CHARS], filename[FILENAME_CHARS], basename[FILENAME_CHARS];
	sprintf(dir,"results_%s/vtk3D", id);
//	sprintf(dir,"results_%s/vtk_%s",id,State::T_LBM_TYPE::id);
	mkdir_p(dir,0755);

	char tmp_dirname[FILENAME_CHARS];
	const char* local_scratch = getenv("LOCAL_SCRATCH");
	if (!local_scratch || local_scratch[0] == '\0')
	{
		// $LOCAL_SCRATCH is not defined or empty - default to regular subdirectory in results_*
		sprintf(tmp_dirname, dir);
		local_scratch = NULL;
	}
	else
	{
		// Write files temporarily into the local scratch and move them to final_dirname at
		// the end, after all MPI processes have completed. This avoids small buffered writes
		// into the shared filesystem on clusters as well as corruption of previous state due
		// to MPI errors...
		sprintf(tmp_dirname, "%s/%s", local_scratch, dir);
	}
	mkdir_p(tmp_dirname,0755);

//	int vtk3Dstyle = vtk3DsingleFile; // enum { vtk3DsingleFile, vtk3DmanyFiles, vtk3DmanyFilesExtraHeader };
	if (vtk3Dstyle == vtk3DsingleFile)
	{
		sprintf(basename,"rank%03d_data_%d.vtk", lbm.rank, cnt[VTK3D].count);
		sprintf(filename,"%s/%s", tmp_dirname, basename);
		writeVTK_3D_singlefile(filename,lbm.physTime(),cnt[VTK3D].count);
	} else
	{
		sprintf(basename,"rank%03d_", lbm.rank);//, cnt[VTK3D].count);
		sprintf(filename,"%s/%s", tmp_dirname, basename);
		writeVTK_3D(filename,lbm.physTime(),cnt[VTK3D].count);
	}

	if (local_scratch)
	{
		// move the files from local_scratch into final_dirname and create a backup of the existing files
		move(tmp_dirname, dir, basename, basename);
	}
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTK_3D(const char* name, real time, int cycle)
{
	bool VTK3D_write_separate_header=(vtk3Dstyle==vtk3DmanyFiles) ? false : true;
	// determine max objects to write
	int max_objects=0;
	{
		int index=0;
		char idd[500];
		real value;
		int dofs;
		while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
			max_objects++;
		log("[vtk preparing to write %d objects]",max_objects);
	}

//	state.log("number of CPU cores:\t%d", omp_get_num_procs());
//	omp_set_num_threads(num_cpu_threads);
	int max_thr = MIN(max_objects, omp_get_num_procs());
	log("[vtk3d %d threads for vtk 3D write]", max_thr);

	if (VTK3D_write_separate_header)
	{
		VTKWriter vtk; // local vtkwriter
		char fname[500];
		sprintf(fname,"%sheader_%d.prevtk",name,cycle);
		// browse output
		FILE* fp = fopen(fname, "w+");
		vtk.writeHeader(fp);
		fprintf(fp,"DATASET RECTILINEAR_GRID\n");
		fprintf(fp,"DIMENSIONS %d %d %d\n", (int)lbm.local_X, (int)lbm.local_Y, (int)lbm.local_Z);
		fprintf(fp,"X_COORDINATES %d float\n", (int)lbm.local_X);
		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
			vtk.writeFloat(fp, lbm.lbm2physX(x));
		vtk.writeBuffer(fp);

		fprintf(fp,"Y_COORDINATES %d float\n", (int)lbm.local_Y);
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
			vtk.writeFloat(fp, lbm.lbm2physY(y));
		vtk.writeBuffer(fp);

		fprintf(fp,"Z_COORDINATES %d float\n", (int)lbm.local_Z);
		for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
			vtk.writeFloat(fp, lbm.lbm2physZ(z));
		vtk.writeBuffer(fp);

		fprintf(fp,"FIELD FieldData %d\n",2);
		fprintf(fp,"TIME %d %d float\n",1,1);
		vtk.writeFloat(fp, time);
		vtk.writeBuffer(fp);

		fprintf(fp,"CYCLE %d %d float\n",1,1);
		vtk.writeFloat(fp, cycle);
		vtk.writeBuffer(fp);

		fprintf(fp,"POINT_DATA %d\n", (int)(lbm.local_X*lbm.local_Y*lbm.local_Z));

		fprintf(fp,"SCALARS wall int 1\n");
		fprintf(fp,"LOOKUP_TABLE default\n");
		for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
			vtk.writeInt(fp, lbm.map(x,y,z));

		fclose(fp);
	}

	#pragma omp parallel for schedule(static) num_threads(max_thr)
	for (int i=0;i<max_objects;i++)
	{
		VTKWriter vtk; // local vtkwriter
		char idd[500];
		real value;
		int dofs;
//		int index=0;
		char fname[500];
		outputData(i, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs); // read idd and dofs
//		{
		sprintf(fname,"%s%s_%d.%s",name,idd,cycle,(VTK3D_write_separate_header)?"prevtk":"vtk");
		log("[vtk3d: writing %s, time %f, cycle %d] ", fname, time, cycle);
		// browse output
		FILE* fp = fopen(fname, "w+");
		if (!VTK3D_write_separate_header)
		{
			vtk.writeHeader(fp);
			fprintf(fp,"DATASET RECTILINEAR_GRID\n");
			fprintf(fp,"DIMENSIONS %d %d %d\n", (int)lbm.local_X, (int)lbm.local_Y, (int)lbm.local_Z);
			fprintf(fp,"X_COORDINATES %d float\n", (int)lbm.local_X);
			for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
				vtk.writeFloat(fp, lbm.lbm2physX(x));
			vtk.writeBuffer(fp);

			fprintf(fp,"Y_COORDINATES %d float\n", (int)lbm.local_Y);
			for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
				vtk.writeFloat(fp, lbm.lbm2physY(y));
			vtk.writeBuffer(fp);

			fprintf(fp,"Z_COORDINATES %d float\n", (int)lbm.local_Z);
			for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
				vtk.writeFloat(fp, lbm.lbm2physZ(z));
			vtk.writeBuffer(fp);

			fprintf(fp,"FIELD FieldData %d\n",2);
			fprintf(fp,"TIME %d %d float\n",1,1);
			vtk.writeFloat(fp, time);
			vtk.writeBuffer(fp);

			fprintf(fp,"CYCLE %d %d float\n",1,1);
			vtk.writeFloat(fp, cycle);
			vtk.writeBuffer(fp);

			fprintf(fp,"POINT_DATA %d\n", (int)(lbm.local_X*lbm.local_Y*lbm.local_Z));

			fprintf(fp,"SCALARS wall int 1\n");
			fprintf(fp,"LOOKUP_TABLE default\n");
			for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
			for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
			for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
				vtk.writeInt(fp, lbm.map(x,y,z));
		}

		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		}
		else
		{
			fprintf(fp,"VECTORS %s float\n",idd);
		}

		for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(i,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
		fclose(fp);
//		log("[vtk3d %s written, time %f, cycle %d] ", fname, time, cycle);
	}
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTK_3D_singlefile(const char* name, real time, int cycle)
{
	VTKWriter vtk;

	FILE* fp = fopen(name, "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)lbm.local_X, (int)lbm.local_Y, (int)lbm.local_Z);
	fprintf(fp,"X_COORDINATES %d float\n", (int)lbm.local_X);
	for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		vtk.writeFloat(fp, lbm.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)lbm.local_Y);
	for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		vtk.writeFloat(fp, lbm.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)lbm.local_Z);
	for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
		vtk.writeFloat(fp, lbm.lbm2physZ(z));
	vtk.writeBuffer(fp);

	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(lbm.local_X*lbm.local_Y*lbm.local_Z));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
	for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
	for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		vtk.writeInt(fp, lbm.map(x,y,z));

	char idd[500];
	real value;
	int dofs;
	int count=0, index=0;
	while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		}
		else
			fprintf(fp,"VECTORS %s float\n",idd);

		for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(index-1,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
		count++;
	}

	fclose(fp);
	log("[vtk %s written, time %f, cycle %d] ", name, time, cycle);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK 3D CUT
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTK_3Dcut(const char* name, real time, int cycle, idx ox, idx oy, idx oz, idx lx, idx ly, idx lz, idx step)
{
	VTKWriter vtk;

	// intersection of the local domain with the box
	lx = MIN(ox + lx, lbm.offset_X + lbm.local_X) - MAX(ox, lbm.offset_X);
	ly = MIN(oy + ly, lbm.offset_Y + lbm.local_Y) - MAX(oy, lbm.offset_Y);
	lz = MIN(oz + lz, lbm.offset_Z + lbm.local_Z) - MAX(oz, lbm.offset_Z);
	ox = MAX(ox, lbm.offset_X);
	oy = MAX(oy, lbm.offset_Y);
	oz = MAX(oz, lbm.offset_Z);

	// box dimensions (round-up integer division)
	idx X = lx / step + (lx % step != 0);
	idx Y = ly / step + (ly % step != 0);
	idx Z = lz / step + (lz % step != 0);
//	log("debug: writeVTK3Dcut X %d Y %d Z %d",(int)X,(int)Y,(int)Z);

	FILE* fp = fopen(name, "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)X, (int)Y, (int)Z);
	fprintf(fp,"X_COORDINATES %d float\n", (int)X);
	for (idx x = ox; x < ox + lx; x += step)
		vtk.writeFloat(fp, lbm.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)Y);
	for (idx y = oy; y < oy + ly; y += step)
		vtk.writeFloat(fp, lbm.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)Z);
	for (idx z = oz; z < oz + lz; z += step)
		vtk.writeFloat(fp, lbm.lbm2physZ(z));
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
		vtk.writeInt(fp, lbm.map(x,y,z));

	char idd[500];
	real value;
	int dofs;
	int count=0, index=0;
	while (outputData(index++, 0, idd, ox,oy,oz, value, dofs))
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
				outputData(index-1,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
		count++;
	}

	fclose(fp);
	log("[vtk %s written, time %f, cycle %d] ", name, time, cycle);
}

template< typename LBM_TYPE >
template< typename... ARGS >
void State<LBM_TYPE>::add3Dcut(idx ox, idx oy, idx oz, idx lx, idx ly, idx lz, idx step, const char* fmt, ARGS... args)
{
	probe3Dvec.push_back( T_PROBE3DCUT() );
	int last = probe3Dvec.size()-1;

	sprintf(probe3Dvec[last].name, fmt, args...);

	probe3Dvec[last].ox = ox;
	probe3Dvec[last].oy = oy;
	probe3Dvec[last].oz = oz;
	probe3Dvec[last].lx = lx;
	probe3Dvec[last].ly = ly;
	probe3Dvec[last].lz = lz;
	probe3Dvec[last].step = step;
	probe3Dvec[last].cycle = 0;
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTKs_3Dcut()
{
	if (probe3Dvec.size()<=0) return;
	// browse all 3D vtk cuts
	for (int i=0;i<probe3Dvec.size(); i++)
	{
		char fname[FILENAME_CHARS];
		sprintf(fname,"results_%s/vtk3Dcut/%s_rank%03d_%d.vtk", id, probe3Dvec[i].name, lbm.rank, probe3Dvec[i].cycle);
		// create parent directories
		create_file(fname);
		writeVTK_3Dcut(
			fname,
			lbm.physTime(),
			probe3Dvec[i].cycle,
			probe3Dvec[i].ox,
			probe3Dvec[i].oy,
			probe3Dvec[i].oz,
			probe3Dvec[i].lx,
			probe3Dvec[i].ly,
			probe3Dvec[i].lz,
			probe3Dvec[i].step
			);
		probe3Dvec[i].cycle++;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK 2D CUT
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename LBM_TYPE >
template< typename... ARGS >
void State<LBM_TYPE>::add2Dcut_X(idx x, const char* fmt, ARGS... args)
{
	if (!lbm.isLocalX(x)) return;

	probe2Dvec.push_back( T_PROBE2DCUT() );
	int last = probe2Dvec.size()-1;

	sprintf(probe2Dvec[last].name, fmt, args...);

	probe2Dvec[last].type = 0;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = x;
}

template< typename LBM_TYPE >
template< typename... ARGS >
void State<LBM_TYPE>::add2Dcut_Y(idx y, const char* fmt, ARGS... args)
{
	if (!lbm.isLocalY(y)) return;

	probe2Dvec.push_back( T_PROBE2DCUT() );
	int last = probe2Dvec.size()-1;

	sprintf(probe2Dvec[last].name, fmt, args...);

	probe2Dvec[last].type = 1;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = y;
}

template< typename LBM_TYPE >
template< typename... ARGS >
void State<LBM_TYPE>::add2Dcut_Z(idx z, const char* fmt, ARGS... args)
{
	if (!lbm.isLocalZ(z)) return;

	probe2Dvec.push_back( T_PROBE2DCUT() );
	int last = probe2Dvec.size()-1;

	sprintf(probe2Dvec[last].name, fmt, args...);

	probe2Dvec[last].type = 2;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = z;
}


template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTKs_2D()
{
	if (probe2Dvec.size()<=0) return;
	// browse all 2D vtk cuts
	for (int i=0;i<probe2Dvec.size(); i++)
	{
		char fname[FILENAME_CHARS];
		sprintf(fname,"results_%s/vtk2D/%s_rank%03d_%d.vtk", id, probe2Dvec[i].name, lbm.rank, probe2Dvec[i].cycle);
		// create parent directories
		create_file(fname);
		switch (probe2Dvec[i].type)
		{
			case 0: writeVTK_2DcutX(fname,lbm.physTime(),probe2Dvec[i].cycle,probe2Dvec[i].position);
				break;
			case 1: writeVTK_2DcutY(fname,lbm.physTime(),probe2Dvec[i].cycle,probe2Dvec[i].position);
				break;
			case 2: writeVTK_2DcutZ(fname,lbm.physTime(),probe2Dvec[i].cycle,probe2Dvec[i].position);
				break;
		}
		probe2Dvec[i].cycle++;
	}
}

// X-Z plane for Y=YPOS
template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTK_2DcutY(const char* name, real time, int cycle, idx YPOS)
{
	VTKWriter vtk;

	FILE* fp = fopen(name, "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)lbm.local_X, 1, (int)lbm.local_Z);
	fprintf(fp,"X_COORDINATES %d float\n", (int)lbm.local_X);
	for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		vtk.writeFloat(fp, lbm.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES 1 float\n");
	vtk.writeFloat(fp, lbm.lbm2physY(YPOS));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)lbm.local_Z);
	for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
		vtk.writeFloat(fp, lbm.lbm2physZ(z));
	vtk.writeBuffer(fp);


	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(1*lbm.local_X*lbm.local_Z));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	idx y=YPOS;
	for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
	for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		vtk.writeInt(fp, lbm.map(x,y,z));

	int count=0, index=0;
	char idd[500];
	real value;
	int dofs;
	while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		} else
			fprintf(fp,"VECTORS %s float\n",idd);

		for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(index-1,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
		count++;

	}

	fclose(fp);
	log("[vtk %s written, time %f, cycle %d] ", name, time, cycle);
}

// Y-Z plane for X=XPOS
template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTK_2DcutX(const char* name, real time, int cycle, idx XPOS)
{
	VTKWriter vtk;

	FILE* fp = fopen(name, "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n",1, (int)lbm.local_Y, (int)lbm.local_Z);

	fprintf(fp,"X_COORDINATES %d float\n", 1);
	vtk.writeFloat(fp, lbm.lbm2physX(XPOS));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)lbm.local_Y);
	for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		vtk.writeFloat(fp, lbm.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)lbm.local_Z);
	for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
		vtk.writeFloat(fp, lbm.lbm2physZ(z));
	vtk.writeBuffer(fp);


	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(1*lbm.local_Y*lbm.local_Z));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	idx x=XPOS;
	for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
	for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		vtk.writeInt(fp, lbm.map(x,y,z));

	int count=0, index=0;
	char idd[500];
	real value;
	int dofs;
	while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
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
		for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(index-1,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
		count++;
	}

	fclose(fp);
	log("[vtk %s written, time %f, cycle %d] ", name, time, cycle);
}

// X-Y plane for Z=ZPOS
template< typename LBM_TYPE >
void State<LBM_TYPE>::writeVTK_2DcutZ(const char* name, real time, int cycle, idx ZPOS)
{
	VTKWriter vtk;

	FILE* fp = fopen(name, "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)lbm.local_X, (int)lbm.local_Y, 1);
	fprintf(fp,"X_COORDINATES %d float\n", (int)lbm.local_X);
	for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		vtk.writeFloat(fp, lbm.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)lbm.local_Y);
	for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		vtk.writeFloat(fp, lbm.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", 1);
	vtk.writeFloat(fp, lbm.lbm2physZ(ZPOS));
	vtk.writeBuffer(fp);

	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(1*lbm.local_X*lbm.local_Y));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	idx z=ZPOS;
	for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
	for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		vtk.writeInt(fp, lbm.map(x,y,z));

	int count=0, index=0;
	char idd[500];
	real value;
	int dofs;
	while (outputData(index++, 0, idd, lbm.offset_X,lbm.offset_Y,lbm.offset_Z, value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		} else
			fprintf(fp,"VECTORS %s float\n",idd);

		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(index-1,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
		count++;
	}

	fclose(fp);
	log("[vtk %s written, time %f, cycle %d] ", name, time, cycle);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// PNG PROJECTION
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename LBM_TYPE >
bool State<LBM_TYPE>::projectPNG_X(const char * filename, idx x0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (!lbm.isLocalX(x0)) return true;

	if (!fileExists(filename)) { printf("file %s does not exist\n",filename); return false; }
	PNGTool P(filename);

	// plane y-z
	idx x = x0;
	for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
	{
		real a = (real)z/(real)(lbm.global_Z - 1); // a in [0,1]
		a = amin + a * (amax - amin); // a in [amin, amax]
		if (mirror) a = 1.0 - a;
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		{
			real b = (real)y/(real)(lbm.global_Y - 1); // b in [0,1]
			b = bmin + b * (bmax - bmin); // b in [bmin, bmax]
			if (flip) b = 1.0 - b;
			if (rotate)
			{
				if (P.intensity(b,a) > 0) lbm.defineWall(x, y, z, true);
			}
			else
			{
				if (P.intensity(a,b) > 0) lbm.defineWall(x, y, z, true);
			}
		}
	}
	return true;
}

template< typename LBM_TYPE >
bool State<LBM_TYPE>::projectPNG_Y(const char * filename, idx y0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (!lbm.isLocalY(y0)) return true;

	if (!fileExists(filename)) { printf("file %s does not exist\n",filename); return false; }
	PNGTool P(filename);

	// plane x-z
	idx y=y0;
	for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
	{
		real a = (real)z/(real)(lbm.global_Z - 1); // a in [0,1]
		a = amin + a * (amax - amin); // a in [amin, amax]
		if (mirror) a = 1.0 - a;
		for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
		{
			real b = (real)x/(real)(lbm.global_X - 1); // b in [0,1]
			b = bmin + b * (bmax - bmin); // b in [bmin, bmax]
			if (flip) b = 1.0 - b;
			if (rotate)
			{
				if (P.intensity(b,a) > 0) lbm.defineWall(x, y, z, true);
			}
			else
			{
				if (P.intensity(a,b) > 0) lbm.defineWall(x, y, z, true);
			}
		}
	}
	return true;
}


template< typename LBM_TYPE >
bool State<LBM_TYPE>::projectPNG_Z(const char * filename, idx z0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (!lbm.isLocalZ(z0)) return true;

	if (!fileExists(filename)) { printf("file %s does not exist\n",filename); return false; }
	PNGTool P(filename);

	// plane x-y
	idx z=z0;
	for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
	{
		real a = (real)x/(real)(lbm.global_X - 1); // a in [0,1]
		a = amin + a * (amax - amin); // a in [amin, amax]
		if (mirror) a = 1.0 - a;
		for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		{
			real b = (real)y/(real)(lbm.global_Y - 1); // b in [0,1]
			b = bmin + b * (bmax - bmin); // b in [bmin, bmax]
			if (flip) b = 1.0 - b;
			if (rotate)
			{
				if (P.intensity(b,a) > 0) lbm.defineWall(x, y, z, true);
			}
			else
			{
				if (P.intensity(a,b) > 0) lbm.defineWall(x, y, z, true);
			}
		}
	}
	return true;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// SAVE & LOAD STATE
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename LBM_TYPE >
void State<LBM_TYPE>::move(const char* srcdir, const char* dstdir, const char* srcfilename, const char* dstfilename)
{
	char src[FILENAME_CHARS];
	char dst[FILENAME_CHARS];
	sprintf(src, "%s/%s", srcdir, srcfilename);
	sprintf(dst, "%s/%s", dstdir, dstfilename);

	// rename works only on the same filesystem
	if (rename(src, dst) == 0)
	{
		log("renamed %s to %s", src, dst);
		return;
	}
	if (errno != EXDEV)
	{
		perror("move: something went wrong!!!");
		return;
	}

	// manual copy data and meta data
	if (move_file(src, dst) == 0)
		log("moved %s to %s", src, dst);
	else
		log("move: manual move failed");
}

//template< typename LBM_TYPE >
//template< typename... ARGS >
//int State<LBM_TYPE>::saveLoadTextData(int direction, const char*dirname, const char*filename, const char*fmt, ARGS&... args)
//{
//	// check if main dir exists
//	mkdir_p(dirname, 0777);
//	char fname[FILENAME_CHARS];
//	sprintf(fname,"%s/%s_rank%03d", dirname, filename, lbm.rank);
//
//	if (direction==MemoryToFile)
//	{
//		FILE*f = fopen(fname,"wt");
//		if (f==0)
//		{
//			log("unable to create file %s",fname);
//			return 0;
//		}
//		fprintf(f,fmt, args...);
//		fclose(f);
//		log("[saveLoadTextData: saved data into %s]",fname);
//	}
//	if (direction==FileToMemory)
//	{
//		FILE*f = fopen(fname,"rt");
//		if (f==0)
//		{
//			log("unable to access file %s",fname);
//			return 0;
//		}
//		fscanf(f,fmt, &args...);
//		fclose(f);
//		log("[saveLoadTextData: read data from %s]",fname);
//	}
//	return 1;
//}

template< typename LBM_TYPE >
template< typename... ARGS >
int State<LBM_TYPE>::saveLoadTextData(int direction, const char*dirname, const char*filename, ARGS&... args)
{
	// check if main dir exists
	mkdir_p(dirname, 0777);
	char fname[FILENAME_CHARS];
	sprintf(fname,"%s/%s_rank%03d", dirname, filename, lbm.rank);

	const std::string fmt = getSaveLoadFmt(args...);

	if (direction==MemoryToFile)
	{
		FILE*f = fopen(fname,"wt");
		if (f==0)
		{
			log("unable to create file %s",fname);
			return 0;
		}
		fprintf(f,fmt.c_str(), args...);
		fclose(f);
		log("[saveLoadTextData: saved data into %s]",fname);
	}
	if (direction==FileToMemory)
	{
		FILE*f = fopen(fname,"rt");
		if (f==0)
		{
			log("unable to access file %s",fname);
			return 0;
		}
		fscanf(f,fmt.c_str(), &args...);
		fclose(f);
		log("[saveLoadTextData: read data from %s]",fname);
	}
	return 1;
}

template< typename LBM_TYPE >
template< typename VARTYPE >
int State<LBM_TYPE>::saveloadBinaryData(int direction, const char*dirname, const char*filename, VARTYPE*data, idx length)
{
	// check if main dir exists
	mkdir_p(dirname, 0777);
	char fname[FILENAME_CHARS];
	sprintf(fname,"%s/%s_rank%03d", dirname, filename, lbm.rank);

	if (direction==MemoryToFile)
	{
		FILE*f = fopen(fname,"wb");
		if (f==0)
		{
			log("unable to create file %s",fname);
			return 0;
		}
		fwrite(data, sizeof(VARTYPE), length, f);
		fclose(f);
		log("[saveLoadBinaryData: saved data into %s]",fname);
	}
	if (direction==FileToMemory)
	{
		FILE*f = fopen(fname,"rb");
		if (f==0)
		{
			log("unable to access file %s",fname);
			return 0;
		}
		fread(data, sizeof(VARTYPE), length, f);
		fclose(f);
		log("[saveLoadBinaryData: read data from %s]",fname);
	}
	return 1;
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::saveAndLoadState(int direction, const char*subdirname)
{
	char final_dirname[FILENAME_CHARS];
	sprintf(final_dirname, "results_%s/%s", id, subdirname);
	mkdir_p(final_dirname, 0777);

	char tmp_dirname[FILENAME_CHARS];
	const char* local_scratch = getenv("LOCAL_SCRATCH");
	if (direction == FileToMemory || !local_scratch || local_scratch[0] == '\0')
	{
		// $LOCAL_SCRATCH is not defined or empty - default to regular subdirectory in results_*
		sprintf(tmp_dirname, final_dirname);
		local_scratch = NULL;
	}
	else
	{
		// Write files temporarily into the local scratch and move them to final_dirname at
		// the end, after all MPI processes have completed. This avoids small buffered writes
		// into the shared filesystem on clusters as well as corruption of previous state due
		// to MPI errors...
		sprintf(tmp_dirname, "%s/%s", local_scratch, final_dirname);
	}

	char nid[200];

//	saveLoadTextData(direction, tmp_dirname, "config", "%d\n%d\n%d\n%d\n%d\n%d\n%d\n%.20le\n",
//			lbm.iterations, lbm.global_X, lbm.global_Y, lbm.global_Z, lbm.local_X, lbm.local_Y, lbm.local_Z, lbm.physFinalTime);
	saveLoadTextData(direction, tmp_dirname, "config", lbm.iterations, lbm.global_X, lbm.global_Y, lbm.global_Z, lbm.local_X, lbm.local_Y, lbm.local_Z, lbm.physFinalTime);

	// save all counter states
	for (int c=0;c<MAX_COUNTER;c++)
	{
		sprintf(nid,"cnt_%d",c);
//		saveLoadTextData(direction, tmp_dirname, nid, "%d\n%le\n", cnt[c].count, cnt[c].period);
		saveLoadTextData(direction, tmp_dirname, nid, cnt[c].count, cnt[c].period);
	}

	// save probes
	for (int i=0;i<probe1Dvec.size();i++)
	{
		sprintf(nid,"probe1D_%d",i);
//		saveLoadTextData(direction, tmp_dirname, nid, "%d\n", probe1Dvec[i].cycle);
		saveLoadTextData(direction, tmp_dirname, nid, probe1Dvec[i].cycle);
	}
	for (int i=0;i<probe1Dlinevec.size();i++)
	{
		sprintf(nid,"probe1Dline_%d",i);
//		saveLoadTextData(direction, tmp_dirname, nid, "%d\n", probe1Dlinevec[i].cycle);
		saveLoadTextData(direction, tmp_dirname, nid, probe1Dlinevec[i].cycle);
	}
	for (int i=0;i<probe2Dvec.size();i++)
	{
		sprintf(nid,"probe2D_%d",i);
//		saveLoadTextData(direction, tmp_dirname, nid, "%d\n", probe2Dvec[i].cycle);
		saveLoadTextData(direction, tmp_dirname, nid, probe2Dvec[i].cycle);
	}

	// save DFs
	for (int dfty=0;dfty<DFMAX;dfty++)
	{
		sprintf(nid,"df_%d",dfty);
		#ifdef HAVE_MPI
		saveloadBinaryData(direction, tmp_dirname, nid, lbm.hfs[dfty].getData(), lbm.hfs[dfty].getLocalStorageSize());
		#else
		saveloadBinaryData(direction, tmp_dirname, nid, lbm.hfs[dfty].getData(), lbm.hfs[dfty].getStorageSize());
		#endif
	}

	// save map
	sprintf(nid,"map");
	#ifdef HAVE_MPI
	saveloadBinaryData(direction, tmp_dirname, nid, lbm.hmap.getData(), lbm.hmap.getLocalStorageSize());
	#else
	saveloadBinaryData(direction, tmp_dirname, nid, lbm.hmap.getData(), lbm.hmap.getStorageSize());
	#endif

	// save macro
	if (LBM_TYPE::MACRO::N>0)
	{
		sprintf(nid,"macro");
		#ifdef HAVE_MPI
		saveloadBinaryData(direction, tmp_dirname, nid, lbm.hmacro.getData(), lbm.hmacro.getLocalStorageSize());
		#else
		saveloadBinaryData(direction, tmp_dirname, nid, lbm.hmacro.getData(), lbm.hmacro.getStorageSize());
		#endif
	}

	if (local_scratch)
	{
		// move the files from local_scratch into final_dirname and create a backup of the existing files
		for (int i = 0; i < 2; i++)
		{
			// wait for all processes to create temporary files
			TNL::MPI::Barrier();

			// first iteration: create temporary files in the destination directory
			// second iteration: rename the temporary files to the target files
			char src[200];
			char dst[200];
			char src_suffix[5];
			char dst_suffix[5];
			if (i == 0)
			{
				log("[moving files from local scratch to temporary files in the destination directory]");
				sprintf(src_suffix, "");
				sprintf(dst_suffix, ".tmp");
			}
			else
			{
				log("[renaming temporary files to the target files]");
				sprintf(src_suffix, ".tmp");
				sprintf(dst_suffix, "");
				sprintf(tmp_dirname, final_dirname);
			}

			sprintf(src, "config_rank%03d%s", lbm.rank, src_suffix);
			sprintf(dst, "config_rank%03d%s", lbm.rank, dst_suffix);
			move(tmp_dirname, final_dirname, src, dst);

			// save all counter states
			for (int c=0;c<MAX_COUNTER;c++)
			{
				sprintf(src, "cnt_%d_rank%03d%s", c, lbm.rank, src_suffix);
				sprintf(dst, "cnt_%d_rank%03d%s", c, lbm.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}

			// save probes
			for (int i=0;i<probe1Dvec.size();i++)
			{
				sprintf(src, "probe1D_%d_rank%03d%s", i, lbm.rank, src_suffix);
				sprintf(dst, "probe1D_%d_rank%03d%s", i, lbm.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}
			for (int i=0;i<probe1Dlinevec.size();i++)
			{
				sprintf(src,"probe1Dline_%d_rank%03d%s", i, lbm.rank, src_suffix);
				sprintf(dst,"probe1Dline_%d_rank%03d%s", i, lbm.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}
			for (int i=0;i<probe2Dvec.size();i++)
			{
				sprintf(src,"probe2D_%d_rank%03d%s", i, lbm.rank, src_suffix);
				sprintf(dst,"probe2D_%d_rank%03d%s", i, lbm.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}

			// save DFs
			for (int dfty=0;dfty<DFMAX;dfty++)
			{
				sprintf(src, "df_%d_rank%03d%s", dfty, lbm.rank, src_suffix);
				sprintf(dst, "df_%d_rank%03d%s", dfty, lbm.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}

			// save map
			sprintf(src, "map_rank%03d%s", lbm.rank, src_suffix);
			sprintf(dst, "map_rank%03d%s", lbm.rank, dst_suffix);
			move(tmp_dirname, final_dirname, src, dst);

			// save macro
			if (LBM_TYPE::MACRO::N>0)
			{
				sprintf(src, "macro_rank%03d%s", lbm.rank, src_suffix);
				sprintf(dst, "macro_rank%03d%s", lbm.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}
		}
	}
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::saveState(bool forced)
{
//	flagCreate("do_save_state");
	if (flagExists("savestate") || !check_savestate_flag || forced)
	{
		log("[saveState invoked]");
		saveAndLoadState(MemoryToFile, "current_state");
		if (delete_savestate_flag && !forced)
		{
			flagDelete("savestate");
//			flagRename("savestate","savestate_done");
			flagCreate("savestate_done");
		}
		if (forced) flagCreate("loadstate");
	}
	// debug
//	saveAndLoadState(FileToMemory, "current_state");
}

template< typename LBM_TYPE >
void State<LBM_TYPE>::loadState(bool forced)
{
//	flagCreate("do_save_state");
	if (flagExists("loadstate") || forced)
	{
		log("[loadState invoked]");
//		printf("Provadim cteni df\n");
		saveAndLoadState(FileToMemory, "current_state");
//		if (delete_savestate_flag)
//			flagDelete("savestate");
//			flagRename("savestate","savestate_saved");
	}
	// debug
//	saveAndLoadState(FileToMemory, "current_state");
}

template< typename LBM_TYPE >
bool State<LBM_TYPE>::wallTimeReached()
{
	bool local_result = false;
	if (wallTime > 0)
	{
		timespec t_actual;
		clock_gettime(CLOCK_REALTIME, &t_actual);
		long actualtimediff = (t_actual.tv_sec - t_init.tv_sec);
		if (actualtimediff >= wallTime)
		{
			log("wallTime reached: %ld / %ld [sec]", actualtimediff, wallTime);
			local_result = true;
		}
	}
	bool result;
	TNL::MPI::Allreduce(&local_result, &result, 1, MPI_LOR, TNL::MPI::AllGroup());
	return result;
}

template< typename LBM_TYPE >
double State<LBM_TYPE>::getWallTime(bool collective)
{
	double walltime = 0;
	if (!collective || lbm.rank == 0)
	{
		timespec t_actual;
		clock_gettime(CLOCK_REALTIME, &t_actual);
		walltime = (t_actual.tv_sec - t_init.tv_sec) + 1e-9 * (t_actual.tv_nsec - t_init.tv_nsec);
	}
	if (collective)
	{
		// collective operation - make sure that all MPI processes return the same walltime (taken from rank 0)
		TNL::MPI::Bcast(&walltime, 1, 0, TNL::MPI::AllGroup());
	}
	return walltime;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// LBM RELATED
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename LBM_TYPE >
void State<LBM_TYPE>::resetLattice(real rho, real vx, real vy, real vz)
{
	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = lbm.offset_X; x < lbm.offset_X + lbm.local_X; x++)
	for (idx z = lbm.offset_Z; z < lbm.offset_Z + lbm.local_Z; z++)
	for (idx y = lbm.offset_Y; y < lbm.offset_Y + lbm.local_Y; y++)
		lbm.setEqLat(x,y,z,rho,vx,vy,vz);
}

// clear Lattice and boundary setup
template< typename LBM_TYPE >
void State<LBM_TYPE>::reset()
{
	lbm.resetMap(LBM_TYPE::BC::GEO_FLUID);
	setupBoundaries();		// this can be virtualized
	lbm.projectWall();
	resetLattice(1.0, 0, 0, 0);
//	resetLattice(1.0, lbmInputVelocityX(), lbmInputVelocityY(),lbmInputVelocityZ());

	//initial time of current simulation
	clock_gettime(CLOCK_REALTIME, &t_init);
}

template< typename LBM_TYPE >
bool State<LBM_TYPE>::estimateMemoryDemands()
{
	long long memDFs = lbm.local_X*lbm.local_Y*lbm.local_Z*27*sizeof(dreal);
	long long memMacro = lbm.local_X*lbm.local_Y*lbm.local_Z*sizeof(dreal)*LBM_TYPE::MACRO::N;
	long long memMap = lbm.local_X*lbm.local_Y*lbm.local_Z*sizeof(map_t);
	long long CPUavail = sysconf(_SC_PHYS_PAGES)*sysconf(_SC_PAGE_SIZE);
	long long GPUavail = 0;
	long long GPUtotal = 0;
	long long GPUtotal_hw = 0;
	long long CPUtotal = memMacro + memMap + DFMAX*memDFs;
	long long CPUDFs = DFMAX*memDFs;
	#ifdef USE_CUDA
	GPUavail = 0;
	GPUtotal_hw =0;
	GPUtotal += DFMAX*memDFs + memMacro + memMap;
//	CPUDFs = 0;

	// get number of CUDA GPUs
//	int num_gpus=0;
//	cudaGetDeviceCount(&num_gpus);

	// display CPU and GPU configuration
//	log("number of CUDA devices:\t%d", num_gpus);
//	for (int i = 0; i < num_gpus; i++)
	{
		int gpu_id;
		cudaGetDevice(&gpu_id);
		cudaDeviceProp dprop;
		cudaGetDeviceProperties(&dprop, gpu_id);
		log("Rank %d uses GPU id %d: %s", lbm.rank, gpu_id, dprop.name);
		// NOTE: cudaSetDevice breaks MPI !!!
//		cudaSetDevice(i);
		size_t free=0, total=0;
		cudaMemGetInfo(&free, &total);
		GPUavail += free;
		GPUtotal_hw += total;
	}

	#else
//	CPUtotal += CPUDFs;
	#endif

	log("Local memory budget analysis / estimation for MPI rank %d", lbm.rank);
	log("CPU RAM for DFs:   %ld MiB", (long)(CPUDFs/1024/1024));
//	log("CPU RAM for lat:   %ld MiB", (long)(memDFs/1024/1024));
	log("CPU RAM for map:   %ld MiB", (long)(memMap/1024/1024));
	log("CPU RAM for macro: %ld MiB", (long)(memMacro/1024/1024));
	log("TOTAL CPU RAM %ld MiB estimated needed, %ld MiB available (%6.4f%%)", (long)(CPUtotal/1024/1024), (long)(CPUavail/1024/1024), (double)(100.0*CPUtotal/CPUavail));
	#ifdef USE_CUDA
	log("GPU RAM for DFs:   %ld MiB", (long)(DFMAX*memDFs/1024/1024));
	log("GPU RAM for map:   %ld MiB", (long)(memMap/1024/1024));
	log("GPU RAM for macro: %ld MiB", (long)(memMacro/1024/1024));
	log("TOTAL GPU RAM %ld MiB estimated needed, %ld MiB available (%6.4f%%), total GPU RAM: %ld MiB", (long)(GPUtotal/1024/1024), (long)(GPUavail/1024/1024), (double)(100.0*GPUtotal/GPUavail), (long)(GPUtotal_hw/1024/1024));
	if (GPUavail <= GPUtotal) return false;
	#endif
	if (CPUavail <= CPUtotal) return false;
	return true;
}
