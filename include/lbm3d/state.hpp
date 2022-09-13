#pragma once

#include "state.h"

#include "fileutils.h"
#include "timeutils.h"

template< typename NSE >
int State<NSE>::addLagrange3D()
{
	char dir[FILENAME_CHARS];
	sprintf(dir,"results_%s",id);
	FF.emplace_back(nse, dir);
	return FF.size()-1;
}

template< typename NSE >
void State<NSE>::computeAllLagrangeForces()
{
	for (std::size_t i=0;i<FF.size();i++)
		if (FF[i].implicitWuShuForcing)
			FF[i].computeWuShuForcesSparse(nse.physTime());
}

template< typename NSE >
bool State<NSE>::getPNGdimensions(const char * filename, int &w, int &h)
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


template< typename NSE >
template< typename... ARGS >
void State<NSE>::log(const char* fmt, ARGS... args)
{
	char dir[FILENAME_CHARS];
	sprintf(dir,"results_%s",id);
	mkdir(dir,0777);
	char fname[FILENAME_CHARS];
	sprintf(fname,"%s/log_rank%03d",dir,nse.rank);

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
template< typename NSE >
template< typename... ARGS >
void State<NSE>::setid(const char* fmt, ARGS... args)
{
	sprintf(id, fmt, args...);
}

template< typename NSE >
void State<NSE>::flagCreate(const char*flagname)
{
	if (nse.rank != 0) return;

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

template< typename NSE >
void State<NSE>::flagDelete(const char*flagname)
{
	if (nse.rank != 0) return;

	char fname[FILENAME_CHARS];
	sprintf(fname,"results_%s/%s",id,flagname);
	if (fileExists(fname)) remove(fname);
}

template< typename NSE >
bool State<NSE>::flagExists(const char*flagname)
{
	char fname[FILENAME_CHARS];
	sprintf(fname,"results_%s/%s",id,flagname);
	return fileExists(fname);
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::mark(const char* fmt, ARGS... args)
{
	if (nse.rank != 0) return;

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
template< typename NSE >
bool State<NSE>::isMark()
{
	bool result;
	if (nse.rank == 0)
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
	TNL::MPI::Bcast(&result, 1, 0, nse.communicator);
	return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK SURFACE
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename NSE >
void State<NSE>::writeVTK_Surface(const char* name, real time, int cycle, Lagrange3D &fil)
{
	VTKWriter vtk;

	char fname[FILENAME_CHARS];
	sprintf(fname,"results_%s/vtk3D/rank%03d_%s.vtk",id,nse.rank,name);
	create_file(fname);

	FILE* fp = fopen(fname, "w+");
	vtk.writeHeader(fp);

	fprintf(fp, "DATASET POLYDATA\n");

	fprintf(fp, "POINTS %d float\n", fil.LL.size());
	for (std::size_t i=0;i<fil.LL.size();i++)
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


template< typename NSE >
void State<NSE>::writeVTK_Points(const char* name, real time, int cycle, Lagrange3D &fil)
{
	VTKWriter vtk;

	char fname[FILENAME_CHARS];
	sprintf(fname,"results_%s/vtk3D/rank%03d_%s.vtk",id,nse.rank,name);
	create_file(fname);

	FILE* fp = fopen(fname, "w+");
	vtk.writeHeader(fp);

	fprintf(fp, "DATASET POLYDATA\n");

	fprintf(fp, "POINTS %d float\n", fil.LL.size());
	for (std::size_t i=0;i<fil.LL.size();i++)
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


template< typename NSE >
template< typename... ARGS >
void State<NSE>::add1Dcut(point_t from, point_t to, const char* fmt, ARGS... args)
{
	probe1Dlinevec.push_back( T_PROBE1DLINECUT() );
	int last = probe1Dlinevec.size()-1;
	sprintf(probe1Dlinevec[last].name, fmt, args...);
	probe1Dlinevec[last].from = from;
	probe1Dlinevec[last].to = to;
	probe1Dlinevec[last].cycle = 0;
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add1Dcut_X(real y, real z, const char* fmt, ARGS... args)
{
	if (!nse.isAnyLocalY(nse.lat.phys2lbmY(y)) || !nse.isAnyLocalZ(nse.lat.phys2lbmZ(z))) return;

	probe1Dvec.push_back( T_PROBE1DCUT() );
	int last = probe1Dvec.size()-1;
	sprintf(probe1Dvec[last].name, fmt, args...);
	probe1Dvec[last].type = 0;
	probe1Dvec[last].pos1 = nse.lat.phys2lbmY(y);
	probe1Dvec[last].pos2 = nse.lat.phys2lbmZ(z);
	probe1Dvec[last].cycle = 0;
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add1Dcut_Y(real x, real z, const char* fmt, ARGS... args)
{
	if (!nse.isAnyLocalX(nse.lat.phys2lbmX(x)) || !nse.isAnyLocalZ(nse.lat.phys2lbmZ(z))) return;

	probe1Dvec.push_back( T_PROBE1DCUT() );
	int last = probe1Dvec.size()-1;
	sprintf(probe1Dvec[last].name, fmt, args...);
	probe1Dvec[last].type = 1;
	probe1Dvec[last].pos1 = nse.lat.phys2lbmX(x);
	probe1Dvec[last].pos2 = nse.lat.phys2lbmZ(z);
	probe1Dvec[last].cycle = 0;
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add1Dcut_Z(real x, real y, const char* fmt, ARGS... args)
{
	if (!nse.isAnyLocalX(nse.lat.phys2lbmX(x)) || !nse.isAnyLocalY(nse.lat.phys2lbmY(y))) return;

	probe1Dvec.push_back( T_PROBE1DCUT() );
	int last = probe1Dvec.size()-1;
	sprintf(probe1Dvec[last].name, fmt, args...);
	probe1Dvec[last].type = 2;
	probe1Dvec[last].pos1 = nse.lat.phys2lbmX(x);
	probe1Dvec[last].pos2 = nse.lat.phys2lbmY(y);
	probe1Dvec[last].cycle = 0;
}

template< typename NSE >
void State<NSE>::writeVTKs_1D()
{
	if (probe1Dvec.size()>0)
	{
		// browse all 1D probeline cuts
		for (std::size_t i=0;i<probe1Dvec.size(); i++)
		{
			char fname[FILENAME_CHARS];
			sprintf(fname,"results_%s/probes1D/%s_rank%03d_%06d", id, probe1Dvec[i].name, nse.rank, probe1Dvec[i].cycle);
			// create parent directories
			create_file(fname);
			log("[1dcut %s]",fname);
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
	for (std::size_t i=0;i<probe1Dlinevec.size(); i++)
	{
		char fname[FILENAME_CHARS];
//		sprintf(fname,"results_%s/probes1D/%s_%06d_t%f", id, probe1Dvec[i].name, probe1Dvec[i].cycle, nse.physTime());
		sprintf(fname,"results_%s/probes1D/%s_rank%03d_%06d", id, probe1Dlinevec[i].name, nse.rank, probe1Dlinevec[i].cycle);
		// create parent directories
		create_file(fname);
		log("[1dcut %s]",fname);
		write1Dcut(probe1Dlinevec[i].from, probe1Dlinevec[i].to, fname);
		probe1Dlinevec[i].cycle++;
	}
}

template< typename NSE >
void State<NSE>::write1Dcut(point_t from, point_t to, const char * fname)
{
	FILE*fout = fopen(fname,"wt"); // append information
	point_t i = nse.lat.phys2lbmPoint(from);
	point_t f = nse.lat.phys2lbmPoint(to);
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
	fprintf(fout,"#time %f s\n", nse.physTime());
	fprintf(fout,"#1:rel_pos");

	int count=2, index=0;
	while (outputData(nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (int i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,i);
	}
	fprintf(fout,"\n");

	for (real s=0;s<=1.0;s+=ds)
	{
		point_t p = i + s * (f - i);
		for (const auto& block : nse.blocks)
		{
			if (!block.isLocalIndex((idx) p.x(), (idx) p.y(), (idx) p.z())) continue;
			if (block.isFluid((idx) p.x(), (idx) p.y(), (idx) p.z()))
			{
				fprintf(fout, "%e", (s*dist-0.5)*nse.lat.physDl);
				index=0;
				while (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs))
				{
					for (int dof=0;dof<dofs;dof++)
					{
						outputData(block, index-1, dof, idd, (idx) p.x(), (idx) p.y(), (idx) p.z(), value, dofs);
						fprintf(fout, "\t%e", value);
					}
				}
				fprintf(fout, "\n");
			}
		}
	}
	fclose(fout);
}

template< typename NSE >
void State<NSE>::write1Dcut_X(idx y, idx z, const char * fname)
{
	FILE*fout = fopen(fname,"wt"); // append information
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout,"#time %f s\n", nse.physTime());
	fprintf(fout,"#1:x");
	int count=2, index=0;
	while (outputData(nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (idx i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,(int)i);
	}
	fprintf(fout,"\n");

	for (const auto& block : nse.blocks)
	for (idx i = block.offset.x(); i < block.offset.x() + block.local.x(); i++)
	{
		if (block.isFluid(i,y,z))
		{
			fprintf(fout, "%e", nse.lat.lbm2physX(i));
			index=0;
			if (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs))
			{
				for (int dof=0;dof<dofs;dof++)
				{
					outputData(block, index-1,dof,idd,i,y,z,value,dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	}
	fclose(fout);
}

template< typename NSE >
void State<NSE>::write1Dcut_Y(idx x, idx z, const char * fname)
{
	FILE*fout = fopen(fname,"wt"); // append information
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout,"#time %f s\n", nse.physTime());
	fprintf(fout,"#1:y");
	int count=2, index=0;
	while (outputData(nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (idx i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,(int)i);
	}
	fprintf(fout,"\n");

	for (const auto& block : nse.blocks)
	for (idx i = block.offset.y(); i < block.offset.y() + block.local.y(); i++)
	{
		if (block.isFluid(x,i,z))
		{
			fprintf(fout, "%e", nse.lat.lbm2physY(i));
			int index=0;
			while (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs))
			{
				for (int dof=0;dof<dofs;dof++)
				{
					outputData(block, index-1,dof,idd,x,i,z,value,dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	}
	fclose(fout);
}

template< typename NSE >
void State<NSE>::write1Dcut_Z(idx x, idx y, const char * fname)
{
	FILE*fout = fopen(fname,"wt"); // append information
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout,"#time %f s\n", nse.physTime());
	fprintf(fout,"#1:z");
	int count=2, index=0;
	while (outputData(nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (idx i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,(int)i);
	}
	fprintf(fout,"\n");

	for (const auto& block : nse.blocks)
	for (idx i = block.offset.z(); i < block.offset.z() + block.local.z(); i++)
	{
		if (block.isFluid(x,y,i))
		{
			fprintf(fout, "%e", nse.lat.lbm2physZ(i));
			index=0;
			while (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs))
			{
				for (int dof=0;dof<dofs;dof++)
				{
					outputData(block, index-1,dof,idd,x,y,i,value,dofs);
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


template< typename NSE >
void State<NSE>::writeVTKs_3D()
{
	char dir[FILENAME_CHARS], filename[FILENAME_CHARS], basename[FILENAME_CHARS];
	sprintf(dir,"results_%s/vtk3D", id);
	mkdir_p(dir,0755);

	for (const auto& block : nse.blocks)
	{
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

		sprintf(basename,"block%03d_%d.vtk", block.id, cnt[VTK3D].count);
		sprintf(filename,"%s/%s", tmp_dirname, basename);
		auto outputData = [this] (const typename T_LBM_NSE::BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
		{
			return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
		};
		block.writeVTK_3D(nse.lat, outputData, filename, nse.physTime(), cnt[VTK3D].count);
		log("[vtk %s written, time %f, cycle %d] ", filename, nse.physTime(), cnt[VTK3D].count);

		if (local_scratch)
		{
			// move the files from local_scratch into final_dirname and create a backup of the existing files
			move(tmp_dirname, dir, basename, basename);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK 3D CUT
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename NSE >
template< typename... ARGS >
void State<NSE>::add3Dcut(idx ox, idx oy, idx oz, idx lx, idx ly, idx lz, idx step, const char* fmt, ARGS... args)
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

template< typename NSE >
void State<NSE>::writeVTKs_3Dcut()
{
	if (probe3Dvec.size()<=0) return;
	// browse all 3D vtk cuts
	for (auto& probevec : probe3Dvec)
	{
		for (const auto& block : nse.blocks)
		{
			char fname[FILENAME_CHARS];
			sprintf(fname,"results_%s/vtk3Dcut/%s_block%03d_%d.vtk", id, probevec.name, block.id, probevec.cycle);
			// create parent directories
			create_file(fname);
			auto outputData = [this] (const typename T_LBM_NSE::BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
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
			log("[vtk %s written, time %f, cycle %d] ", fname, nse.physTime(), probevec.cycle);
		}
		probevec.cycle++;
	}
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK 2D CUT
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add2Dcut_X(idx x, const char* fmt, ARGS... args)
{
	if (!nse.isAnyLocalX(x)) return;

	probe2Dvec.push_back( T_PROBE2DCUT() );
	int last = probe2Dvec.size()-1;

	sprintf(probe2Dvec[last].name, fmt, args...);

	probe2Dvec[last].type = 0;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = x;
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add2Dcut_Y(idx y, const char* fmt, ARGS... args)
{
	if (!nse.isAnyLocalY(y)) return;

	probe2Dvec.push_back( T_PROBE2DCUT() );
	int last = probe2Dvec.size()-1;

	sprintf(probe2Dvec[last].name, fmt, args...);

	probe2Dvec[last].type = 1;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = y;
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add2Dcut_Z(idx z, const char* fmt, ARGS... args)
{
	if (!nse.isAnyLocalZ(z)) return;

	probe2Dvec.push_back( T_PROBE2DCUT() );
	int last = probe2Dvec.size()-1;

	sprintf(probe2Dvec[last].name, fmt, args...);

	probe2Dvec[last].type = 2;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = z;
}


template< typename NSE >
void State<NSE>::writeVTKs_2D()
{
	if (probe2Dvec.size()<=0) return;
	// browse all 2D vtk cuts
	for (auto& probevec : probe2Dvec)
	{
		for (const auto& block : nse.blocks)
		{
			char fname[FILENAME_CHARS];
			sprintf(fname,"results_%s/vtk2D/%s_block%03d_%d.vtk", id, probevec.name, block.id, probevec.cycle);
			// create parent directories
			create_file(fname);
			auto outputData = [this] (const typename T_LBM_NSE::BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
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


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// PNG PROJECTION
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename NSE >
bool State<NSE>::projectPNG_X(const char * filename, idx x0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (!nse.isLocalX(x0)) return true;

	if (!fileExists(filename)) { printf("file %s does not exist\n",filename); return false; }
	PNGTool P(filename);

	// plane y-z
	idx x = x0;
	for (idx z = nse.lat.offset.z(); z < nse.lat.offset.z() + nse.lat.local.z(); z++)
	{
		real a = (real)z/(real)(nse.lat.global.z() - 1); // a in [0,1]
		a = amin + a * (amax - amin); // a in [amin, amax]
		if (mirror) a = 1.0 - a;
		for (idx y = nse.lat.offset.y(); y < nse.lat.offset.y() + nse.lat.local.y(); y++)
		{
			real b = (real)y/(real)(nse.lat.global.y() - 1); // b in [0,1]
			b = bmin + b * (bmax - bmin); // b in [bmin, bmax]
			if (flip) b = 1.0 - b;
			if (rotate)
			{
				if (P.intensity(b,a) > 0) nse.defineWall(x, y, z, true);
			}
			else
			{
				if (P.intensity(a,b) > 0) nse.defineWall(x, y, z, true);
			}
		}
	}
	return true;
}

template< typename NSE >
bool State<NSE>::projectPNG_Y(const char * filename, idx y0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (!fileExists(filename)) { printf("file %s does not exist\n",filename); return false; }
	PNGTool P(filename);

	for (auto& block : nse.blocks)
	{
		if (!block.isLocalY(y0)) continue;

		// plane x-z
		idx y=y0;
		for (idx z = block.offset.z(); z < block.offset.z() + block.local.z(); z++)
		{
			real a = (real)z/(real)(nse.lat.global.z() - 1); // a in [0,1]
			a = amin + a * (amax - amin); // a in [amin, amax]
			if (mirror) a = 1.0 - a;
			for (idx x = block.offset.x(); x < block.offset.x() + block.local.x(); x++)
			{
				real b = (real)x/(real)(nse.lat.global.x() - 1); // b in [0,1]
				b = bmin + b * (bmax - bmin); // b in [bmin, bmax]
				if (flip) b = 1.0 - b;
				if (rotate)
				{
					if (P.intensity(b,a) > 0) block.defineWall(x, y, z, true);
				}
				else
				{
					if (P.intensity(a,b) > 0) block.defineWall(x, y, z, true);
				}
			}
		}
	}
	return true;
}


template< typename NSE >
bool State<NSE>::projectPNG_Z(const char * filename, idx z0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (!nse.isLocalZ(z0)) return true;

	if (!fileExists(filename)) { printf("file %s does not exist\n",filename); return false; }
	PNGTool P(filename);

	// plane x-y
	idx z=z0;
	for (idx x = nse.lat.offset.x(); x < nse.lat.offset.x() + nse.lat.local.x(); x++)
	{
		real a = (real)x/(real)(nse.lat.global.x() - 1); // a in [0,1]
		a = amin + a * (amax - amin); // a in [amin, amax]
		if (mirror) a = 1.0 - a;
		for (idx y = nse.lat.offset.y(); y < nse.lat.offset.y() + nse.lat.local.y(); y++)
		{
			real b = (real)y/(real)(nse.lat.global.y() - 1); // b in [0,1]
			b = bmin + b * (bmax - bmin); // b in [bmin, bmax]
			if (flip) b = 1.0 - b;
			if (rotate)
			{
				if (P.intensity(b,a) > 0) nse.defineWall(x, y, z, true);
			}
			else
			{
				if (P.intensity(a,b) > 0) nse.defineWall(x, y, z, true);
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


template< typename NSE >
void State<NSE>::move(const char* srcdir, const char* dstdir, const char* srcfilename, const char* dstfilename)
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

//template< typename NSE >
//template< typename... ARGS >
//int State<NSE>::saveLoadTextData(int direction, const char*dirname, const char*filename, const char*fmt, ARGS&... args)
//{
//	// check if main dir exists
//	mkdir_p(dirname, 0777);
//	char fname[FILENAME_CHARS];
//	sprintf(fname,"%s/%s_rank%03d", dirname, filename, nse.rank);
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

template< typename NSE >
template< typename... ARGS >
int State<NSE>::saveLoadTextData(int direction, const char*dirname, const char*filename, ARGS&... args)
{
	// check if main dir exists
	mkdir_p(dirname, 0777);
	char fname[FILENAME_CHARS];
	sprintf(fname,"%s/%s_rank%03d", dirname, filename, nse.rank);

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

template< typename NSE >
template< typename VARTYPE >
int State<NSE>::saveloadBinaryData(int direction, const char*dirname, const char*filename, VARTYPE*data, idx length)
{
	// check if main dir exists
	mkdir_p(dirname, 0777);
	char fname[FILENAME_CHARS];
	sprintf(fname,"%s/%s_rank%03d", dirname, filename, nse.rank);

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

template< typename NSE >
void State<NSE>::saveAndLoadState(int direction, const char*subdirname)
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
//			nse.iterations, nse.lat.global.x(), nse.lat.global.y(), nse.lat.global.z(), nse.lat.local.x(), nse.lat.local.y(), nse.lat.local.z(), nse.physFinalTime);
// FIXME: save and restore number of blocks
//	saveLoadTextData(direction, tmp_dirname, "config", nse.iterations, nse.lat.global.x(), nse.lat.global.y(), nse.lat.global.z(), nse.lat.local.x(), nse.lat.local.y(), nse.lat.local.z(), nse.physFinalTime);
	saveLoadTextData(direction, tmp_dirname, "config", nse.iterations, nse.lat.global.x(), nse.lat.global.y(), nse.lat.global.z(), nse.physFinalTime);
//	for (auto& block : nse.blocks)
//		saveLoadTextData(direction, tmp_dirname, "config", block.local.x(), block.local.y(), block.local.z(), block.offset.x(), block.offset.y(), block.offset.z());

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

	for (auto& block : nse.blocks)
	{
		// save DFs
		for (int dfty=0;dfty<DFMAX;dfty++)
		{
			sprintf(nid,"df_%d",dfty);
			#ifdef HAVE_MPI
			saveloadBinaryData(direction, tmp_dirname, nid, block.hfs[dfty].getData(), block.hfs[dfty].getLocalStorageSize());
			#else
			saveloadBinaryData(direction, tmp_dirname, nid, block.hfs[dfty].getData(), block.hfs[dfty].getStorageSize());
			#endif
		}

		// save map
		sprintf(nid,"map");
		#ifdef HAVE_MPI
		saveloadBinaryData(direction, tmp_dirname, nid, block.hmap.getData(), block.hmap.getLocalStorageSize());
		#else
		saveloadBinaryData(direction, tmp_dirname, nid, block.hmap.getData(), block.hmap.getStorageSize());
		#endif

		// save macro
		if (NSE::MACRO::N>0)
		{
			sprintf(nid,"macro");
			#ifdef HAVE_MPI
			saveloadBinaryData(direction, tmp_dirname, nid, block.hmacro.getData(), block.hmacro.getLocalStorageSize());
			#else
			saveloadBinaryData(direction, tmp_dirname, nid, block.hmacro.getData(), block.hmacro.getStorageSize());
			#endif
		}
	}

	if (local_scratch)
	{
		// move the files from local_scratch into final_dirname and create a backup of the existing files
		for (int i = 0; i < 2; i++)
		{
			// wait for all processes to create temporary files
			TNL::MPI::Barrier(nse.communicator);

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

			sprintf(src, "config_rank%03d%s", nse.rank, src_suffix);
			sprintf(dst, "config_rank%03d%s", nse.rank, dst_suffix);
			move(tmp_dirname, final_dirname, src, dst);

			// save all counter states
			for (int c=0;c<MAX_COUNTER;c++)
			{
				sprintf(src, "cnt_%d_rank%03d%s", c, nse.rank, src_suffix);
				sprintf(dst, "cnt_%d_rank%03d%s", c, nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}

			// save probes
			for (int i=0;i<probe1Dvec.size();i++)
			{
				sprintf(src, "probe1D_%d_rank%03d%s", i, nse.rank, src_suffix);
				sprintf(dst, "probe1D_%d_rank%03d%s", i, nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}
			for (int i=0;i<probe1Dlinevec.size();i++)
			{
				sprintf(src,"probe1Dline_%d_rank%03d%s", i, nse.rank, src_suffix);
				sprintf(dst,"probe1Dline_%d_rank%03d%s", i, nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}
			for (int i=0;i<probe2Dvec.size();i++)
			{
				sprintf(src,"probe2D_%d_rank%03d%s", i, nse.rank, src_suffix);
				sprintf(dst,"probe2D_%d_rank%03d%s", i, nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}

			// save DFs
			for (int dfty=0;dfty<DFMAX;dfty++)
			{
				sprintf(src, "df_%d_rank%03d%s", dfty, nse.rank, src_suffix);
				sprintf(dst, "df_%d_rank%03d%s", dfty, nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}

			// save map
			sprintf(src, "map_rank%03d%s", nse.rank, src_suffix);
			sprintf(dst, "map_rank%03d%s", nse.rank, dst_suffix);
			move(tmp_dirname, final_dirname, src, dst);

			// save macro
			if (NSE::MACRO::N>0)
			{
				sprintf(src, "macro_rank%03d%s", nse.rank, src_suffix);
				sprintf(dst, "macro_rank%03d%s", nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}
		}
	}
}

template< typename NSE >
void State<NSE>::saveState(bool forced)
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

template< typename NSE >
void State<NSE>::loadState(bool forced)
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

template< typename NSE >
bool State<NSE>::wallTimeReached()
{
	bool local_result = false;
	if (wallTime > 0)
	{
		long actualtimediff = timer_total.getRealTime();
		if (actualtimediff >= wallTime)
		{
			log("wallTime reached: %ld / %ld [sec]", actualtimediff, wallTime);
			local_result = true;
		}
	}
	return TNL::MPI::reduce(local_result, MPI_LOR, nse.communicator);
}

template< typename NSE >
double State<NSE>::getWallTime(bool collective)
{
	double result = 0;
	if (!collective || nse.rank == 0)
	{
		result = timer_total.getRealTime();
	}
	if (collective)
	{
		// collective operation - make sure that all MPI processes return the same walltime (taken from rank 0)
		TNL::MPI::Bcast(&result, 1, 0, nse.communicator);
	}
	return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// LBM RELATED
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename NSE >
bool State<NSE>::estimateMemoryDemands()
{
	long long memDFs = 0;
	long long memMacro = 0;
	long long memMap = 0;
	for (const auto& block : nse.blocks) {
		const long long XYZ = block.local.x() * block.local.y() * block.local.z();
		memDFs += XYZ * sizeof(dreal) * NSE::Q;
		memMacro += XYZ * sizeof(dreal) * NSE::MACRO::N;
		memMap += XYZ * sizeof(map_t);
	}

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
		log("Rank %d uses GPU id %d: %s", nse.rank, gpu_id, dprop.name);
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

	log("Local memory budget analysis / estimation for MPI rank %d", nse.rank);
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

// clear Lattice and boundary setup
template< typename NSE >
void State<NSE>::reset()
{
	nse.resetMap(NSE::BC::GEO_FLUID);
	setupBoundaries();		// this can be virtualized
	nse.projectWall();
	resetLattice(1.0, 0, 0, 0);
//	resetLattice(1.0, lbmInputVelocityX(), lbmInputVelocityY(),lbmInputVelocityZ());
}

template< typename NSE >
void State<NSE>::resetLattice(real rho, real vx, real vy, real vz)
{
	// NOTE: it is important to reset *all* lattice sites (i.e. including ghost layers) when using the A-A pattern
	// (because GEO_INFLOW and GEO_OUTFLOW_EQ access the ghost layer in streaming)
	nse.forAllLatticeSites( [&] (typename T_LBM_NSE::BLOCK& block, idx x, idx y, idx z) {
		block.setEqLat(x,y,z,rho,vx,vy,vz);
	} );
}

template< typename NSE >
void State<NSE>::SimInit()
{
	glups_prev_time = glups_prev_iterations = 0;

	timer_SimInit.reset();
	timer_SimUpdate.reset();
	timer_AfterSimUpdate.reset();
	timer_compute.reset();
	timer_compute_overlaps.reset();
	timer_wait_communication.reset();
	timer_wait_computation.reset();

	timer_SimInit.start();

	log("MPI info: rank=%d, nproc=%d, lat.global=[%d,%d,%d]", nse.rank, nse.nproc, nse.lat.global.x(), nse.lat.global.y(), nse.lat.global.z());
	for (auto& block : nse.blocks)
		log("LBM block %d: local=[%d,%d,%d], offset=[%d,%d,%d]", block.id, block.local.x(), block.local.y(), block.local.z(), block.offset.x(), block.offset.y(), block.offset.z());

	log("\nSTART: simulation NSE:%s lbmVisc %e physDl %e physDt %e", NSE::COLL::id, nse.lbmViscosity(), nse.lat.physDl, nse.physDt);

	// reset counters
	for (int c=0;c<MAX_COUNTER;c++) cnt[c].count = 0;
	cnt[SAVESTATE].count = 1;  // skip initial save of state
	nse.iterations = 0;

	// check for loadState
//	if(flagExists("current_state/df_0"))
	if(flagExists("loadstate"))
	{
		loadState(); // load saved state into CPU memory
		nse.physStartTime = nse.physTime();
		nse.allocateDeviceData();
	}
	else
	{
		// allocate before reset - it might initialize on the GPU...
		nse.allocateDeviceData();

		// setup map and DFs in CPU memory
		reset();

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
	}

	copyAllToDevice();

#ifdef HAVE_MPI
	// synchronize overlaps with MPI (initial synchronization can be synchronous)
	nse.synchronizeMapDevice();
	nse.synchronizeDFsDevice(df_cur);
	if (NSE::MACRO::use_syncMacro)
		nse.synchronizeMacroDevice();
#endif

	timer_SimInit.stop();
}

template< typename NSE >
void State<NSE>::SimUpdate()
{
	timer_SimUpdate.start();

	// debug
	for (auto& block : nse.blocks)
	if (block.data.lbmViscosity == 0) {
		log("error: LBM viscosity is 0");
		nse.terminate = true;
		return;
	}

	// flags
	bool doComputeVelocitiesStar=false;
	bool doCopyQuantitiesStarToHost=false;
	bool doZeroForceOnDevice=false;
	bool doZeroForceOnHost=false;
	bool doComputeLagrangePhysics=false;
	bool doCopyForceToDevice=false;

	// determine global flags
	// NOTE: all Lagrangian points are assumed to be on the first GPU
	// TODO
//	if (nse.data.rank == 0 && FF.size() > 0)
	if (FF.size() > 0)
	{
		doComputeLagrangePhysics=true;
		for (std::size_t i=0;i<FF.size();i++)
		if (FF[i].implicitWuShuForcing)
		{
			doComputeVelocitiesStar=true;
			switch (FF[i].ws_compute)
			{
				case ws_computeCPU:
				case ws_computeCPU_TNL:
					doCopyQuantitiesStarToHost=true;
					doZeroForceOnHost=true;
					doCopyForceToDevice=true;
					break;
				case ws_computeGPU_TNL:
				case ws_computeHybrid_TNL:
				case ws_computeHybrid_TNL_zerocopy:
				case ws_computeGPU_CUSPARSE:
				case ws_computeHybrid_CUSPARSE:
					doZeroForceOnDevice=true;
					break;
			}
		}
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

	if (doComputeVelocitiesStar)
	{
		for (auto& block : nse.blocks)
		{
		#ifdef USE_CUDA
			const dim3 gridSize = get_grid_size(block);
			if (doZeroForceOnDevice)
				cudaLBMComputeVelocitiesStarAndZeroForce< NSE ><<<gridSize, block.block_size>>>(block.data, nse.rank, nse.nproc);
			else
				cudaLBMComputeVelocitiesStar< NSE ><<<gridSize, block.block_size>>>(block.data, nse.rank, nse.nproc);
			checkCudaDevice;
		#else
			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
			if (doZeroForceOnDevice)
				LBMComputeVelocitiesStarAndZeroForce< NSE >(block.data, nse.rank, nse.nproc, x, y, z);
			else
				LBMComputeVelocitiesStar< NSE >(block.data, nse.rank, nse.nproc, x, y, z);
		#endif
		}
		if (doCopyQuantitiesStarToHost)
		{
			nse.copyMacroToHost();
		}
	}


	// reset lattice force vectors dfx and dfy
	if (doZeroForceOnHost)
	{
		nse.resetForces();
	}

	if (doComputeLagrangePhysics)
	{
		computeAllLagrangeForces();
	}

	if (doCopyForceToDevice)
	{
		nse.copyForcesToDevice();
	}


	// call hook method (used e.g. for extra kernels in the non-Newtonian model)
	computeBeforeLBMKernel();


#ifdef AA_PATTERN
	uint8_t output_df = df_cur;
#endif
#ifdef AB_PATTERN
	uint8_t output_df = df_out;
#endif

#ifdef USE_CUDA
	#ifdef HAVE_MPI
	if (nse.nproc == 1)
	{
	#endif
		timer_compute.start();
		for (auto& block : nse.blocks)
		{
			const dim3 blockSize = block.block_size;
			const dim3 gridSize = get_grid_size(block);
			cudaLBMKernel< NSE ><<<gridSize, blockSize>>>(block.data, nse.rank, nse.nproc, (idx) 0);
		}
		cudaDeviceSynchronize();
		checkCudaDevice;
		// copying of overlaps is not necessary for nproc == 1 (nproc is checked in streaming as well)
		timer_compute.stop();
	#ifdef HAVE_MPI
	}
	else
	{
		timer_compute.start();
		timer_compute_overlaps.start();

		for (auto& block : nse.blocks)
		{
			const dim3 blockSize = block.block_size;
			const dim3 gridSizeForBoundary(block.df_overlap_X(), block.local.y()/block.block_size.y, block.local.z());
			const dim3 gridSizeForInternal(block.local.x() - 2*block.df_overlap_X(), block.local.y()/block.block_size.y, block.local.z());

			// compute on boundaries (NOTE: 1D distribution is assumed)
			cudaLBMKernel< NSE ><<<gridSizeForBoundary, blockSize, 0, cuda_streams[0]>>>(block.data, block.id, nse.total_blocks, (idx) 0);
			cudaLBMKernel< NSE ><<<gridSizeForBoundary, blockSize, 0, cuda_streams[1]>>>(block.data, block.id, nse.total_blocks, block.local.x() - block.df_overlap_X());

			// compute on internal lattice sites
			cudaLBMKernel< NSE ><<<gridSizeForInternal, blockSize, 0, cuda_streams[2]>>>(block.data, block.id, nse.total_blocks, block.df_overlap_X());
		}

		// wait for the computations on boundaries to finish
		cudaStreamSynchronize(cuda_streams[0]);
		cudaStreamSynchronize(cuda_streams[1]);
		timer_compute_overlaps.stop();

		// start communication of the latest DFs and dmacro on overlaps
		timer_wait_communication.start();
		nse.synchronizeDFsDevice_start(output_df);
		if (NSE::MACRO::use_syncMacro)
			nse.synchronizeMacroDevice_start();

		// wait for the communication to finish
		// (it is important to do this before waiting for the computation, otherwise MPI won't progress)
		for (auto& block : nse.blocks)
			block.waitAllCommunication();
		timer_wait_communication.stop();

		// wait for the computation on the interior to finish
		timer_wait_computation.start();
		cudaStreamSynchronize(cuda_streams[2]);

		// synchronize the whole GPU and check errors
		cudaDeviceSynchronize();
		checkCudaDevice;
		timer_wait_computation.stop();

		timer_compute.stop();
	}
	#endif
#else
	timer_compute.start();
	for (auto& block : nse.blocks)
	{
		#pragma omp parallel for schedule(static) collapse(2)
		for (idx x=0; x<block.local.x(); x++)
		for (idx z=0; z<block.local.z(); z++)
		for (idx y=0; y<block.local.y(); y++)
		{
			LBMKernel< NSE >(block.data, x, y, z, nse.rank, nse.nproc);
		}
	}
	timer_compute.stop();
	#ifdef HAVE_MPI
	// TODO: overlap computation with synchronization, just like above
	timer_wait_communication.start();
	nse.synchronizeDFsDevice(output_df);
	timer_wait_communication.stop();
	#endif
#endif

	nse.iterations++;

	bool doit=false;
	for (int c=0;c<MAX_COUNTER;c++) if (c!=PRINT && c!=SAVESTATE) if (cnt[c].action(nse.physTime())) doit = true;
	if (doit)
	{
		// common copy
		nse.copyMacroToHost();
		// to be able to compute rho, vx, vy, vz etc... based on DFs on CPU to save GPU memory FIXME may not work with ESOTWIST
		if (NSE::CPU_MACRO::N>0)
			nse.copyDFsToHost(output_df);
	}

	timer_SimUpdate.stop();
}

template< typename NSE >
void State<NSE>::AfterSimUpdate()
{
	timer_AfterSimUpdate.start();

	// call hook method (used e.g. for the coupled LBM-MHFEM solver)
	computeAfterLBMKernel();

	bool write_info = false;

	if (cnt[VTK1D].action(nse.physTime()) ||
	    cnt[VTK2D].action(nse.physTime()) ||
	    cnt[VTK3D].action(nse.physTime()) ||
	    cnt[VTK3DCUT].action(nse.physTime()) ||
	    cnt[PROBE1].action(nse.physTime()) ||
	    cnt[PROBE2].action(nse.physTime()) ||
	    cnt[PROBE3].action(nse.physTime())
	    )
	{
		// cpu macro
		nse.computeCPUMacroFromLat();
		// probe1
		if (cnt[PROBE1].action(nse.physTime()))
		{
			probe1();
			cnt[PROBE1].count++;
		}
		// probe2
		if (cnt[PROBE2].action(nse.physTime()))
		{
			probe2();
			cnt[PROBE2].count++;
		}
		// probe3
		if (cnt[PROBE3].action(nse.physTime()))
		{
			probe3();
			cnt[PROBE3].count++;
		}
		// 3D VTK
		if (cnt[VTK3D].action(nse.physTime()))
		{
			writeVTKs_3D();
			cnt[VTK3D].count++;
		}
		// 3D VTK CUT
		if (cnt[VTK3DCUT].action(nse.physTime()))
		{
			writeVTKs_3Dcut();
			cnt[VTK3DCUT].count++;
		}
		// 2D VTK
		if (cnt[VTK2D].action(nse.physTime()))
		{
			writeVTKs_2D();
			cnt[VTK2D].count++;
		}
		// 1D VTK
		if (cnt[VTK1D].action(nse.physTime()))
		{
			writeVTKs_1D();
			cnt[VTK1D].count++;
		}
		write_info = true;
	}

	if (cnt[PRINT].action(nse.physTime()))
	{
		write_info = true;
		cnt[PRINT].count++;
	}

	// statReset is called after all probes and VTK output
	// copy macro from host to device after reset
	if (cnt[STAT_RESET].action(nse.physTime()))
	{
		statReset();
		nse.copyMacroToDevice();
		cnt[STAT_RESET].count++;
	}
	if (cnt[STAT2_RESET].action(nse.physTime()))
	{
		stat2Reset();
		nse.copyMacroToDevice();
		cnt[STAT2_RESET].count++;
	}

	// only the first process writes GLUPS
	// getting the rank from MPI_COMM_WORLD is intended here - other ranks may be redirected to a file when the ranks are reordered
	if (TNL::MPI::GetRank(MPI_COMM_WORLD) == 0)
	if (nse.iterations > 1)
	if (write_info)
	{
		// get time diff
		const double now = timer_total.getRealTime();
		const double timediff = TNL::max(1e-6, now - glups_prev_time);

		// to avoid numerical errors - split LUPS computation in two parts
		double LUPS = (nse.iterations - glups_prev_iterations) / timediff;
		LUPS *= nse.lat.global.x() * nse.lat.global.y() * nse.lat.global.z();

		// save prev time and iterations
		glups_prev_time = now;
		glups_prev_iterations = nse.iterations;

		// simple estimate of time of accomplishment
		double ETA = getWallTime() * (nse.physFinalTime - nse.physTime()) / (nse.physTime() - nse.physStartTime);

		if (verbosity > 0)
		{
			log("GLUPS=%.3f iter=%d t=%1.3fs dt=%1.2e lbmVisc=%1.2e WT=%.0fs ETA=%.0fs",
				LUPS * 1e-9,
				nse.iterations,
				nse.physTime(),
				nse.physDt,
				nse.lbmViscosity(),
				getWallTime(),
				ETA
			);
		}
	}

	timer_AfterSimUpdate.stop();
}

template< typename NSE >
void State<NSE>::AfterSimFinished()
{
	// only the first process writes the info
	if (TNL::MPI::GetRank(MPI_COMM_WORLD) == 0)
	if (nse.iterations > 1)
	if (verbosity > 0)
	{
		log("total walltime: %.1f s, SimInit time: %.1f s, SimUpdate time: %.1f s, AfterSimUpdate time: %.1f s",
			getWallTime(),
			timer_SimInit.getRealTime(),
			timer_SimUpdate.getRealTime(),
			timer_AfterSimUpdate.getRealTime()
		);
		log("compute time: %.1f s, compute overlaps time: %.1f s, wait for communication time: %.1f s, wait for computation time: %.1f s",
			timer_compute.getRealTime(),
			timer_compute_overlaps.getRealTime(),
			timer_wait_communication.getRealTime(),
			timer_wait_computation.getRealTime()
		);
		const double avgLUPS = nse.lat.global.x() * nse.lat.global.y() * nse.lat.global.z() * (nse.iterations / (timer_SimUpdate.getRealTime() + timer_AfterSimUpdate.getRealTime()));
		const double computeLUPS = nse.lat.global.x() * nse.lat.global.y() * nse.lat.global.z() * (nse.iterations / timer_compute.getRealTime());
		log("final GLUPS: average (based on SimInit + SimUpdate + AfterSimUpdate time): %.3f, based on compute time: %.3f",
			avgLUPS * 1e-9,
			computeLUPS * 1e-9
		);
	}
}

template< typename NSE >
void State<NSE>::updateKernelData()
{
	nse.updateKernelData();

	// this is not in nse.updateKernelData so that it can be overridden for ADE
	for( auto& block : nse.blocks )
		block.data.lbmViscosity = nse.lbmViscosity();
}

template< typename NSE >
void State<NSE>::copyAllToDevice()
{
	nse.copyMapToDevice();
	nse.copyDFsToDevice();
	nse.copyMacroToDevice();  // important when a state has been loaded
}

template< typename NSE >
void State<NSE>::copyAllToHost()
{
	nse.copyMapToHost();
	nse.copyDFsToHost();
	nse.copyMacroToHost();
}
