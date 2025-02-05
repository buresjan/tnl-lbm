#pragma once

#include <fmt/format.h>

#include "adios_writer.h"

template<typename TRAITS>
ADIOSWriter<TRAITS>::ADIOSWriter(
	TNL::MPI::Comm communicator,
	const std::string& basename,
	idx3d global,
	idx3d local,
	idx3d offset,
	point_t physOrigin,
	real physDl,
	int cycle
)
#ifdef HAVE_MPI
: adios(communicator)
#else
: adios()
#endif
{
	io = adios.DeclareIO("io");
	io.SetEngine("BP4");
	filename = basename + ".bp";

	if (cycle==0)
		engine = io.Open(filename, adios2::Mode::Write);
	else
		engine = io.Open(filename, adios2::Mode::Append);

	this->global = global;
	this->local = local;
	this->offset = offset;
	this->physOrigin = physOrigin;
	this->physDl = physDl;

	engine.BeginStep();
}

template<typename TRAITS>
template<typename T>
void ADIOSWriter<TRAITS>::write(std::string varName, T val)
{
	recordVariable(varName, 0);

	adios2::Variable<T> value = io.DefineVariable<T>(varName);

	engine.Put(value,val);
	engine.PerformPuts();
}

template<typename TRAITS>
template<typename T>
void ADIOSWriter<TRAITS>::write(std::string varName, std::vector<T>& val, int dim)
{
	recordVariable(varName, dim);

	adios2::Dims shape({size_t(global.z()), size_t(global.y()), size_t(global.x())});
	adios2::Dims start({size_t(offset.z()), size_t(offset.y()), size_t(offset.x())});
	adios2::Dims count({size_t(local.z()), size_t(local.y()), size_t(local.x())});
	adios2::Variable<T> values = io.DefineVariable<T>(varName, shape, start, count);

	engine.Put(values,val.data());
	engine.PerformPuts();
}

template<typename TRAITS >
void ADIOSWriter<TRAITS>::recordVariable(const std::string& name, int dim)
{
	if (variables.count(name) > 0)
		throw std::invalid_argument("Variable \"" + name + "\" is already defined.");
	if (dim != 0 && dim != 1 && dim != 3)
		throw std::invalid_argument("Invalid dimension of \"" + name + "\"(" + std::to_string(dim) + ").");

	variables[name] = dim;
}

template<typename TRAITS >
void ADIOSWriter<TRAITS>::addVTKAttributes()
{
	const std::string extentG = fmt::format("0 {} 0 {} 0 {}", global.z(), global.y(), global.x());
	const std::string extentL = fmt::format("0 {} 0 {} 0 {}", local.z(), local.y(), local.x());
	const std::string origin = fmt::format("{} {} {}", physOrigin.x(), physOrigin.y(), physOrigin.z());
	const std::string spacing = fmt::format("{} {} {}", physDl, physDl, physDl);

	std::string dataArrays;
	for (const auto& [name, dim] : variables)
	{
		switch (dim)
		{
			case 0:
				dataArrays += "<DataArray Name=\"" + name + "\"> " + name + " </DataArray>\n";
				break;
			case 1:
				dataArrays += "<DataArray Name=\"" + name + "\"/>\n";
				break;
			case 3:
				dataArrays += "<DataArray Name=\"" + name + "\"/>\n";
				//dataArrays += "<DataArray Name=\"" + name + "\" NumberOfComponents=\"3\"/>\n";
				break;
		}
	}

	const std::string imageData = R"(
		<?xml version="1.0"?>
		<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
			<ImageData WholeExtent=")" + extentG + R"(" Origin=")" + origin + R"(" Spacing=")" + spacing + R"(">
				<Piece Extent=")" + extentL + R"(">
					<CellData Scalars="data">)"
					+ dataArrays + R"(
					</CellData>
				</Piece>
			</ImageData>
		</VTKFile>)";

	io.DefineAttribute<std::string>("vtk.xml", imageData);
}

template<typename TRAITS >
ADIOSWriter<TRAITS>::~ADIOSWriter()
{
	if (!variables.empty()) {
		addVTKAttributes();
	}

	engine.EndStep();
	engine.Close();
}
