#pragma once

#include <TNL/Containers/StaticVector.h>

/**
 * \brief Minimal class representing an equidistant D-dimensional lattice.
 *
 * It does not have a \e Device template argument, because it does not describe
 * a data structure - it contains only the information describing the physical
 * domain of the lattice (i.e., the metadata of the \e ImageData tag in the VTI
 * file format). Note that VTK supports only 1, 2, or 3-dimensional lattices.
 * The class is usable as \e Mesh in the \ref TNL::Meshes::Writers::VTIWriter.
 */
template< int D_ = 3, typename real = float, typename idx = int >
struct Lattice
{
   // these type names are required for compatibility with TNL
   using RealType = real;
   using GlobalIndexType = idx;
   using PointType = TNL::Containers::StaticVector< D_, real >;
   using CoordinatesType = TNL::Containers::StaticVector< D_, idx >;

   //! \brief Dimension of the lattice.
   static constexpr int D = D_;

   //! \brief Global size of the lattice.
   CoordinatesType global = 0;

   //! \brief Size of the local extent on the global lattice.
//   CoordinatesType local = 0;

   //! \brief Offset of the local extent on the global lattice.
//   CoordinatesType offset = 0;

   //! \brief Physical coordinates of the left bottom front lattice site.
   PointType physOrigin = 0;

   //! \brief Spatial step, i.e. the distance between two neighboring lattice
   //! sites in physical coordinates.
   RealType physDl = 0;

   // getters for physical coordinates (note that here x,y,z are *global* lattice indices)
   PointType lbm2physPoint(idx x, idx y, idx z) const { return physOrigin + (PointType(x, y, z) - 0.5) * physDl; }
   real lbm2physX(idx x) const { return physOrigin.x() + (x-0.5) * physDl; }
   real lbm2physY(idx y) const { return physOrigin.y() + (y-0.5) * physDl; }
   real lbm2physZ(idx z) const { return physOrigin.z() + (z-0.5) * physDl; }

   // physical to lattice coordinates (but still real rather than idx, rounding can be done later)
   PointType phys2lbmPoint(PointType p) const { return (p - physOrigin) / physDl + 0.5; }
   real phys2lbmX(real x) const { return (x - physOrigin.x()) / physDl + 0.5; }
   real phys2lbmY(real y) const { return (y - physOrigin.y()) / physDl + 0.5; }
   real phys2lbmZ(real z) const { return (z - physOrigin.z()) / physDl + 0.5; }


   //! \brief Returns the spatial dimension of the lattice.
   static constexpr int getMeshDimension()
   {
      return D;
   }

   //! \brief Returns the spatial dimension of the lattice.
   static constexpr int getDimension()
   {
      return D;
   }

   //! \brief Returns the global lattice size.
   const CoordinatesType& size() const
   {
      return global;
   }

   //! \brief Returns the size of the **grid** represented by the lattice
   //! (i.e., the number of voxels between the lattice sites).
   const CoordinatesType& getDimensions() const
   {
      return global - 1;
   }

   //! \brief Returns the origin of the lattice.
   const PointType& getOrigin() const
   {
      return physOrigin;
   }

   //! \brief Returns the space steps of the grid/lattice.
   const PointType& getSpaceSteps() const
   {
      return physDl;
   }
};
