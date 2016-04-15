/*
//@HEADER
// ************************************************************************
// 
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
// 
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
// 
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_MPISPACE_HPP
#define KOKKOS_MPISPACE_HPP

#include <Kokkos_Core_fwd.hpp>

#if defined( KOKKOS_HAVE_MPI )

#include <iosfwd>
#include <typeinfo>
#include <string>

#include <Kokkos_HostSpace.hpp>

#include <impl/Kokkos_AllocationTracker.hpp>

#include <mpi.h>

/*--------------------------------------------------------------------------*/

namespace Kokkos {

/** \brief  MPI shared memory management */

class MpiSpace {
public:

  //! Tag this class as a kokkos memory space
  typedef MpiSpace             memory_space ;
  typedef Kokkos::Mpi          execution_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  typedef MPI_Aint             size_type ;

  /*--------------------------------*/

  MpiSpace();
  MpiSpace( MPI_Comm team_comm );
  MpiSpace( MpiSpace && rhs ) = default ;
  MpiSpace( const MpiSpace & rhs ) = default ;
  MpiSpace & operator = ( MpiSpace && rhs ) = default ;
  MpiSpace & operator = ( const MpiSpace & rhs ) = default ;
  ~MpiSpace() = default ;

  /**\brief  Allocate untracked memory in the cuda space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the cuda space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /*--------------------------------*/
  /** \brief  Error reporting for HostSpace attempt to access MpiSpace */
  static void access_error();
  static void access_error( const void * const );

private:

  MPI_Comm  m_team_comm ; ///< The shared memory communicator
};

} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

void DeepCopyAsyncMpi( void * dst , const void * src , size_t n);

template<> struct DeepCopy< MpiSpace , MpiSpace , Mpi>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Mpi & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< MpiSpace , HostSpace , Mpi >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Mpi & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , MpiSpace , Mpi >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const Mpi & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace> struct DeepCopy< MpiSpace , MpiSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< MpiSpace , MpiSpace , Mpi >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpi (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< MpiSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< MpiSpace , HostSpace , Mpi>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpi (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , MpiSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , MpiSpace , Mpi >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpi (dst,src,n);
  }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** Running in MpiSpace attempting to access HostSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::MpiSpace , Kokkos::HostSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in MpiSpace attempting to access an unknown space: error */
template< class OtherSpace >
struct VerifyExecutionCanAccessMemorySpace<
  typename enable_if< ! is_same<Kokkos::MpiSpace,OtherSpace>::value , Kokkos::MpiSpace >::type ,
  OtherSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("Mpi code attempted to access unknown Space memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("Mpi code attempted to access unknown Space memory"); }
};

/** Running in HostSpace attempting to access MpiSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::MpiSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

template<>
class SharedAllocationRecord< Kokkos::MpiSpace , void >
  : public SharedAllocationRecord< void , void >
{
private:

  friend class SharedAllocationRecord< Kokkos::MpiUVMSpace , void > ;

  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

  static ::cudaTextureObject_t
  attach_texture_object( const unsigned sizeof_alias
                       , void * const   alloc_ptr
                       , const size_t   alloc_size ); 

  static RecordBase s_root_record ;

  ::cudaTextureObject_t   m_tex_obj ;
  const Kokkos::MpiSpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_tex_obj(0), m_space() {}

  SharedAllocationRecord( const Kokkos::MpiSpace        & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::MpiSpace &  arg_space
                                          , const std::string       &  arg_label
                                          , const size_t               arg_alloc_size );

  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::MpiSpace & arg_space
                         , const std::string & arg_label
                         , const size_t arg_alloc_size );

  /**\brief  Reallocate tracked memory in the space */
  static
  void * reallocate_tracked( void * const arg_alloc_ptr
                           , const size_t arg_alloc_size );

  /**\brief  Deallocate tracked memory in the space */
  static
  void deallocate_tracked( void * const arg_alloc_ptr );

  static SharedAllocationRecord * get_record( void * arg_alloc_ptr );

  template< typename AliasType >
  inline
  ::cudaTextureObject_t attach_texture_object()
    {
      static_assert( ( std::is_same< AliasType , int >::value ||
                       std::is_same< AliasType , ::int2 >::value ||
                       std::is_same< AliasType , ::int4 >::value )
                   , "Mpi texture fetch only supported for alias types of int, ::int2, or ::int4" );

      if ( m_tex_obj == 0 ) {
        m_tex_obj = attach_texture_object( sizeof(AliasType)
                                         , (void*) RecordBase::m_alloc_ptr
                                         , RecordBase::m_alloc_size );
      }

      return m_tex_obj ;
    }

  template< typename AliasType >
  inline
  int attach_texture_object_offset( const AliasType * const ptr )
    {
      // Texture object is attached to the entire allocation range
      return ptr - reinterpret_cast<AliasType*>( RecordBase::m_alloc_ptr );
    }

  static void print_records( std::ostream & , const Kokkos::MpiSpace & , bool detail = false );
};

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_HAVE_MPI ) */
#endif /* #define KOKKOS_MPISPACE_HPP */

