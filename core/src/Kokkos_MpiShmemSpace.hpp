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

#ifndef KOKKOS_MPISHMEMSPACE_HPP
#define KOKKOS_MPISHMEMSPACE_HPP

#include <Kokkos_Core_fwd.hpp>

#if defined( KOKKOS_HAVE_MPISHMEM )

#include <iosfwd>
#include <typeinfo>
#include <string>

#include <Kokkos_HostSpace.hpp>

#include <impl/Kokkos_AllocationTracker.hpp>

#include <mpi.h>

/*--------------------------------------------------------------------------*/

namespace Kokkos {

/** \brief  MpiShmem on-device memory management */

class MpiShmemSpace {
public:

  //! Tag this class as a kokkos memory space
  typedef MpiShmemSpace             memory_space ;
  typedef Kokkos::MpiShmem          execution_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  typedef MPI_AInt                  size_type ;

  /*--------------------------------*/

  MpiShmemSpace();
  MpiShmemSpace( MpiShmemSpace && rhs ) = default ;
  MpiShmemSpace( const MpiShmemSpace & rhs ) = default ;
  MpiShmemSpace & operator = ( MpiShmemSpace && rhs ) = default ;
  MpiShmemSpace & operator = ( const MpiShmemSpace & rhs ) = default ;
  ~MpiShmemSpace() = default ;

  /**\brief  Allocate untracked memory in the cuda space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the cuda space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /*--------------------------------*/
  /** \brief  Error reporting for HostSpace attempt to access MpiShmemSpace */
  static void access_error();
  static void access_error( const void * const );

private:

  MPI_Comm  team_comm ; ///< Which MpiShmem device

  // friend class Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::MpiShmemSpace , void > ;
};

namespace Impl {
/// \brief Initialize lock array for arbitrary size atomics.
///
/// Arbitrary atomics are implemented using a hash table of locks
/// where the hash value is derived from the address of the
/// object for which an atomic operation is performed.
/// This function initializes the locks to zero (unset).
void init_lock_array_cuda_space();

/// \brief Retrieve the pointer to the lock array for arbitrary size atomics.
///
/// Arbitrary atomics are implemented using a hash table of locks
/// where the hash value is derived from the address of the
/// object for which an atomic operation is performed.
/// This function retrieves the lock array pointer.
/// If the array is not yet allocated it will do so.
int* lock_array_cuda_space_ptr(bool deallocate = false);
}
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

/** \brief  MpiShmem memory that is accessible to Host execution space
 *          through MpiShmem's unified virtual memory (UVM) runtime.
 */
class MpiShmemUVMSpace {
public:

  //! Tag this class as a kokkos memory space
  typedef MpiShmemUVMSpace          memory_space ;
  typedef MpiShmem                  execution_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;
  typedef unsigned int          size_type ;

  /** \brief  If UVM capability is available */
  static bool available();

  /*--------------------------------*/

#if ! KOKKOS_USING_EXP_VIEW

  typedef Impl::MpiShmemUVMAllocator allocator;

  /** \brief  Allocate a contiguous block of memory.
   *
   *  The input label is associated with the block of memory.
   *  The block of memory is tracked via reference counting where
   *  allocation gives it a reference count of one.
   */
  static Impl::AllocationTracker allocate_and_track( const std::string & label, const size_t size );


  /** \brief  MpiShmem specific function to attached texture object to an allocation.
   *          Output the texture object, base pointer, and offset from the input pointer.
   */
#if defined( __MPISHMEMCC__ )
  static void texture_object_attach(  Impl::AllocationTracker const & tracker
                                    , unsigned type_size
                                    , ::cudaChannelFormatDesc const & desc
                                   );
#endif

#endif /* #if ! KOKKOS_USING_EXP_VIEW */

  /*--------------------------------*/

  MpiShmemUVMSpace();
  MpiShmemUVMSpace( MpiShmemUVMSpace && rhs ) = default ;
  MpiShmemUVMSpace( const MpiShmemUVMSpace & rhs ) = default ;
  MpiShmemUVMSpace & operator = ( MpiShmemUVMSpace && rhs ) = default ;
  MpiShmemUVMSpace & operator = ( const MpiShmemUVMSpace & rhs ) = default ;
  ~MpiShmemUVMSpace() = default ;

  /**\brief  Allocate untracked memory in the cuda space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the cuda space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /*--------------------------------*/

private:

  int  m_device ; ///< Which MpiShmem device
};

} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

/** \brief  Host memory that is accessible to MpiShmem execution space
 *          through MpiShmem's host-pinned memory allocation.
 */
class MpiShmemHostPinnedSpace {
public:

  //! Tag this class as a kokkos memory space
  /** \brief  Memory is in HostSpace so use the HostSpace::execution_space */
  typedef HostSpace::execution_space  execution_space ;
  typedef MpiShmemHostPinnedSpace         memory_space ;
  typedef Kokkos::Device<execution_space,memory_space> device_type;
  typedef unsigned int                size_type ;

  /*--------------------------------*/

#if ! KOKKOS_USING_EXP_VIEW

  typedef Impl::MpiShmemHostAllocator allocator ;

  /** \brief  Allocate a contiguous block of memory.
   *
   *  The input label is associated with the block of memory.
   *  The block of memory is tracked via reference counting where
   *  allocation gives it a reference count of one.
   */
  static Impl::AllocationTracker allocate_and_track( const std::string & label, const size_t size );

#endif /* #if ! KOKKOS_USING_EXP_VIEW */

  /*--------------------------------*/

  MpiShmemHostPinnedSpace();
  MpiShmemHostPinnedSpace( MpiShmemHostPinnedSpace && rhs ) = default ;
  MpiShmemHostPinnedSpace( const MpiShmemHostPinnedSpace & rhs ) = default ;
  MpiShmemHostPinnedSpace & operator = ( MpiShmemHostPinnedSpace && rhs ) = default ;
  MpiShmemHostPinnedSpace & operator = ( const MpiShmemHostPinnedSpace & rhs ) = default ;
  ~MpiShmemHostPinnedSpace() = default ;

  /**\brief  Allocate untracked memory in the space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;

  /*--------------------------------*/
};

} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

void DeepCopyAsyncMpiShmem( void * dst , const void * src , size_t n);

template<> struct DeepCopy< MpiShmemSpace , MpiShmemSpace , MpiShmem>
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const MpiShmem & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< MpiShmemSpace , HostSpace , MpiShmem >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const MpiShmem & , void * dst , const void * src , size_t );
};

template<> struct DeepCopy< HostSpace , MpiShmemSpace , MpiShmem >
{
  DeepCopy( void * dst , const void * src , size_t );
  DeepCopy( const MpiShmem & , void * dst , const void * src , size_t );
};

template<class ExecutionSpace> struct DeepCopy< MpiShmemSpace , MpiShmemSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< MpiShmemSpace , MpiShmemSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< MpiShmemSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< MpiShmemSpace , HostSpace , MpiShmem>( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< HostSpace , MpiShmemSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , MpiShmemSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< MpiShmemSpace , MpiShmemUVMSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< MpiShmemSpace , MpiShmemSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< MpiShmemSpace , MpiShmemHostPinnedSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< MpiShmemSpace , HostSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};


template<class ExecutionSpace>
struct DeepCopy< MpiShmemUVMSpace , MpiShmemSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< MpiShmemSpace , MpiShmemSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< MpiShmemUVMSpace , MpiShmemUVMSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< MpiShmemSpace , MpiShmemSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

template<class ExecutionSpace>
struct DeepCopy< MpiShmemUVMSpace , MpiShmemHostPinnedSpace , ExecutionSpace>
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< MpiShmemSpace , HostSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< MpiShmemUVMSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< MpiShmemSpace , HostSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};


template<class ExecutionSpace> struct DeepCopy< MpiShmemHostPinnedSpace , MpiShmemSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , MpiShmemSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< MpiShmemHostPinnedSpace , MpiShmemUVMSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , MpiShmemSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< MpiShmemHostPinnedSpace , MpiShmemHostPinnedSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , HostSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< MpiShmemHostPinnedSpace , HostSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , HostSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};


template<class ExecutionSpace> struct DeepCopy< HostSpace , MpiShmemUVMSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , MpiShmemSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

template<class ExecutionSpace> struct DeepCopy< HostSpace , MpiShmemHostPinnedSpace , ExecutionSpace >
{
  inline
  DeepCopy( void * dst , const void * src , size_t n )
  { (void) DeepCopy< HostSpace , HostSpace , MpiShmem >( dst , src , n ); }

  inline
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n )
  {
    exec.fence();
    DeepCopyAsyncMpiShmem (dst,src,n);
  }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** Running in MpiShmemSpace attempting to access HostSpace: error */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::MpiShmemSpace , Kokkos::HostSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("MpiShmem code attempted to access HostSpace memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("MpiShmem code attempted to access HostSpace memory"); }
};

/** Running in MpiShmemSpace accessing MpiShmemUVMSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::MpiShmemSpace , Kokkos::MpiShmemUVMSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in MpiShmemSpace accessing MpiShmemHostPinnedSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::MpiShmemSpace , Kokkos::MpiShmemHostPinnedSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

/** Running in MpiShmemSpace attempting to access an unknown space: error */
template< class OtherSpace >
struct VerifyExecutionCanAccessMemorySpace<
  typename enable_if< ! is_same<Kokkos::MpiShmemSpace,OtherSpace>::value , Kokkos::MpiShmemSpace >::type ,
  OtherSpace >
{
  enum { value = false };
  KOKKOS_INLINE_FUNCTION static void verify( void )
    { Kokkos::abort("MpiShmem code attempted to access unknown Space memory"); }

  KOKKOS_INLINE_FUNCTION static void verify( const void * )
    { Kokkos::abort("MpiShmem code attempted to access unknown Space memory"); }
};

//----------------------------------------------------------------------------
/** Running in HostSpace attempting to access MpiShmemSpace */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::MpiShmemSpace >
{
  enum { value = false };
  inline static void verify( void ) { MpiShmemSpace::access_error(); }
  inline static void verify( const void * p ) { MpiShmemSpace::access_error(p); }
};

/** Running in HostSpace accessing MpiShmemUVMSpace is OK */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::MpiShmemUVMSpace >
{
  enum { value = true };
  inline static void verify( void ) { }
  inline static void verify( const void * ) { }
};

/** Running in HostSpace accessing MpiShmemHostPinnedSpace is OK */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::MpiShmemHostPinnedSpace >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) {}
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) {}
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

template<>
class SharedAllocationRecord< Kokkos::MpiShmemSpace , void >
  : public SharedAllocationRecord< void , void >
{
private:

  friend class SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void > ;

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
  const Kokkos::MpiShmemSpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_tex_obj(0), m_space() {}

  SharedAllocationRecord( const Kokkos::MpiShmemSpace        & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::MpiShmemSpace &  arg_space
                                          , const std::string       &  arg_label
                                          , const size_t               arg_alloc_size );

  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::MpiShmemSpace & arg_space
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
                   , "MpiShmem texture fetch only supported for alias types of int, ::int2, or ::int4" );

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

  static void print_records( std::ostream & , const Kokkos::MpiShmemSpace & , bool detail = false );
};


template<>
class SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >
  : public SharedAllocationRecord< void , void >
{
private:

  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

  static RecordBase s_root_record ;

  ::cudaTextureObject_t      m_tex_obj ;
  const Kokkos::MpiShmemUVMSpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_tex_obj(0), m_space() {}

  SharedAllocationRecord( const Kokkos::MpiShmemUVMSpace     & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::MpiShmemUVMSpace &  arg_space
                                          , const std::string          &  arg_label
                                          , const size_t                  arg_alloc_size
                                          );

  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::MpiShmemUVMSpace & arg_space
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
                   , "MpiShmem texture fetch only supported for alias types of int, ::int2, or ::int4" );

      if ( m_tex_obj == 0 ) {
        m_tex_obj = SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
          attach_texture_object( sizeof(AliasType)
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

  static void print_records( std::ostream & , const Kokkos::MpiShmemUVMSpace & , bool detail = false );
};

template<>
class SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >
  : public SharedAllocationRecord< void , void >
{
private:

  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

  static RecordBase s_root_record ;

  const Kokkos::MpiShmemHostPinnedSpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_space() {}

  SharedAllocationRecord( const Kokkos::MpiShmemHostPinnedSpace     & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::MpiShmemHostPinnedSpace &  arg_space
                                          , const std::string          &  arg_label
                                          , const size_t                  arg_alloc_size
                                          );
  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::MpiShmemHostPinnedSpace & arg_space
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

  static void print_records( std::ostream & , const Kokkos::MpiShmemHostPinnedSpace & , bool detail = false );
};

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_HAVE_MPISHMEM ) */
#endif /* #define KOKKOS_MPISHMEMSPACE_HPP */


