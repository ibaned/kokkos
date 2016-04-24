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

#include <Kokkos_HostSpace.hpp>

#include <mpi.h>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

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

  /**\brief  Allocate untracked memory in the shmem space */
  void * allocate( const size_t arg_alloc_size ) const ;

  /**\brief  Deallocate untracked memory in the shmem space */
  void deallocate( void * const arg_alloc_ptr
                 , const size_t arg_alloc_size ) const ;
};

} // namespace Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template<class ExecutionSpace>
struct DeepCopy<Experimental::MpiShmemSpace,HostSpace,ExecutionSpace> {
  DeepCopy( void * dst , const void * src , size_t n ) {
    memcpy( dst , src , n );
  }
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n ) {
    exec.fence();
    memcpy( dst , src , n );
  }
};

template<class ExecutionSpace>
struct DeepCopy<HostSpace,Experimental::MpiShmemSpace,ExecutionSpace> {
  DeepCopy( void * dst , const void * src , size_t n ) {
    memcpy( dst , src , n );
  }
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n ) {
    exec.fence();
    memcpy( dst , src , n );
  }
};

template<class ExecutionSpace>
struct DeepCopy<Experimental::MpiShmemSpace,Experimental::MpiShmemSpace,ExecutionSpace> {
  DeepCopy( void * dst , const void * src , size_t n ) {
    memcpy( dst , src , n );
  }
  DeepCopy( const ExecutionSpace& exec, void * dst , const void * src , size_t n ) {
    exec.fence();
    memcpy( dst , src , n );
  }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** Running in MpiShmemSpace attempting to access HostSpace: ok */
template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::Experimental::MpiShmemSpace , Kokkos::HostSpace >
{
  enum { value = true };
  inline static void verify( void ) {}
  inline static void verify( const void * ) { }
};

template<>
struct VerifyExecutionCanAccessMemorySpace< Kokkos::HostSpace , Kokkos::Experimental::MpiShmemSpace >
{
  enum { value = true };
  inline static void verify( void ) { }
  inline static void verify( const void * ) { }
};

} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

template<>
class SharedAllocationRecord< Kokkos::Experimental::MpiShmemSpace , void >
  : public SharedAllocationRecord< void , void >
{
private:
  typedef SharedAllocationRecord< void , void >  RecordBase ;

  SharedAllocationRecord( const SharedAllocationRecord & ) = delete ;
  SharedAllocationRecord & operator = ( const SharedAllocationRecord & ) = delete ;

  static void deallocate( RecordBase * );

  static RecordBase s_root_record ;

  MPI_Win                 m_win ;
  const Kokkos::Experimental::MpiShmemSpace m_space ;

protected:

  ~SharedAllocationRecord();
  SharedAllocationRecord() : RecordBase(), m_win(MPI_WIN_NULL), m_space() {}

  SharedAllocationRecord( const Kokkos::Experimental::MpiShmemSpace & arg_space
                        , const std::string              & arg_label
                        , const size_t                     arg_alloc_size
                        , const RecordBase::function_type  arg_dealloc = & deallocate
                        );

public:

  std::string get_label() const ;

  static SharedAllocationRecord * allocate( const Kokkos::Experimental::MpiShmemSpace &  arg_space
                                          , const std::string       &  arg_label
                                          , const size_t               arg_alloc_size );

  /**\brief  Allocate tracked memory in the space */
  static
  void * allocate_tracked( const Kokkos::Experimental::MpiShmemSpace & arg_space
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

  static void print_records( std::ostream & , const Kokkos::MpiShmemSpace & , bool detail = false );
};

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

#endif /* #if defined( KOKKOS_HAVE_MPISHMEM ) */
#endif /* #define KOKKOS_MPISHMEMSPACE_HPP */

