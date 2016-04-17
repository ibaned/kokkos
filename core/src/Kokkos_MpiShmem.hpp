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

#ifndef KOKKOS_MPISHMEM_HPP
#define KOKKOS_MPISHMEM_HPP

#include <Kokkos_Core_fwd.hpp>

// If MPI execution space is enabled then use this header file.

#if defined( KOKKOS_HAVE_MPISHMEM )

#include <iosfwd>

#include <Kokkos_MpiShmemSpace.hpp>

#include <Kokkos_Parallel.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_Tags.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {
namespace Impl {
class MpiShmemExec ;
} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

/// \class MpiShmem
/// \brief Kokkos Execution Space that uses MPI shared memory communicators.
///
/// An "execution space" represents a parallel execution model.  It tells Kokkos
/// how to parallelize the execution of kernels in a parallel_for or
/// parallel_reduce.  For example, the Threads execution space uses Pthreads or
/// C++11 threads on a CPU, the OpenMP execution space uses the OpenMP language
/// extensions, and the Serial execution space executes "parallel" kernels
/// sequentially.  The MpiShmem execution space uses MPI's
/// shared memory communicator splitting to execute kernels across MPI
/// ranks on the same CPU.
class MpiShmem {
public:
  //! \name Type declarations that all Kokkos execution spaces must provide.
  //@{

  //! Tag this class as a kokkos execution space
  typedef MpiShmem              execution_space ;

  //! This execution space's preferred memory space.
  typedef MpiShmemSpace         memory_space ;

  //! This execution space preferred device_type
  typedef Kokkos::Device<execution_space,memory_space> device_type;

  //! The size_type best suited for this execution space.
  typedef memory_space::size_type  size_type ;

  //! This execution space's preferred array layout.
  typedef LayoutRight            array_layout ;

  //!
  typedef ScratchMemorySpace< MpiShmem >  scratch_memory_space ;

  //@}
  //--------------------------------------------------
  //! \name Functions that all Kokkos devices must implement.
  //@{

  /// \brief True if and only if this method is being called in a
  ///   thread-parallel function.
  inline static int in_parallel() {
    TODO!!!
  }

  /** \brief  Set the device in a "sleep" state. A noop for MpiShmem.  */
  static bool sleep();

  /** \brief Wake the device from the 'sleep' state. A noop for MpiShmem. */
  static bool wake();

  /// \brief Wait until all dispatched functors complete.
  ///
  /// This is currently a noop for MpiShmem, although we could
  /// do something like MPI_Ireduce in the future.
  static void fence();

  //! Free any resources being consumed by the device.
  static void finalize();

  //! Has been initialized
  static int is_initialized();

  /** \brief  Return the maximum amount of concurrency.  */
  static int concurrency();

  //! Print configuration information to the given output stream.
  static void print_configuration( std::ostream & , const bool detail = false );

  //@}
  //--------------------------------------------------
  //! \name  MpiShmem space instances
  //@{

  ~MpiShmem();
  MpiShmem();
  explicit MpiShmem( MPI_Comm world_comm
                   , MPI_Comm team_comm );

  MpiShmem( MpiShmem && ) = default ;
  MpiShmem( const MpiShmem & ) = default ;
  MpiShmem & operator = ( MpiShmem && ) = default ;
  MpiShmem & operator = ( const MpiShmem & ) = default ;

  //! Initialize, telling the MPI run-time library which device to use.
  static void initialize( MPI_Comm world_comm = MPI_COMM_WORLD
                        , MPI_Comm team_comm = MPI_COMM_NULL );

  static MPI_Comm team_comm();
  static MPI_Comm world_comm();
  /// \brief Get which team in the world this proc belongs to
  static int which_team();

  static int team_rank();
  static int team_size();
  static int world_rank();
  static int world_size();

  //@}
  //--------------------------------------------------------------------------
};

} // namespace Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template<>
struct VerifyExecutionCanAccessMemorySpace
  < Kokkos::Experimental::MpiShmemSpace
  , Kokkos::Experimental::MpiShmem::scratch_memory_space
  >
{
  enum { value = true };
  KOKKOS_INLINE_FUNCTION static void verify( void ) { }
  KOKKOS_INLINE_FUNCTION static void verify( const void * ) { }
};

template<>
struct VerifyExecutionCanAccessMemorySpace
  < Kokkos::HostSpace
  , Kokkos::Experimental::MpiShmem::scratch_memory_space
  >
{
  enum { value = true };
  inline static void verify( void ) { }
  inline static void verify( const void * ) { }
};

} // namespace Impl
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

#include <MpiShmem/Kokkos_MpiShmemExec.hpp>
#include <MpiShmem/Kokkos_MpiShmem_Parallel.hpp>

//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_HAVE_MPI ) */
#endif /* #ifndef KOKKOS_MPI_HPP */
