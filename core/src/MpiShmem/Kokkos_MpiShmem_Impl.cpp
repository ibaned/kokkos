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

/*--------------------------------------------------------------------------*/
/* Kokkos interfaces */

#include <Kokkos_Core.hpp>

/* only compile this file if MPISHMEM is enabled for Kokkos */
#ifdef KOKKOS_HAVE_MPISHMEM

#include <MpiShmem/Kokkos_MpiShmem_Internal.hpp>
#include <impl/Kokkos_AllocationTracker.hpp>
#include <impl/Kokkos_Error.hpp>

/*--------------------------------------------------------------------------*/
/* Standard 'C' libraries */
#include <stdlib.h>

/* Standard 'C++' libraries */
#include <iostream>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl  {

// put MpiShmemInternal here as a singleton

}// namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

MpiShmem::size_type MpiShmem::detect_device_count()
{ return Impl::MpiShmemInternalDevices::singleton().m_cudaDevCount ; }

int MpiShmem::concurrency() {
  return 131072;
}

int MpiShmem::is_initialized()
{ return Impl::MpiShmemInternal::singleton().is_initialized(); }

void MpiShmem::initialize( const MpiShmem::SelectDevice config , size_t num_instances )
{ Impl::MpiShmemInternal::singleton().initialize( config.cuda_device_id , num_instances ); }

MpiShmem::size_type MpiShmem::device_arch()
{
  const int dev_id = Impl::MpiShmemInternal::singleton().m_cudaDev ;

  int dev_arch = 0 ;

  if ( 0 <= dev_id ) {
    const struct cudaDeviceProp & cudaProp =
      Impl::MpiShmemInternalDevices::singleton().m_cudaProp[ dev_id ] ;

    dev_arch = cudaProp.major * 100 + cudaProp.minor ;
  }

  return dev_arch ;
}

void MpiShmem::finalize()
{ Impl::MpiShmemInternal::singleton().finalize(); }

MpiShmem::MpiShmem()
  : m_device( Impl::MpiShmemInternal::singleton().m_cudaDev )
  , m_stream( 0 )
{
  Impl::MpiShmemInternal::singleton().verify_is_initialized( "MpiShmem instance constructor" );
}

MpiShmem::MpiShmem( const int instance_id )
  : m_device( Impl::MpiShmemInternal::singleton().m_cudaDev )
  , m_stream(
      Impl::MpiShmemInternal::singleton().verify_is_initialized( "MpiShmem instance constructor" )
        ? Impl::MpiShmemInternal::singleton().m_stream[ instance_id % Impl::MpiShmemInternal::singleton().m_streamCount ]
        : 0 )
{}

void MpiShmem::print_configuration( std::ostream & s , const bool )
{ Impl::MpiShmemInternal::singleton().print_configuration( s ); }

bool MpiShmem::sleep() { return false ; }

bool MpiShmem::wake() { return true ; }

void MpiShmem::fence()
{
  Kokkos::Impl::cuda_device_synchronize();
}

} // namespace Kokkos

#endif // KOKKOS_HAVE_MPISHMEM
//----------------------------------------------------------------------------


