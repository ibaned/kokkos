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

#include <Kokkos_Macros.hpp>

/* only compile this file if MPISHMEM is enabled for Kokkos */
#ifdef KOKKOS_HAVE_MPISHMEM

#include <Kokkos_MpiShmem.hpp>
#include <Kokkos_MpiShmemSpace.hpp>

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {

MpiShmemSpace::MpiShmemSpace()
{
}

void * MpiShmemSpace::allocate( const size_t arg_alloc_size ) const
{
  void * ptr = NULL;

  MPISHMEM_SAFE_CALL( cudaMalloc( &ptr, arg_alloc_size ) );

  return ptr ;
}

void MpiShmemSpace::deallocate( void * const arg_alloc_ptr , const size_t /* arg_alloc_size */ ) const
{
  try {
    MPISHMEM_SAFE_CALL( cudaFree( arg_alloc_ptr ) );
  } catch(...) {}
}

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::s_root_record ;

::cudaTextureObject_t
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
attach_texture_object( const unsigned sizeof_alias
                     , void *   const alloc_ptr
                     , size_t   const alloc_size )
{
  enum { TEXTURE_BOUND_1D = 1u << 27 };

  if ( ( alloc_ptr == 0 ) || ( sizeof_alias * TEXTURE_BOUND_1D <= alloc_size ) ) {
    std::ostringstream msg ;
    msg << "Kokkos::MpiShmemSpace ERROR: Cannot attach texture object to"
        << " alloc_ptr(" << alloc_ptr << ")"
        << " alloc_size(" << alloc_size << ")"
        << " max_size(" << ( sizeof_alias * TEXTURE_BOUND_1D ) << ")" ;
    std::cerr << msg.str() << std::endl ;
    std::cerr.flush();
    Kokkos::Impl::throw_runtime_exception( msg.str() );
  }

  ::cudaTextureObject_t tex_obj ;

  struct cudaResourceDesc resDesc ;
  struct cudaTextureDesc  texDesc ;

  memset( & resDesc , 0 , sizeof(resDesc) );
  memset( & texDesc , 0 , sizeof(texDesc) );

  resDesc.resType                = cudaResourceTypeLinear ;
  resDesc.res.linear.desc        = ( sizeof_alias ==  4 ?  cudaCreateChannelDesc< int >() :
                                   ( sizeof_alias ==  8 ?  cudaCreateChannelDesc< ::int2 >() :
                                  /* sizeof_alias == 16 */ cudaCreateChannelDesc< ::int4 >() ) );
  resDesc.res.linear.sizeInBytes = alloc_size ;
  resDesc.res.linear.devPtr      = alloc_ptr ;

  MPISHMEM_SAFE_CALL( cudaCreateTextureObject( & tex_obj , & resDesc, & texDesc, NULL ) );

  return tex_obj ;
}

std::string
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::get_label() const
{
  SharedAllocationHeader header ;

  Kokkos::Impl::DeepCopy< Kokkos::HostSpace , Kokkos::MpiShmemSpace >( & header , RecordBase::head() , sizeof(SharedAllocationHeader) );

  return std::string( header.m_label );
}

SharedAllocationRecord< Kokkos::MpiShmemSpace , void > *
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
allocate( const Kokkos::MpiShmemSpace &  arg_space
        , const std::string       &  arg_label
        , const size_t               arg_alloc_size
        )
{
  return new SharedAllocationRecord( arg_space , arg_label , arg_alloc_size );
}

void
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
~SharedAllocationRecord()
{
  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
SharedAllocationRecord( const Kokkos::MpiShmemSpace & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      ( & SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::s_root_record
      , reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( sizeof(SharedAllocationHeader) + arg_alloc_size ) )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
  , m_tex_obj( 0 )
  , m_space( arg_space )
{
  SharedAllocationHeader header ;

  // Fill in the Header information
  header.m_record = static_cast< SharedAllocationRecord< void , void > * >( this );

  strncpy( header.m_label
          , arg_label.c_str()
          , SharedAllocationHeader::maximum_label_length
          );

  // Copy to device memory
  Kokkos::Impl::DeepCopy<MpiShmemSpace,HostSpace>::DeepCopy( RecordBase::m_alloc_ptr , & header , sizeof(SharedAllocationHeader) );
}

//----------------------------------------------------------------------------

void * SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
allocate_tracked( const Kokkos::MpiShmemSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  Kokkos::Impl::DeepCopy<MpiShmemSpace,MpiShmemSpace>( r_new->data() , r_old->data()
                                             , std::min( r_old->size() , r_new->size() ) );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

//----------------------------------------------------------------------------

SharedAllocationRecord< Kokkos::MpiShmemSpace , void > *
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::get_record( void * alloc_ptr )
{
  using Header     = SharedAllocationHeader ;
  using RecordBase = SharedAllocationRecord< void , void > ;
  using RecordMpiShmem = SharedAllocationRecord< Kokkos::MpiShmemSpace , void > ;

#if 0
  // Copy the header from the allocation
  Header head ;

  Header const * const head_cuda = alloc_ptr ? Header::get_header( alloc_ptr ) : (Header*) 0 ;

  if ( alloc_ptr ) {
    Kokkos::Impl::DeepCopy<HostSpace,MpiShmemSpace>::DeepCopy( & head , head_cuda , sizeof(SharedAllocationHeader) );
  }

  RecordMpiShmem * const record = alloc_ptr ? static_cast< RecordMpiShmem * >( head.m_record ) : (RecordMpiShmem *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head_cuda ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::get_record ERROR" ) );
  }

#else

  // Iterate the list to search for the record among all allocations
  // requires obtaining the root of the list and then locking the list.

  RecordMpiShmem * const record = static_cast< RecordMpiShmem * >( RecordBase::find( & s_root_record , alloc_ptr ) );

  if ( record == 0 ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::get_record ERROR" ) );
  }

#endif

  return record ;
}

void
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
print_records( std::ostream & s , const Kokkos::MpiShmemSpace & space , bool detail )
{
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "MpiShmemUVM" , & s_root_record , detail );
}

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

#endif // KOKKOS_HAVE_MPISHMEM
