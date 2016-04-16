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

void MpiShmemUVMSpace::deallocate( void * const arg_alloc_ptr , const size_t /* arg_alloc_size */ ) const
{
  try {
    MPISHMEM_SAFE_CALL( cudaFree( arg_alloc_ptr ) );
  } catch(...) {}
}

void MpiShmemHostPinnedSpace::deallocate( void * const arg_alloc_ptr , const size_t /* arg_alloc_size */ ) const
{
  try {
    MPISHMEM_SAFE_CALL( cudaFreeHost( arg_alloc_ptr ) );
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

SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::s_root_record ;

SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::s_root_record ;

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

std::string
SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::get_label() const
{
  return std::string( RecordBase::head()->m_label );
}

std::string
SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::get_label() const
{
  return std::string( RecordBase::head()->m_label );
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

SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void > *
SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::
allocate( const Kokkos::MpiShmemUVMSpace &  arg_space
        , const std::string          &  arg_label
        , const size_t                  arg_alloc_size
        )
{
  return new SharedAllocationRecord( arg_space , arg_label , arg_alloc_size );
}

SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void > *
SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::
allocate( const Kokkos::MpiShmemHostPinnedSpace &  arg_space
        , const std::string                 &  arg_label
        , const size_t                         arg_alloc_size
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

void
SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::
deallocate( SharedAllocationRecord< void , void > * arg_rec )
{
  delete static_cast<SharedAllocationRecord*>(arg_rec);
}

void
SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::
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

SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::
~SharedAllocationRecord()
{
  m_space.deallocate( SharedAllocationRecord< void , void >::m_alloc_ptr
                    , SharedAllocationRecord< void , void >::m_alloc_size
                    );
}

SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::
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

SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::
SharedAllocationRecord( const Kokkos::MpiShmemUVMSpace & arg_space
                      , const std::string          & arg_label
                      , const size_t                 arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      ( & SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::s_root_record
      , reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( sizeof(SharedAllocationHeader) + arg_alloc_size ) )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
  , m_tex_obj( 0 )
  , m_space( arg_space )
{
  // Fill in the Header information, directly accessible via UVM

  RecordBase::m_alloc_ptr->m_record = this ;

  strncpy( RecordBase::m_alloc_ptr->m_label
          , arg_label.c_str()
          , SharedAllocationHeader::maximum_label_length
          );
}

SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::
SharedAllocationRecord( const Kokkos::MpiShmemHostPinnedSpace & arg_space
                      , const std::string                 & arg_label
                      , const size_t                        arg_alloc_size
                      , const SharedAllocationRecord< void , void >::function_type arg_dealloc
                      )
  // Pass through allocated [ SharedAllocationHeader , user_memory ]
  // Pass through deallocation function
  : SharedAllocationRecord< void , void >
      ( & SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::s_root_record
      , reinterpret_cast<SharedAllocationHeader*>( arg_space.allocate( sizeof(SharedAllocationHeader) + arg_alloc_size ) )
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
  , m_space( arg_space )
{
  // Fill in the Header information, directly accessible via UVM

  RecordBase::m_alloc_ptr->m_record = this ;

  strncpy( RecordBase::m_alloc_ptr->m_label
          , arg_label.c_str()
          , SharedAllocationHeader::maximum_label_length
          );
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

void * SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::
allocate_tracked( const Kokkos::MpiShmemUVMSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  Kokkos::Impl::DeepCopy<MpiShmemUVMSpace,MpiShmemUVMSpace>( r_new->data() , r_old->data()
                                             , std::min( r_old->size() , r_new->size() ) );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

void * SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::
allocate_tracked( const Kokkos::MpiShmemHostPinnedSpace & arg_space
                , const std::string & arg_alloc_label
                , const size_t arg_alloc_size )
{
  if ( ! arg_alloc_size ) return (void *) 0 ;

  SharedAllocationRecord * const r =
    allocate( arg_space , arg_alloc_label , arg_alloc_size );

  RecordBase::increment( r );

  return r->data();
}

void SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::
deallocate_tracked( void * const arg_alloc_ptr )
{
  if ( arg_alloc_ptr != 0 ) {
    SharedAllocationRecord * const r = get_record( arg_alloc_ptr );

    RecordBase::decrement( r );
  }
}

void * SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::
reallocate_tracked( void * const arg_alloc_ptr
                  , const size_t arg_alloc_size )
{
  SharedAllocationRecord * const r_old = get_record( arg_alloc_ptr );
  SharedAllocationRecord * const r_new = allocate( r_old->m_space , r_old->get_label() , arg_alloc_size );

  Kokkos::Impl::DeepCopy<MpiShmemHostPinnedSpace,MpiShmemHostPinnedSpace>( r_new->data() , r_old->data()
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

SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void > *
SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::get_record( void * alloc_ptr )
{
  using Header     = SharedAllocationHeader ;
  using RecordMpiShmem = SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void > ;

  Header * const h = alloc_ptr ? reinterpret_cast< Header * >( alloc_ptr ) - 1 : (Header *) 0 ;

  if ( ! alloc_ptr || h->m_record->m_alloc_ptr != h ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::get_record ERROR" ) );
  }

  return static_cast< RecordMpiShmem * >( h->m_record );
}

SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void > *
SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::get_record( void * alloc_ptr )
{
  using Header     = SharedAllocationHeader ;
  using RecordMpiShmem = SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void > ;

  Header * const h = alloc_ptr ? reinterpret_cast< Header * >( alloc_ptr ) - 1 : (Header *) 0 ;

  if ( ! alloc_ptr || h->m_record->m_alloc_ptr != h ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::get_record ERROR" ) );
  }

  return static_cast< RecordMpiShmem * >( h->m_record );
}

// Iterate records to print orphaned memory ...
void
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
print_records( std::ostream & s , const Kokkos::MpiShmemSpace & space , bool detail )
{
  SharedAllocationRecord< void , void > * r = & s_root_record ;

  char buffer[256] ;

  SharedAllocationHeader head ;

  if ( detail ) {
    do {
      if ( r->m_alloc_ptr ) {
        Kokkos::Impl::DeepCopy<HostSpace,MpiShmemSpace>::DeepCopy( & head , r->m_alloc_ptr , sizeof(SharedAllocationHeader) );
      }
      else {
        head.m_label[0] = 0 ;
      }

      //Formatting dependent on sizeof(uintptr_t)
      const char * format_string;

      if (sizeof(uintptr_t) == sizeof(unsigned long)) { 
        format_string = "MpiShmem addr( 0x%.12lx ) list( 0x%.12lx 0x%.12lx ) extent[ 0x%.12lx + %.8ld ] count(%d) dealloc(0x%.12lx) %s\n";
      }
      else if (sizeof(uintptr_t) == sizeof(unsigned long long)) { 
        format_string = "MpiShmem addr( 0x%.12llx ) list( 0x%.12llx 0x%.12llx ) extent[ 0x%.12llx + %.8ld ] count(%d) dealloc(0x%.12llx) %s\n";
      }

      snprintf( buffer , 256 
              , format_string
              , reinterpret_cast<uintptr_t>( r )
              , reinterpret_cast<uintptr_t>( r->m_prev )
              , reinterpret_cast<uintptr_t>( r->m_next )
              , reinterpret_cast<uintptr_t>( r->m_alloc_ptr )
              , r->m_alloc_size
              , r->m_count
              , reinterpret_cast<uintptr_t>( r->m_dealloc )
              , head.m_label
              );
      std::cout << buffer ;
      r = r->m_next ;
    } while ( r != & s_root_record );
  }
  else {
    do {
      if ( r->m_alloc_ptr ) {

        Kokkos::Impl::DeepCopy<HostSpace,MpiShmemSpace>::DeepCopy( & head , r->m_alloc_ptr , sizeof(SharedAllocationHeader) );

        //Formatting dependent on sizeof(uintptr_t)
        const char * format_string;

        if (sizeof(uintptr_t) == sizeof(unsigned long)) { 
          format_string = "MpiShmem [ 0x%.12lx + %ld ] %s\n";
        }
        else if (sizeof(uintptr_t) == sizeof(unsigned long long)) { 
          format_string = "MpiShmem [ 0x%.12llx + %ld ] %s\n";
        }

        snprintf( buffer , 256 
                , format_string
                , reinterpret_cast< uintptr_t >( r->data() )
                , r->size()
                , head.m_label
                );
      }
      else {
        snprintf( buffer , 256 , "MpiShmem [ 0 + 0 ]\n" );
      }
      std::cout << buffer ;
      r = r->m_next ;
    } while ( r != & s_root_record );
  }
}

void
SharedAllocationRecord< Kokkos::MpiShmemUVMSpace , void >::
print_records( std::ostream & s , const Kokkos::MpiShmemUVMSpace & space , bool detail )
{
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "MpiShmemUVM" , & s_root_record , detail );
}

void
SharedAllocationRecord< Kokkos::MpiShmemHostPinnedSpace , void >::
print_records( std::ostream & s , const Kokkos::MpiShmemHostPinnedSpace & space , bool detail )
{
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "MpiShmemHostPinned" , & s_root_record , detail );
}

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace {
  __global__ void init_lock_array_kernel() {
    unsigned i = blockIdx.x*blockDim.x + threadIdx.x;

    if(i<MPISHMEM_SPACE_ATOMIC_MASK+1)
      kokkos_impl_cuda_atomic_lock_array[i] = 0;
  }
}

namespace Impl {
int* lock_array_cuda_space_ptr(bool deallocate) {
  static int* ptr = NULL;
  if(deallocate) {
    cudaFree(ptr);
    ptr = NULL;
  }

  if(ptr==NULL && !deallocate)
    cudaMalloc(&ptr,sizeof(int)*(MPISHMEM_SPACE_ATOMIC_MASK+1));
  return ptr;
}

void init_lock_array_cuda_space() {
  int is_initialized = 0;
  if(! is_initialized) {
    int* lock_array_ptr = lock_array_cuda_space_ptr();
    cudaMemcpyToSymbol( kokkos_impl_cuda_atomic_lock_array , & lock_array_ptr , sizeof(int*) );
    init_lock_array_kernel<<<(MPISHMEM_SPACE_ATOMIC_MASK+255)/256,256>>>();
  }
}

}
}
#endif // KOKKOS_HAVE_MPISHMEM
