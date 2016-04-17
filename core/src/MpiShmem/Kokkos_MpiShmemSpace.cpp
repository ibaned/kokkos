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
namespace Experimental {

MpiShmemSpace::MpiShmemSpace()
{
}

void * MpiShmemSpace::allocate( const size_t total_data_size,
                              , const size_t header_size
                              , MPI_Win* win ) const
{
  void * ptr = NULL;
  size_t team_size = static_cast<size_t>(MpiShmem::team_size());
  size_t team_rank = static_cast<size_t>(MpiShmem::team_size());
  size_t quot = total_data_size / team_size;
  size_t rem = total_data_size % team_size;
  size_t local_size = quot;
  if (team_rank < rem)
    local_size++;
  if (team_rank == 0)
    local_size += header_size;
  MPI_Win_allocate_shared(local_size
                         ,sizeof(size_type)
                         ,MPI_INFO_NULL
                         ,MpiShmem::team_comm()
                         ,&ptr
                         ,win);
  return ptr ;
}

void MpiShmemSpace::deallocate( void * const arg_alloc_ptr , const size_t /* arg_alloc_size */ ) const
{
  MPI_Free_mem(arg_alloc_ptr);
}

} // namespace Experimental
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Experimental {
namespace Impl {

SharedAllocationRecord< void , void >
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::s_root_record ;

SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
SharedAllocationRecord( const Kokkos::MpiShmemSpace & arg_space
                      , const std::string       & arg_label
                      , const size_t              arg_alloc_size
                      , const RecordBase::function_type arg_dealloc
                      )
  : SharedAllocationRecord< void , void >
      ( & SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::s_root_record
  // The need to come up with the allocation pointer immediately when
  // calling the base class constructor really makes this awkward.
  // MPI needs to create the allocation and Window simultaneously,
  // so notice the &m_win at the end here...
  // having initialize methods for SharedAllocationRecord<void,void> would help.
      , reinterpret_cast<SharedAllocationHeader*>(
            arg_space.allocate(arg_alloc_size
                              ,sizeof(SharedAllocationHeader)
                              ,&m_win))
      , sizeof(SharedAllocationHeader) + arg_alloc_size
      , arg_dealloc
      )
  , m_space( arg_space )
{
  // Fill in the Header information
  RecordBase::m_alloc_ptr->m_record =
      static_cast< SharedAllocationRecord< void , void > * >( this );
  strncpy( RecordBase::m_alloc_ptr->m_label
          , arg_label.c_str()
          , SharedAllocationHeader::maximum_label_length
          );
}

// TODO: much of the following is copy-pasted from
// HostSpace, since the only real difference is how
// we (de)allocate.
// move this code to some shared place.

std::string
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::get_label() const
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

  Kokkos::Impl::DeepCopy<MpiShmemSpace,MpiShmemSpace>(
       r_new->data() , r_old->data()
     , std::min( r_old->size() , r_new->size() ) );

  RecordBase::increment( r_new );
  RecordBase::decrement( r_old );

  return r_new->data();
}

//----------------------------------------------------------------------------

SharedAllocationRecord< Kokkos::MpiShmemSpace , void > *
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::get_record( void * alloc_ptr )
{
  typedef SharedAllocationHeader  Header ;
  typedef SharedAllocationRecord< Kokkos::HostSpace , void >  RecordHost ;

  SharedAllocationHeader const * const head   = alloc_ptr ? Header::get_header( alloc_ptr ) : (SharedAllocationHeader *)0 ;
  RecordHost                   * const record = head ? static_cast< RecordHost * >( head->m_record ) : (RecordHost *) 0 ;

  if ( ! alloc_ptr || record->m_alloc_ptr != head ) {
    Kokkos::Impl::throw_runtime_exception( std::string("Kokkos::Experimental::Impl::SharedAllocationRecord< Kokkos::HostSpace , void >::get_record ERROR" ) );
  }

  return record ;
}

void
SharedAllocationRecord< Kokkos::MpiShmemSpace , void >::
print_records( std::ostream & s , const Kokkos::MpiShmemSpace & space , bool detail )
{
  SharedAllocationRecord< void , void >::print_host_accessible_records( s , "MpiShmem" , & s_root_record , detail );
}

} // namespace Impl
} // namespace Experimental
} // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

#endif // KOKKOS_HAVE_MPISHMEM
