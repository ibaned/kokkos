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
class MpiShmemInternal {
private:
  MpiShmemInternal( const MpiShmemInternal & );
  MpiShmemInternal & operator = ( const MpiShmemInternal & );
public:

  typedef MpiShmem::size_type size_type;

  MPI_Comm m_world;
  MPI_Comm m_team;
  bool     m_called_mpi_init;

  static MpiShmemInternal & singleton();

  int verify_is_initialized( const char * const label ) const;

  int is_initialized() const
  {
    return m_world != MPI_COMM_NULL && m_team != MPI_COMM_NULL;
  }

  void initialize( MPI_Comm world, MPI_Comm team );
  void finalize();

  void print_configuration( std::ostream & ) const ;

  void fence() const;

  ~MpiShmemInternal();

  MpiShmemInternal()
    : m_world( MPI_COMM_NULL )
    , m_team( MPI_COMM_NULL )
    , m_called_mpi_init( false )
  {}
};

MpiShmemInternal & MpiShmemInternal::singleton()
{
  static MpiShmemInternal self;
  return self;
}

void MpiShmemInternal::initialize( MPI_Comm world, MPI_Comm team )
{
  int is_mpi_initialized;
  MPI_Initialized(&is_mpi_initialized);
  if (!is_mpi_initialized) {
    MPI_Init(NULL,NULL);
    m_called_mpi_init = true;
  }
  m_world = world;
  if (team == MPI_COMM_NULL)
    MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &team);
  m_team = team;
}

void MpiShmemInternal::finalize()
{
  if (m_team != MPI_COMM_NULL)
    MPI_Comm_free(&m_team);
  if (m_called_mpi_init)
    MPI_Finalize();
}

void MpiShmemInternal::print_configuration( std::ostream & ) const
{
}

void MpiShmemInternal::fence() const
{
  Kokkos::memory_fence();
}

MpiShmemInternal::~MpiShmemInternal()
{
}

}// namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

int MpiShmem::concurrency()
{
  return MpiShmem::team_size();
}

int MpiShmem::is_initialized()
{
  return Impl::MpiShmemInternal::singleton().is_initialized();
}

static void MpiShmem::initialize( MPI_Comm world_comm
                                , MPI_Comm team_comm );
{
  Impl::MpiShmemInternal::singleton().initialize(
      world_comm, team_comm );
}

void MpiShmem::finalize()
{
  Impl::MpiShmemInternal::singleton().finalize();
}

MpiShmem::MpiShmem()
{
}

MpiShmem::~MpiShmem()
{
}

void MpiShmem::print_configuration( std::ostream & s , const bool )
{
  Impl::MpiShmemInternal::singleton().print_configuration( s );
}

bool MpiShmem::sleep()
{
  return false;
}

bool MpiShmem::wake()
{
  return true;
}

void MpiShmem::fence()
{
  Impl::MpiShmemInternal::singleton().fence();
}

MPI_Comm MpiShmem::team_comm()
{
  return MpiShmemInternal::instance().m_team;
}

MPI_Comm MpiShmem::world_comm()
{
  return MpiShmemInternal::instance().m_team;
}

  /// \brief Get which team in the world this proc belongs to
int MpiShmem::which_team()
{
  return world_rank() / team_size();
}

int MpiShmem::team_rank()
{
  int rank;
  MPI_Comm_rank( team_comm(), &rank );
  return rank;
}

int MpiShmem::team_size()
{
  int size;
  MPI_Comm_rank( team_comm(), &size );
  return size;
}

int MpiShmem::world_rank()
{
  int size;
  MPI_Comm_rank( world_comm(), &size );
  return size;
}

int MpiShmem::world_size()
{
  int size;
  MPI_Comm_rank( world_comm(), &size );
  return size;
}

} // namespace Kokkos

#endif // KOKKOS_HAVE_MPISHMEM
//----------------------------------------------------------------------------


