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

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>

#define USE_MEMORY_POOL_V2

using ExecSpace   = Kokkos::DefaultExecutionSpace ;
using MemorySpace = Kokkos::DefaultExecutionSpace::memory_space ;

using MemoryPool =
#if defined( USE_MEMORY_POOL_V2 )
Kokkos::Experimental::MemoryPoolv2< ExecSpace > ;
#else
Kokkos::Experimental::MemoryPool< ExecSpace > ;
#endif

static void print_debug_code(uint64_t code) {
  if (!(code & DEBUG_BLOCK_FITS_ANY_SUPERBLOCK)) {
    std::cerr << "DIDNT FIT????\n";
  }
  if (code & DEBUG_PREFETCHING_LOOP) {
    std::cerr << "looped back around while(1) for prefetch\n";
  }
  if (code & DEBUG_PREEMPTED_PREFETCHING) {
    std::cerr << "was preempted while prefetching\n";
  }
  if (code & DEBUG_PARTFULL_MEANS_100) {
    std::cerr << "searched for partfull for allocation, so looked for < 100% full\n";
  }
  if (code & DEBUG_PARTFULL_MEANS_94) {
    std::cerr << "searched for partfull for prefetch, so looked for < 94% full\n";
  }
  if (code & DEBUG_HINT_FOUND_LOCKED_PARTFULL) {
    std::cerr << "hint found locked when searching for partfull\n";
  }
  if (code & DEBUG_ITERATED_PARTFULL_SEARCH) {
    std::cerr << "searched more than one superblock for partfull\n";
  }
  if (code & DEBUG_WRAPPED_PARTFULL_SEARCH) {
    std::cerr << "wrapped around the superblocks for partfull\n";
  }
  if (code & DEBUG_NO_FOUND_PARTFULL) {
    std::cerr << "did not find a partfull superblock\n";
  }
  if (code & DEBUG_FOUND_PARTFULL_IN_SEARCH) {
    std::cerr << "found a partfull superblock during search\n";
  }
  if (code & DEBUG_FOUND_PARTFULL_ID) {
    std::cerr << "found a partfull superblock id\n";
  }
  if (code & DEBUG_TRY_UPDATE_HINT) {
    std::cerr << "tried to update the hint via CAS\n";
  }
  if (code & DEBUG_LOCKED_HINT) {
    std::cerr << "locked the hint\n";
  }
  if (code & DEBUG_ITERATED_EMPTY_SEARCH) {
    std::cerr << "searched more than one superblock for empty\n";
  }
  if (code & DEBUG_WRAPPED_EMPTY_SEARCH) {
    std::cerr << "wrapped around the superblocks for empty\n";
  }
  if (code & DEBUG_CLAIMED_EMPTY) {
    std::cerr << "claimed an empty superblock\n";
  }
  if (code & DEBUG_UNLOCKED_HINT) {
    std::cerr << "unlocked the hint\n";
  }
  if (code & DEBUG_FOUND_EMPTY) {
    std::cerr << "found an empty superblock\n";
  }
  if (code & DEBUG_HINT_FOUND_LOCKED_EMPTY_ALLOCATING) {
    std::cerr << "hint found locked when searching for empty superblock for own allocation\n";
  }
  if (code & DEBUG_HINT_FOUND_LOCKED_EMPTY_PREFETCH) {
    std::cerr << "hint found locked when searching for empty superblock for prefetch\n";
  }
  if (code & DEBUG_PREFETCH_SUCCESSFUL) {
    std::cerr << "successfully prefetched a block\n";
  }
  if (code & DEBUG_FOUND_SUPERBLOCK) {
    std::cerr << "found superblock to allocate into\n";
  }
  if (code & DEBUG_TRIED_ACQUIRING_BIT) {
    std::cerr << "tried to acquire a bitset bit\n";
  }
  if (code & DEBUG_ACQUIRED_BIT) {
    std::cerr << "acquired a bitset bit\n";
  }
  if (code & DEBUG_ASSIGNED_PREFETCH) {
    std::cerr << "assigned prefetch duty after acquiring a bit\n";
  }
  if (code & DEBUG_LOSER) {
    std::cerr << "is a loser. either lost a race for a bit or assigned to prefetch.\n";
  }
  if (code & DEBUG_FAILED) {
    std::cerr << "allocate() failed !\n";
  }
}

struct TestFunctor {

  typedef Kokkos::View< uintptr_t * , ExecSpace >  ptrs_type ;
  typedef Kokkos::View< uint64_t * , ExecSpace >  debugs_type ;

  enum : unsigned { chunk = 64 };

  MemoryPool  pool ;
  ptrs_type   ptrs ;
  debugs_type   debugs ;
  unsigned    stride_chunk ;
  unsigned    fill_stride ;
  unsigned    range_iter ;
  unsigned    repeat ;

  TestFunctor( size_t    total_alloc_size
             , unsigned  min_superblock_size
             , unsigned  number_alloc
             , unsigned  arg_stride_alloc
             , unsigned  arg_stride_chunk
             , unsigned  arg_repeat )
    : pool()
    , ptrs()
    , debugs()
    , stride_chunk(0)
    , fill_stride(0)
    , repeat(0)
    {
      MemorySpace m ;
#if defined( USE_MEMORY_POOL_V2 )
      pool = MemoryPool( m , total_alloc_size , min_superblock_size );
#else
      pool = MemoryPool( m , total_alloc_size , Kokkos::Impl::integral_power_of_two_that_contains( min_superblock_size ) );
#endif
      ptrs = ptrs_type( Kokkos::view_alloc( m , "ptrs") , number_alloc );
      debugs = debugs_type( Kokkos::view_alloc( m , "debugs") , number_alloc );
      fill_stride = arg_stride_alloc ;
      stride_chunk = arg_stride_chunk ;
      range_iter   = fill_stride * number_alloc ;
      repeat       = arg_repeat ;
    }

  void print_on_fail() {
    pool.print_state(std::cerr);
    typename decltype(pool)::usage_statistics stats;
    pool.get_usage_statistics(stats);
    std::cerr << "capacity bytes " << stats.capacity_bytes << '\n';
    std::cerr << "superblock bytes " << stats.superblock_bytes << '\n';
    std::cerr << "capacity superblocks " << stats.capacity_superblocks << '\n';
    std::cerr << "consumed superblocks " << stats.consumed_superblocks << '\n';
    std::cerr << "consumed blocks " << stats.consumed_blocks << '\n';
    std::cerr << "consumed bytes " << stats.consumed_bytes << '\n';
    std::cerr << "reserved blocks " << stats.reserved_blocks << '\n';
    std::cerr << "reserved bytes " << stats.reserved_bytes << '\n';
  }

  //----------------------------------------

  typedef long value_type ;

  //----------------------------------------

  struct TagFill {};

  KOKKOS_INLINE_FUNCTION
  void operator()( TagFill , int i , value_type & update ) const noexcept
    {
      if ( 0 == i % fill_stride ) {

        const int j = i / fill_stride ;

        const unsigned size_alloc = chunk * ( 1 + ( j % stride_chunk ) );

        uint64_t debug;
        ptrs(j) = (uintptr_t) pool.allocate(size_alloc, 0, &debug);
        debugs(j) = debug;

        if ( ptrs(j) ) ++update ;
        else printf("i = %d failed, update = %ld\n", i, update);
      }
    }

  bool test_fill()
    {
      typedef Kokkos::RangePolicy< ExecSpace , TagFill > policy ;

      long result = 0 ;

      Kokkos::parallel_reduce( policy(0,range_iter), *this , result );

      // if all is well just return
      if (result == ptrs.extent(0)) return true;

      // otherwise print tons of debug information
      std::cerr << "# successful results " << result << " / " << ptrs.extent(0) << '\n';
      print_on_fail();
      auto h_debugs = Kokkos::create_mirror_view(debugs);
      Kokkos::deep_copy(h_debugs, debugs);
      std::map<uint64_t, size_t> unique_codes_and_reps;
      for (size_t i = 0; i < h_debugs.size(); ++i) {
        auto code = h_debugs(i);
        if (!unique_codes_and_reps.count(code)) {
          unique_codes_and_reps[code] = i;
        }
      }
      for (auto pair : unique_codes_and_reps) {
        auto code = pair.first;
        auto i = pair.second;
        std::cerr << "i = " << i << " had these debug traits: "
          << std::hex << code << std::dec << '\n';
        print_debug_code(code);
      }
      std::cerr << "END DEBUG TRAITS\n";

      return false;
    }

  //----------------------------------------

  struct TagDel {};

  KOKKOS_INLINE_FUNCTION
  void operator()( TagDel , int i ) const noexcept
    {
      if ( 0 == i % fill_stride ) {

        const int j = i / fill_stride ;

        const unsigned size_alloc = chunk * ( 1 + ( j % stride_chunk ) );

        pool.deallocate( (void*) ptrs(j) , size_alloc );
      }
    }

  void test_del()
    {
      typedef Kokkos::RangePolicy< ExecSpace , TagDel > policy ;

      Kokkos::parallel_for( policy(0,range_iter), *this );
    }

  //----------------------------------------

  struct TagAllocDealloc {};

  KOKKOS_INLINE_FUNCTION
  void operator()( TagAllocDealloc , int i , long & update ) const noexcept
    {
      if ( 0 == i % fill_stride ) {

        const int j = i / fill_stride ;

        if ( 0 == j % 3 ) {

          for ( int k = 0 ; k < repeat ; ++k ) {

            const unsigned size_alloc = chunk * ( 1 + ( j % stride_chunk ) );

            pool.deallocate( (void*) ptrs(j) , size_alloc );
        
            ptrs(j) = (uintptr_t) pool.allocate(size_alloc);

            if ( 0 == ptrs(j) ) update++ ;
          }
        }
      }
    }

  bool test_alloc_dealloc()
    {
      typedef Kokkos::RangePolicy< ExecSpace , TagAllocDealloc > policy ;

      long error_count = 0 ;

      Kokkos::parallel_reduce( policy(0,range_iter), *this , error_count );

      if (0 == error_count) return true;
      print_on_fail();
      return false;
    }
};



int main( int argc , char* argv[] )
{
  static const char help_flag[] = "--help" ;
  static const char alloc_size_flag[]   = "--alloc_size=" ;
  static const char super_size_flag[]   = "--super_size=" ;
  static const char chunk_span_flag[]   = "--chunk_span=" ;
  static const char fill_stride_flag[]  = "--fill_stride=" ;
  static const char fill_level_flag[]   = "--fill_level=" ;
  static const char repeat_outer_flag[] = "--repeat_outer=" ;
  static const char repeat_inner_flag[] = "--repeat_inner=" ;

  long total_alloc_size    = 1000000 ;
  int  min_superblock_size =   10000 ;
  int  chunk_span          =       5 ;
  int  fill_stride        =       1 ;
  int  fill_level         =      70 ;
  int  repeat_outer   =       1 ;
  int  repeat_inner   =       1 ;

  int  ask_help = 0 ;

  for(int i=1;i<argc;i++)
  {
     const char * const a = argv[i];

     if ( ! strncmp(a,help_flag,strlen(help_flag) ) ) ask_help = 1 ;

     if ( ! strncmp(a,alloc_size_flag,strlen(alloc_size_flag) ) )
       total_alloc_size = atol( a + strlen(alloc_size_flag) );

     if ( ! strncmp(a,super_size_flag,strlen(super_size_flag) ) )
       min_superblock_size = atoi( a + strlen(super_size_flag) );

     if ( ! strncmp(a,fill_stride_flag,strlen(fill_stride_flag) ) )
       fill_stride = atoi( a + strlen(fill_stride_flag) );

     if ( ! strncmp(a,fill_level_flag,strlen(fill_level_flag) ) )
       fill_level = atoi( a + strlen(fill_level_flag) );

     if ( ! strncmp(a,chunk_span_flag,strlen(chunk_span_flag) ) )
       chunk_span = atoi( a + strlen(chunk_span_flag) );

     if ( ! strncmp(a,repeat_outer_flag,strlen(repeat_outer_flag) ) )
       repeat_outer = atoi( a + strlen(repeat_outer_flag) );

     if ( ! strncmp(a,repeat_inner_flag,strlen(repeat_inner_flag) ) )
       repeat_inner = atoi( a + strlen(repeat_inner_flag) );
  }

  const int mean_chunk   = TestFunctor::chunk * ( 1 + ( chunk_span / 2 ) );
  const int number_alloc = double(total_alloc_size) * double(fill_level) /
                           ( double(mean_chunk) * double(100) );

  double time = 0 ;

  int error = 0 ;

  if ( ask_help ) {
    std::cout << "command line options:"
              << " " << help_flag
              << " " << alloc_size_flag << "##"
              << " " << super_size_flag << "##"
              << " " << fill_stride_flag << "##"
              << " " << fill_level_flag << "##"
              << " " << chunk_span_flag << "##"
              << " " << repeat_outer_flag << "##"
              << " " << repeat_inner_flag << "##"
              << std::endl ;
  }
  else {

    Kokkos::initialize(argc,argv);

    std::cerr << "number_alloc " << number_alloc << '\n';
    TestFunctor functor( total_alloc_size
                       , min_superblock_size
                       , number_alloc
                       , fill_stride
                       , chunk_span
                       , repeat_inner );

    if ( ! functor.test_fill() ) {
      Kokkos::abort("  fill failed");
    }

    Kokkos::Impl::Timer timer ;

    for ( int i = 0 ; i < repeat_outer ; ++i ) {
      error |= ! functor.test_alloc_dealloc();
    }

    time = timer.seconds();

    Kokkos::finalize();
  }

  printf( "\"mempool: alloc super stride level span inner outer number time\" %ld %d %d %d %d %d %d %d %f\n"
        , total_alloc_size
        , min_superblock_size
        , fill_stride
        , fill_level
        , chunk_span
        , repeat_inner
        , repeat_outer
        , number_alloc
        , time );

  if ( error ) { fprintf(stderr,"  TEST FAILED\n"); }

  return 0 ;
}

