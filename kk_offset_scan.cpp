#include<Kokkos_Core.hpp>
#include<cstdio>

void test_scan(int N, Kokkos::View<int*>& src, Kokkos::View<int*>& out) {
  assert(src.size() >= N+1);
  assert(out.size() >= N+1);
  std::stringstream ss;
  ss << "offsetScan_" << N;
  std::string name = ss.str();

  Kokkos::parallel_scan(name, N+1,
      KOKKOS_LAMBDA(int i, int& partial_sum, bool is_final) {
      if(is_final) out(i) = partial_sum;
      partial_sum += src(i);
      });
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
  auto size = std::size_t(1) << 21;
  //allocate the input and output arrays once to mimic
  //the omega_h memory pool
  Kokkos::View<int*> src("source",size);
  Kokkos::parallel_for("ones", size, KOKKOS_LAMBDA(int i) { src(i)=1; });
  Kokkos::View<int*> out("source",size);
  Kokkos::Timer timer;
  auto timeStart = timer.seconds();
  for(int i=0; i<16; i++) {
    size = size >> 1;
    printf("%d size %lu\n", i, size);
    for(int j=0; j<100; j++) {
      test_scan(size,src,out);
    }
  }
  printf("total time(s) %f\n", timer.seconds()-timeStart);
}
  Kokkos::finalize();
  return 0;
}
