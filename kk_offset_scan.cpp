#include<Kokkos_Core.hpp>
#include<cstdio>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
    int N = argc>1?atoi(argv[1]):100;
    Kokkos::View<int*>post("postfix_sum",N);
    Kokkos::View<int*>pre("prefix_sum",N+1);
    Kokkos::View<int*>src("source",N+1);

    Kokkos::View<int*, Kokkos::HostSpace> host("host", N+1);
    for(int i=0; i<host.size(); i++) host(i) = 2;
    Kokkos::deep_copy(src, host);

    int result = 0;
    Kokkos::parallel_scan("Loop1", N+1,
      KOKKOS_LAMBDA(int i, int& partial_sum, bool is_final) {
      if(is_final) pre(i) = partial_sum;
      partial_sum += src(i);
    }, result);

    Kokkos::deep_copy(host, pre);
    printf("pre\n"); 
    for(int i=0; i<host.size(); i++) printf("%d ", host(i));
    printf("\n"); 
  }
  Kokkos::finalize();
}
