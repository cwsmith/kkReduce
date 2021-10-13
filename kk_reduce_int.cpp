#include "Kokkos_Core.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  {
  const int size = 5;

  int res;
  Kokkos::parallel_reduce(
    Kokkos::RangePolicy<>(0, size), 
    KOKKOS_LAMBDA(const int& i, int& update) {
      update = 1;
    }
  , res);
  std::cerr << "res " << res << "\n";
  assert(res==size);
  }
  Kokkos::finalize();
  return 0;
}
