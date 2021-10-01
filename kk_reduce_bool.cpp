#include "Kokkos_Core.hpp"

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  const int size = 5;

  bool res;
  Kokkos::parallel_reduce(
    Kokkos::RangePolicy<>(0, size), 
    KOKKOS_LAMBDA(const int& i, bool& update) {
      update = true;
  }, Kokkos::LAnd<bool>(res) );
  assert(res==true);
  return 0;
}
