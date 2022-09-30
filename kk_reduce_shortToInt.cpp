#include "Kokkos_Core.hpp"
#include "Kokkos_StdAlgorithms.hpp"
#include <utility> // std::make_pair

namespace KK = Kokkos;
using Exec = KK::DefaultExecutionSpace;

typedef std::int8_t I8;
typedef std::int32_t LO;

template< typename T>
struct i32Plus {
  KOKKOS_INLINE_FUNCTION
    LO operator()(const T & a, const T & b) const {
      LO al(a);
      LO bl(b);
      LO res = al + bl;
      //printf("res al bl %d %d %d\n", res, al, bl);
      return res;
    }
};

int main(int argc, char** argv) {
  KK::initialize(argc, argv);
  const int size = 5;

  KK::View<I8*> in("i8View",size);
  KK::parallel_for("ones", size,
    KOKKOS_LAMBDA (const int& i) { in[i] = 1; }
  );
  KK::View<LO*> out("loView",size+1);
  int sz = out.size();
  auto outSub = KK::subview(out, std::make_pair(1,sz));

  assert(outSub.size()==in.size());
  auto kkOp = i32Plus<I8>();
  KK::Experimental::inclusive_scan(Exec(), in, outSub, kkOp);
  KK::fence();
  KK::parallel_for("bar", size+1,
    KOKKOS_LAMBDA (const int& i) { printf("%d %d\n", i, out[i]); }
  );
  KK::finalize();
  return 0;
}
