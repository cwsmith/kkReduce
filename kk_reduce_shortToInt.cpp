#include "Kokkos_Core.hpp"
#include "Kokkos_StdAlgorithms.hpp"
#include <utility> // std::make_pair

namespace KK = Kokkos;
namespace KE = Kokkos::Experimental;
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
  int hasFailed;
  {
  const int size = 5;

  //input array of short ints
  KK::View<I8*> in("i8View",size);
  KE::fill(Exec(), in, I8(1)); //fill with 1s

  //input array of ints
  KK::View<LO*> out("loView",size+1);

  //write exclusive prefix sum into outSub
  auto outSub = KK::subview(out, std::make_pair(1,size+1));
  auto kkOp = i32Plus<I8>();
  KE::inclusive_scan(Exec(), in, outSub, kkOp);

  //check the result
  KK::View<LO*> matchCount("matchCount",1);
  KK::View<LO*,KK::HostSpace> matchCount_h("matchCount_h",1);
  KK::parallel_for("check", size+1,
    KOKKOS_LAMBDA (const int& i) {
      if(out[i] != i) {
        printf("out(%d) = %d should be %d\n", i, out(i), i);
      } else {
        KK::atomic_increment(&matchCount(0));
      }
    }
  );
  KK::deep_copy(matchCount_h,matchCount);
  KK::fence();
  hasFailed = !(matchCount_h(0) == size+1);
  }
  KK::finalize();
  return hasFailed;;
}
