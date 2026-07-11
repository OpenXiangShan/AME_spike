// vqdotu.vv vd, vs2, vs1, vm
#include "vdot4a_common.h"

require_extension(EXT_ZVDOT4A);
require(P.VU.vsew == e32);

VI_VV_LOOP
({
  VQDOT(vs1, vs2, uint8_t, uint8_t);
  vd = (vd + result) & 0xffffffff;
})
