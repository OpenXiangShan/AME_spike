require_extension('P');
require_rv32;
sreg_t sshamt = P_FIELD(RS2, 0, 8);
sreg_t val;
if (RS1 == 0) {
  val = 0;
} else if (sshamt >= 32) {
  val = (RS1 & 0x80000000) ? 0x80000000 : 0x7fffffff;
  P.set_vxsat();
} else if (sshamt <= -32) {
  val = 0;
} else {
  val = sshamt >= 0 ? P_SAT(32, static_cast<sreg_t> (RS1) << sshamt) : ((RS1 >> -sshamt) + ((RS1 >> (-sshamt - 1)) & 1));
}

WRITE_RD(sext32(val));
