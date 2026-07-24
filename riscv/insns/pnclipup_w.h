require_extension('P');
require_rv64;
uint64_t s1 = RS1;
uint64_t s2 = RS2;

reg_t sat_w0 = P_USAT_FULL(32, s1);
reg_t sat_w1 = P_USAT_FULL(32, s2);

WRITE_RD(((uint64_t)(uint32_t)sat_w1 << 32) | (uint32_t)sat_w0);

