require_extension('P');
require_rv64;
int64_t s1 = (int64_t)RS1;
int64_t s2 = (int64_t)RS2;

sreg_t sat_w0 = P_SAT(32, s1);
sreg_t sat_w1 = P_SAT(32, s2);

WRITE_RD(((uint64_t)(uint32_t)sat_w1 << 32) | (uint32_t)sat_w0);

