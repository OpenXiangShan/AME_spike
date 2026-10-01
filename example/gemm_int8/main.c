// Copyright (c) 2026 BOSC & ICT, CAS
// All rights reserved.
// See LICENSE for license details.
#define ZTT_LLM_CASE ZTT_LLM_GEMM_INT8
#define ZTT_LLM_CASE_NAME "INT8 GEMM with INT32/INT64 conversion (N=128)"
#include "ztt_llm_case.h"

int
main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s REPORT_PATH\n", argv[0]);
        return 2;
    }
    struct ztt_llm_case_result result;
    result.status = ztt_llm_run_case(&result);
    if (!ztt_llm_write_report(argv[1], result.output, result.element_count,
                              result.element_bytes, result.status,
                              ZTT_LLM_CASE, ZTT_LLM_CASE_NAME)) {
        fprintf(stderr, "cannot write report: %s\n", argv[1]);
        return 1;
    }
    if (result.status != 0) {
        fprintf(stderr, "%s amestatus=0x%llx\n", ZTT_LLM_CASE_NAME,
                (unsigned long long)result.status);
        return 1;
    }
    return 0;
}
