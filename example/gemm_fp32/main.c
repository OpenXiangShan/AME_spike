/*
 * Copyright (c) 2026 BOSC & ICT, CAS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define ZTT_LLM_CASE ZTT_LLM_GEMM_FP32
#if defined(ZTT_EXTERNAL_MATRIX_INPUT)
#define ZTT_LLM_CASE_NAME ZTT_MATRIX_CASE_NAME
#else
#define ZTT_LLM_CASE_NAME "FP32 GEMM C += A x B (N=128)"
#endif
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
    if (!ztt_llm_write_report(
            argv[1], result.output, result.element_count,
            result.element_bytes, result.status, ZTT_LLM_CASE,
            ZTT_LLM_CASE_NAME)) {
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
