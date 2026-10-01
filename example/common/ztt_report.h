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

#ifndef ZTT_REPORT_H
#define ZTT_REPORT_H

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ZTT_GEMM_MAX_CASES 24
#define ZTT_GEMM_MAX_VALUES 576

#define ZTT_GEMM_REPORT_MAGIC UINT64_C(0x5a545447454d4d31)
#define ZTT_GEMM_REPORT_VERSION UINT64_C(1)

/* Raw element bit patterns make reports independent of host floating ABI. */
struct ztt_gemm_report {
    uint64_t magic;
    uint64_t version;
    uint64_t suite;
    uint64_t case_count;
    uint64_t values_per_case;
    uint64_t case_values[ZTT_GEMM_MAX_CASES];
    uint64_t values[ZTT_GEMM_MAX_CASES][ZTT_GEMM_MAX_VALUES];
};

static inline void
ztt_gemm_report_init(struct ztt_gemm_report *report, uint64_t suite,
                     uint64_t case_count)
{
    memset(report, 0, sizeof(*report));
    report->magic = ZTT_GEMM_REPORT_MAGIC;
    report->version = ZTT_GEMM_REPORT_VERSION;
    report->suite = suite;
    report->case_count = case_count;
    report->values_per_case = ZTT_GEMM_MAX_VALUES;
}

static inline int
ztt_gemm_report_write(const char *path, const struct ztt_gemm_report *report,
                      const char *const case_names[ZTT_GEMM_MAX_CASES])
{
    FILE *file = fopen(path, "w");
    if (!file)
        return 0;
    int ok = fprintf(file,
                     "Ztt GEMM result report (UTF-8)\n"
                     "format_version: %" PRIu64 "\n"
                     "suite: %" PRIu64 "\n"
                     "case_count: %" PRIu64 "\n\n",
                     report->version, report->suite, report->case_count) > 0;
    for (uint64_t test = 0; ok && test < report->case_count; ++test) {
        const char *name = case_names[test] ? case_names[test] : "unnamed";
        ok = fprintf(file, "case[%" PRIu64 "]: %s\n", test, name) > 0;
        for (uint64_t value = 0; ok && value < report->case_values[test];
             ++value)
            ok = fprintf(file, "  [%03" PRIu64 "] 0x%016" PRIx64 "\n",
                         value, report->values[test][value]) > 0;
        ok = ok && fputc('\n', file) != EOF;
    }
    return ok && fclose(file) == 0;
}

static inline void
ztt_gemm_report_bits(struct ztt_gemm_report *report, unsigned test,
                     const void *values, unsigned count,
                     unsigned element_bytes)
{
    if (test >= ZTT_GEMM_MAX_CASES || count > ZTT_GEMM_MAX_VALUES)
        return;
    const uint8_t *bytes = values;
    for (unsigned i = 0; i < count; ++i) {
        uint64_t bits = 0;
        memcpy(&bits, bytes + i * element_bytes, element_bytes);
        report->values[test][i] = bits;
    }
    report->case_values[test] = count;
}

#endif
