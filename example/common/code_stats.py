#!/usr/bin/env python3
#
# Copyright (c) 2026 BOSC & ICT, CAS
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

"""Print basic source statistics for the example tree."""

from __future__ import annotations

import sys
from pathlib import Path


CATEGORIES = {
    "C/C++": {".c", ".cc", ".cpp", ".h", ".hh", ".hpp"},
    "Assembly": {".s", ".S"},
    "Python": {".py"},
    "Make": {"", ".mk"},
    "JSON": {".json"},
}


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    totals = {category: [0, 0, 0] for category in CATEGORIES}
    for path in root.rglob("*"):
        if not path.is_file() or "build" in path.parts or "__pycache__" in path.parts:
            continue
        category = next(
            (
                name
                for name, suffixes in CATEGORIES.items()
                if path.suffix in suffixes
                and (name != "Make" or path.name == "Makefile" or path.suffix == ".mk")
            ),
            None,
        )
        if category is None:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.count("\n") + bool(content and not content.endswith("\n"))
        totals[category][0] += 1
        totals[category][1] += int(lines)
        totals[category][2] += path.stat().st_size

    total = tuple(sum(values[index] for values in totals.values()) for index in range(3))
    print("\nCode statistics (build output excluded):")
    print("  category        files   lines   bytes")
    for category, (files, lines, byte_count) in totals.items():
        print(f"  {category:<14} {files:>5} {lines:>7} {byte_count:>8}")
    print(f"  {'total':<14} {total[0]:>5} {total[1]:>7} {total[2]:>8}")


if __name__ == "__main__":
    main()
