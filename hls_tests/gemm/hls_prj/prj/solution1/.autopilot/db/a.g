#!/bin/sh
lli=${LLVMINTERP-lli}
exec $lli \
    /curr/jaywang/research/systolic_compile/ppcg/polysa/hls_tests/gemm/hls_prj/prj/solution1/.autopilot/db/a.g.bc ${1+"$@"}
