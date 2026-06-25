#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
vcs -full64 -sverilog -timescale=1ns/1ps -debug_access+all \
    -f filelist_vcs.f \
    -top tb_vit_fulldesign_compile \
    -l logs/vcs_compile.log
./simv -l logs/vcs_run.log
