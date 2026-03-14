#!/bin/bash
# Build the omsim shared library (libverify) for the current platform.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMSIM_DIR="$SCRIPT_DIR/vendor/omsim"
LIB_DIR="$SCRIPT_DIR/lib"

if [ ! -d "$OMSIM_DIR" ]; then
    echo "Error: omsim source not found at $OMSIM_DIR"
    echo "Run: git submodule update --init"
    exit 1
fi

mkdir -p "$LIB_DIR"

SOURCES="collision.c decode.c parse.c sim.c steady-state.c verifier.c"
CFLAGS="-O2 -std=c11 -pedantic -Wall -Wno-missing-braces"

OS="$(uname -s)"
case "$OS" in
    Darwin)
        OUTPUT="$LIB_DIR/libverify.dylib"
        cc $CFLAGS -shared -fpic -o "$OUTPUT" \
            $(cd "$OMSIM_DIR" && echo $SOURCES | tr ' ' '\n' | sed "s|^|$OMSIM_DIR/|") \
            -lm
        ;;
    Linux)
        OUTPUT="$LIB_DIR/libverify.so"
        cc $CFLAGS -shared -fpic -o "$OUTPUT" \
            $(cd "$OMSIM_DIR" && echo $SOURCES | tr ' ' '\n' | sed "s|^|$OMSIM_DIR/|") \
            -lm
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

echo "Built $OUTPUT"
