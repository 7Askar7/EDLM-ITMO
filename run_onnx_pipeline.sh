#!/bin/bash
# Full ONNX INT8 pipeline: Export -> Quantize -> Benchmark

set -e

echo "========================================"
echo "ONNX INT8 Inference Pipeline"
echo "========================================"

MODEL_TYPE=${1:-espcn}
QUANTIZER=${2:-dsq}

if [ "$MODEL_TYPE" != "espcn" ]; then
    echo "Error: ONNX pipeline currently supports only ESPCN."
    exit 1
fi

CHECKPOINT="results/quantized/espcn_${QUANTIZER}/espcn_${QUANTIZER}_final.pt"
ONNX_FP32="results/onnx/espcn_${QUANTIZER}_fp32.onnx"
BENCHMARK_JSON="results/onnx/espcn_${QUANTIZER}_benchmark.json"
NUM_SAMPLES=200
BATCH_SIZE=16
QUANT_METHOD="both"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo ""
echo "========================================"
echo "Step 1: Exporting $MODEL_TYPE to ONNX"
echo "========================================"
# Use 'python' instead of assuming it's in path or alias, relying on active environment
python scripts/export_to_onnx.py \
    --model_type "$MODEL_TYPE" \
    --checkpoint "$CHECKPOINT" \
    --output "$ONNX_FP32"

if [ ! -f "$ONNX_FP32" ]; then
    echo "Error: ONNX export failed!"
    exit 1
fi

echo "✓ Export successful: $ONNX_FP32"

echo ""
echo "========================================"
echo "Step 2: INT8 Quantization & Benchmark"
echo "========================================"
python scripts/benchmark_onnx.py \
    --model_type "$MODEL_TYPE" \
    --onnx_model "$ONNX_FP32" \
    --quantize "$QUANT_METHOD" \
    --output_dir results/onnx \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --latency_batches 20 \
    --calibration_samples 100 \
    --output_json "$BENCHMARK_JSON"

echo ""
echo "========================================"
echo "Pipeline Complete!"
echo "========================================"
echo "Results saved to: $BENCHMARK_JSON"
echo ""
echo "Output files:"
echo "  - FP32 ONNX:         $ONNX_FP32"
echo "  - INT8 Dynamic:      ${ONNX_FP32%.onnx}_int8_dynamic.onnx"
if [ "$QUANT_METHOD" == "both" ] || [ "$QUANT_METHOD" == "static" ]; then
    echo "  - INT8 Static:       ${ONNX_FP32%.onnx}_int8_static.onnx"
fi
echo "  - Benchmark Results: $BENCHMARK_JSON"
echo ""
echo "To view results:"
echo "  cat $BENCHMARK_JSON | python -m json.tool"
