@echo off
REM Full ONNX INT8 pipeline: Export -> Quantize -> Benchmark

setlocal enabledelayedexpansion

echo ========================================
echo ONNX INT8 Inference Pipeline
echo ========================================

REM Configuration
set MODEL_TYPE=%1
set QUANTIZER=%2

if "%MODEL_TYPE%"=="" set MODEL_TYPE=espcn
if "%QUANTIZER%"=="" set QUANTIZER=dsq

if /I not "%MODEL_TYPE%"=="espcn" (
    echo Error: ONNX pipeline currently supports only ESPCN.
    exit /b 1
)

set CHECKPOINT=results\quantized\espcn_%QUANTIZER%\espcn_%QUANTIZER%_final.pt
set ONNX_FP32=results\onnx\espcn_%QUANTIZER%_fp32.onnx
set BENCHMARK_JSON=results\onnx\espcn_%QUANTIZER%_benchmark.json
set NUM_SAMPLES=200
set BATCH_SIZE=16
set QUANT_METHOD=both

REM Check if checkpoint exists
if not exist "%CHECKPOINT%" (
    echo Error: Checkpoint not found: %CHECKPOINT%
    exit /b 1
)

REM Step 1: Export to ONNX
echo.
echo ========================================
echo Step 1: Exporting %MODEL_TYPE% to ONNX
echo ========================================
python scripts\export_to_onnx.py ^
    --model_type %MODEL_TYPE% ^
    --checkpoint %CHECKPOINT% ^
    --output %ONNX_FP32%

if not exist "%ONNX_FP32%" (
    echo Error: ONNX export failed!
    exit /b 1
)

echo ✓ Export successful: %ONNX_FP32%

REM Step 2: Quantize and Benchmark
echo.
echo ========================================
echo Step 2: INT8 Quantization ^& Benchmark
echo ========================================
python scripts\benchmark_onnx.py ^
    --model_type %MODEL_TYPE% ^
    --onnx_model %ONNX_FP32% ^
    --quantize %QUANT_METHOD% ^
    --output_dir results\onnx ^
    --num_samples %NUM_SAMPLES% ^
    --batch_size %BATCH_SIZE% ^
    --latency_batches 20 ^
    --calibration_samples 100 ^
    %EXTRA_ARGS% ^
    --output_json %BENCHMARK_JSON%

echo.
echo ========================================
echo Pipeline Complete!
echo ========================================
echo Results saved to: %BENCHMARK_JSON%
echo.
echo Output files:
echo   - FP32 ONNX:         %ONNX_FP32%
echo   - INT8 Dynamic:      %ONNX_FP32:~0,-5%_int8_dynamic.onnx
if "%QUANT_METHOD%"=="both" (
    echo   - INT8 Static:       %ONNX_FP32:~0,-5%_int8_static.onnx
)
echo   - Benchmark Results: %BENCHMARK_JSON%
echo.
echo To view results:
echo   type %BENCHMARK_JSON%
