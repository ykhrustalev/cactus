#!/bin/bash -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

usage() {
    echo "Usage: $0 <model_name> [options]"
    echo ""
    echo "Convert a HuggingFace model and upload to S3"
    echo ""
    echo "Arguments:"
    echo "  model_name           HuggingFace model name (e.g., LiquidAI/LFM2-1.2B)"
    echo ""
    echo "Options:"
    echo "  --precision          Precision: INT8, FP16, FP32 (default: INT8)"
    echo "  --output-dir         Output directory (default: ./converted_models/<model_name>)"
    echo "  --s3-bucket          S3 bucket (default: liquid-eb-resources)"
    echo "  --aws-profile        AWS profile (default: dev)"
    echo "  --skip-upload        Skip S3 upload, only convert and zip"
    echo "  --help               Show this help message"
    exit 1
}

MODEL_NAME=""
PRECISION="INT8"
OUTPUT_DIR=""
S3_BUCKET="liquid-eb-resources"
AWS_PROFILE="dev"
SKIP_UPLOAD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --s3-bucket)
            S3_BUCKET="$2"
            shift 2
            ;;
        --aws-profile)
            AWS_PROFILE="$2"
            shift 2
            ;;
        --skip-upload)
            SKIP_UPLOAD=true
            shift
            ;;
        --help)
            usage
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            if [ -z "$MODEL_NAME" ]; then
                MODEL_NAME="$1"
            else
                echo "Unexpected argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

if [ -z "$MODEL_NAME" ]; then
    echo "Error: model_name is required"
    usage
fi

# Derive a safe directory name from model name
MODEL_SAFE_NAME=$(echo "$MODEL_NAME" | tr '/' '_' | tr ':' '_')

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$PROJECT_ROOT/converted_models/$MODEL_SAFE_NAME"
fi

echo "Converting model: $MODEL_NAME"
echo "Precision: $PRECISION"
echo "Output directory: $OUTPUT_DIR"

# Run conversion
python3 "$SCRIPT_DIR/convert_hf.py" "$MODEL_NAME" "$OUTPUT_DIR" --precision "$PRECISION"

# Get git hash
GIT_HASH=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)

# Create zip archive
ZIP_NAME="${MODEL_SAFE_NAME}-${PRECISION}-${GIT_HASH}.zip"
ZIP_PATH="$OUTPUT_DIR/../$ZIP_NAME"

echo "Creating archive: $ZIP_NAME"
(cd "$OUTPUT_DIR" && zip -r "../$ZIP_NAME" .)

echo "Archive created: $ZIP_PATH"

# Upload to S3
if [ "$SKIP_UPLOAD" = false ]; then
    echo "Uploading $ZIP_NAME to s3://$S3_BUCKET/models/..."
    aws s3 cp "$ZIP_PATH" "s3://$S3_BUCKET/models/$ZIP_NAME" --profile "$AWS_PROFILE"
    echo "Upload complete: s3://$S3_BUCKET/models/$ZIP_NAME"
else
    echo "Skipping S3 upload (--skip-upload specified)"
fi

echo "Done!"
