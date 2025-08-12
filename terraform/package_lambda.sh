#!/bin/bash
# Package Lambda function for deployment

set -e

echo "📦 Packaging Lambda function for batch processing..."

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Using temp directory: $TEMP_DIR"

# Copy Lambda function
cp lambda_batch_processor.py $TEMP_DIR/index.py

# Create requirements file for Lambda dependencies
cat > $TEMP_DIR/requirements.txt <<EOF
boto3==1.26.137
EOF

# Install dependencies (if any beyond boto3 which is included in Lambda)
# cd $TEMP_DIR
# pip install -r requirements.txt -t . --no-deps

# Create deployment package
cd $TEMP_DIR
zip -r lambda_batch_processor.zip index.py

# Move package to terraform directory
mv lambda_batch_processor.zip $OLDPWD/

# Cleanup
cd $OLDPWD
rm -rf $TEMP_DIR

echo "✅ Lambda package created: lambda_batch_processor.zip"
echo "   Size: $(du -h lambda_batch_processor.zip | cut -f1)"
echo ""
echo "📌 Note: This is a placeholder Lambda function."
echo "   For production, integrate with actual ML inference code."