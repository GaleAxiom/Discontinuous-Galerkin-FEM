#!/bin/bash
# Format all C++ source files using clang-format

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Find clang-format
CLANG_FORMAT=""
if command -v clang-format &> /dev/null; then
    CLANG_FORMAT="clang-format"
elif [ -f "/opt/homebrew/opt/llvm/bin/clang-format" ]; then
    CLANG_FORMAT="/opt/homebrew/opt/llvm/bin/clang-format"
elif [ -f "/usr/local/opt/llvm/bin/clang-format" ]; then
    CLANG_FORMAT="/usr/local/opt/llvm/bin/clang-format"
else
    echo "Error: clang-format not found. Please install it:"
    echo "  - macOS: brew install llvm"
    echo "  - Ubuntu: sudo apt-get install clang-format"
    exit 1
fi

echo -e "${BLUE}Using clang-format: $CLANG_FORMAT${NC}"
echo -e "${BLUE}Formatting C++ files in dgfem/...${NC}"

# Find and format all .cpp and .hpp files
find dgfem/src dgfem/include dgfem/tests dgfem/examples -type f \( -name "*.cpp" -o -name "*.hpp" \) | while read -r file; do
    echo -e "  Formatting: ${GREEN}$file${NC}"
    "$CLANG_FORMAT" -i "$file"
done

echo -e "${BLUE}Done!${NC}"
