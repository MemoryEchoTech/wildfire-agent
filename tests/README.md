# Wildfire Agent Test Suite

Comprehensive testing framework for the Wildfire Agent system with support for both standalone tool testing and integrated agent testing.

## Test Structure

```
tests/
├── __init__.py                     # Test package initialization
├── README.md                       # This documentation
├── run_tool_tests.py              # Main test runner with dependency checking
├── tool_tests/                    # Direct tool testing (no LLM required)
│   ├── __init__.py
│   └── test_wildfire_yolo_toolkit.py  # YOLO toolkit unit tests
└── test_data/                     # Test data and fixtures
    └── __init__.py
```

## Running Tests

### Quick Start

```bash
# Run all available tool tests
python tests/run_tool_tests.py

# List available test modules
python tests/run_tool_tests.py --list

# Run specific tool tests
python tests/run_tool_tests.py --filter test_wildfire_yolo_toolkit

# Include LLM-dependent tests (requires API keys)
python tests/run_tool_tests.py --include-llm
```

### Test Categories

#### 1. Tool Tests (Non-LLM)
Tests individual tools directly without invoking agents or LLMs:

- ✅ **YOLO Toolkit** - Object detection and wildfire analysis
- 🔧 **Code Execution Toolkit** - Script execution capabilities  
- 🔧 **Search Toolkit** - Web search functionality
- 🔧 **File Write Toolkit** - File operations

#### 2. LLM-Dependent Tool Tests
Tests tools that require LLM capabilities (skipped by default):

- 🤖 **Image Analysis Toolkit** - LLM-powered image understanding
- 🤖 **Video Analysis Toolkit** - LLM-powered video understanding
- 🤖 **Document Processing Toolkit** - LLM-powered document analysis

## Test Features

### Dependency Management
- Automatic dependency checking before running tests
- Graceful skipping of tests with missing dependencies
- Clear reporting of missing requirements

### Environment Flexibility
- Works with or without conda environments
- Automatically detects available tools and models
- Handles missing YOLO models gracefully

### Comprehensive Coverage
- Unit tests for tool initialization
- Functional tests with real data (Maui wildfire image)
- Error handling and edge case testing
- Integration verification

## YOLO Toolkit Tests

The YOLO toolkit tests demonstrate comprehensive testing patterns:

```python
# Test Categories Covered:
✅ Toolkit initialization and configuration
✅ Object detection with test images
✅ Wildfire-specific analysis functionality
✅ Real-world image processing (Maui wildfire)
✅ Error handling for invalid inputs
✅ Output directory management
✅ Tool discovery and registration
✅ Dependency availability checking
```

### Test Results Example

```
============================================================
TOOL TESTS SUMMARY
============================================================
Total modules: 1
Total tests run: 10
Failures: 0
Errors: 0
Skipped: 0

Success rate: 100.0%
============================================================
```

## Adding New Tool Tests

To add tests for a new tool:

1. Create `tests/tool_tests/test_your_toolkit.py`
2. Follow the pattern in `test_wildfire_yolo_toolkit.py`
3. Add dependency requirements to `run_tool_tests.py`
4. Mark LLM-dependent tools in the dependency map

### Test Template

```python
import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from your_module import YourToolkit

class TestYourToolkit(unittest.TestCase):
    def setUp(self):
        self.toolkit = YourToolkit()
    
    def test_initialization(self):
        self.assertIsInstance(self.toolkit, YourToolkit)
    
    def test_basic_functionality(self):
        result = self.toolkit.your_method()
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
```

## Dependencies

### Core Testing Dependencies
- `unittest` (built-in)
- `pathlib` (built-in)
- `json` (built-in)

### Tool-Specific Dependencies
- **YOLO Tests**: `ultralytics`, `opencv-python`, `torch`, `torchvision`
- **Image Tests**: `Pillow`, `opencv-python`  
- **Browser Tests**: `selenium`
- **Video Tests**: `opencv-python`, `ffmpeg`

## Benefits

1. **Fast Feedback** - Test tools independently without full agent initialization
2. **Reliable CI/CD** - Consistent test results across environments
3. **Debugging Support** - Isolate tool issues from agent complexity
4. **Documentation** - Tests serve as usage examples
5. **Quality Assurance** - Ensure tool reliability before integration

## Integration with CI/CD

The test runner returns appropriate exit codes for CI/CD integration:
- `0` - All tests passed
- `1` - Some tests failed or had errors

Example GitHub Actions integration:
```yaml
- name: Run Tool Tests
  run: python tests/run_tool_tests.py
```