# Contributing to Wildfire Agent

Welcome to the Wildfire Agent project! We're excited that you're interested in contributing to this AI-powered wildfire management and analysis system. This guide will help you get started with development, understand the project architecture, and contribute effectively.

## 📋 Table of Contents

- [Project Introduction](#-project-introduction)
- [Development Environment Setup](#-development-environment-setup)
- [Project Structure](#-project-structure)
- [Agent Architecture](#-agent-architecture)
- [Available Toolkits and Tools](#-available-toolkits-and-tools)
- [Creating Custom Toolkits](#-creating-custom-toolkits)
- [Writing Tests](#-writing-tests)
- [Contribution Guidelines](#-contribution-guidelines)
- [Code Style and Standards](#-code-style-and-standards)
- [Submitting Contributions](#-submitting-contributions)

## 🔥 Project Introduction

Wildfire Agent is a specialized AI system for wildfire disaster management, emergency response, and geospatial analysis. Built on the proven [OWL framework](https://github.com/camel-ai/owl) and [CAMEL-AI](https://github.com/camel-ai/camel) architecture, it provides intelligent assistance for:

- **Real-time wildfire monitoring** with satellite imagery analysis
- **Emergency evacuation planning** with route optimization
- **Risk assessment** for communities and infrastructure
- **YOLO-based object detection** for fire, smoke, and infrastructure
- **Multi-agent coordination** for complex analysis tasks

### Key Technologies

- **CAMEL-AI Framework**: Multi-agent role-playing architecture
- **OWL Platform**: Optimized workforce learning for task automation
- **YOLO Computer Vision**: Real-time object detection for wildfire scenarios
- **Qwen Vision Language Models**: Advanced satellite imagery analysis
- **Gradio Web Interface**: User-friendly emergency management dashboard

## 🛠️ Development Environment Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- [Conda](https://docs.conda.io/en/latest/miniconda.html) package manager
- Git for version control

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/wildfire-agent.git
cd wildfire-agent
```

### Step 2: Create Conda Environment

**Important**: We strongly recommend using conda for dependency management, especially for computer vision components.

```bash
# Create conda environment with Python 3.10
conda create -n wildfire-agent python=3.10 -y
conda activate wildfire-agent

# Install core Python dependencies
pip install -r requirements.txt

# Install computer vision dependencies via conda (recommended)
conda install ultralytics opencv pytorch torchvision torchaudio -c pytorch -y

# Verify YOLO installation
python -c "from ultralytics import YOLO; print('✅ YOLO ready!')"
```

### Step 3: Install System Dependencies (Optional)

For advanced GIS capabilities:

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install gdal-bin python3-gdal qgis grass

# macOS with Homebrew
brew install gdal qgis grass
```

### Step 4: Environment Configuration

```bash
# Copy environment template
cp .env_template .env

# Configure API keys in .env file
# Required: QWEN_API_KEY for vision analysis
# Optional: OPENAI_API_KEY, GOOGLE_API_KEY, etc.
```

### Step 5: Verify Installation

```bash
# Run tool tests to verify setup
python tests/run_tool_tests.py

# Test wildfire agent
python examples/run_qwen_wildfire.py "Test system functionality"

# Launch web interface
python owl/webapp_wildfire.py
```

## 📁 Project Structure

```
wildfire-agent/
├── examples/                          # Example scripts and usage demos
│   ├── run_qwen_wildfire.py          # Main wildfire agent script
│   └── run*.py                       # Other model examples
├── owl/                              # Core OWL framework integration
│   ├── utils/                        # Utility modules and toolkits
│   │   ├── wildfire_yolo_toolkit.py  # YOLO detection toolkit
│   │   ├── document_toolkit.py       # Document processing
│   │   └── common.py                 # Shared utilities
│   ├── webapp_wildfire.py            # Gradio web interface
│   └── logs/                         # Application logs
├── tests/                            # Comprehensive test suite
│   ├── tool_tests/                   # Direct tool testing
│   │   └── test_wildfire_yolo_toolkit.py
│   ├── run_tool_tests.py             # Test runner with dependency checking
│   └── README.md                     # Testing documentation
├── wildfire_workspace/               # Auto-generated analysis workspaces
│   ├── latest/                       # Symlink to most recent session
│   └── session_YYYYMMDD_HHMMSS/      # Timestamped session directories
├── workspace/                        # Sample data and images
│   └── Maui Wildfires Image.jpg      # Example satellite imagery
├── requirements.txt                  # Python dependencies
├── .env_template                     # Environment variable template
├── README.md                         # Project documentation
└── CONTRIBUTING.md                   # This file
```

### Key Directories

- **`examples/`**: Entry points and usage demonstrations
- **`owl/utils/`**: Custom toolkits and framework extensions
- **`tests/`**: Unittest framework for tool validation
- **`wildfire_workspace/`**: Organized output from agent sessions
- **`owl/webapp_wildfire.py`**: Web interface for emergency managers

## 🤖 Agent Architecture

### Multi-Agent Design

Wildfire Agent uses a **RolePlaying society pattern** with specialized agents:

```python
# Agent Roles
society = RolePlaying(
    user_role_name="Emergency Manager",      # Coordinates wildfire response
    assistant_role_name="Wildfire AI Agent", # Performs analysis tasks
    user_agent_kwargs={"model": models["user"]},
    assistant_agent_kwargs={"model": models["assistant"], "tools": tools}
)
```

### Model Configuration

Different models optimized for specific tasks:

```python
models = {
    "user": ModelType.QWEN_MAX,           # Task coordination
    "assistant": ModelType.QWEN_MAX,      # General analysis
    "browsing": ModelType.QWEN_VL_MAX,    # Web browsing with vision
    "image": ModelType.QWEN_VL_MAX,       # Satellite imagery analysis
    "video": ModelType.QWEN_VL_MAX,       # Aerial footage analysis
    "document": ModelType.QWEN_VL_MAX,    # Document processing
}
```

### Workspace Management

Automatic workspace creation for organized analysis:

```python
def setup_wildfire_workspace() -> pathlib.Path:
    """Create timestamped workspace with specialized directories"""
    # Creates: satellite_imagery/, analysis_results/, maps_and_visualizations/
    # evacuation_plans/, risk_assessments/, documents/, code_execution/, temp/
```

## 🧰 Available Toolkits and Tools

### Core Toolkits

| Toolkit | Purpose | Key Tools |
|---------|---------|-----------|
| **WildfireYOLOToolkit** | Object detection for fire, smoke, infrastructure | `detect_objects_in_image()`, `analyze_wildfire_image()` |
| **ImageAnalysisToolkit** | LLM-powered satellite imagery analysis | Vision analysis, scene understanding |
| **VideoAnalysisToolkit** | Aerial footage and drone video analysis | Video processing, temporal analysis |
| **BrowserToolkit** | Web browsing for real-time data | Weather data, fire reports, news |
| **SearchToolkit** | Web search capabilities | DuckDuckGo, Google, Wikipedia search |
| **CodeExecutionToolkit** | Python script execution | GIS calculations, data processing |
| **FileWriteToolkit** | Organized file operations | Reports, maps, analysis outputs |
| **DocumentProcessingToolkit** | Document analysis and generation | PDF processing, report creation |
| **ExcelToolkit** | Spreadsheet data analysis | Population data, infrastructure lists |

### YOLO Toolkit Example

```python
from owl.utils.wildfire_yolo_toolkit import WildfireYOLOToolkit

# Initialize toolkit
yolo_toolkit = WildfireYOLOToolkit(
    model_path="yolo11n.pt",
    confidence_threshold=0.3,
    output_dir="./detections"
)

# Detect objects
result = yolo_toolkit.detect_objects_in_image("satellite_image.jpg")

# Wildfire-specific analysis
analysis = yolo_toolkit.analyze_wildfire_image("wildfire_scene.jpg")
```

### Search Tools Example

```python
# Web search for current fire conditions
search_toolkit = SearchToolkit()
fire_info = search_toolkit.search_google("California wildfire current status 2024")
weather_data = search_toolkit.search_duckduckgo("weather forecast fire risk")
```

## 🔧 Creating Custom Toolkits

### Step 1: Inherit from BaseToolkit

```python
from camel.toolkits.base import BaseToolkit
from typing import List
import logging

class MyCustomToolkit(BaseToolkit):
    r"""Custom toolkit for specialized wildfire analysis.
    
    This toolkit provides functionality for:
    - Custom analysis capability 1
    - Custom analysis capability 2
    - Integration with external services
    """
    
    def __init__(self, config_param: str = "default_value"):
        r"""Initialize the custom toolkit.
        
        Args:
            config_param (str): Configuration parameter for the toolkit.
        """
        self.config_param = config_param
        self.logger = logging.getLogger(__name__)
        
        # Initialize any required resources
        self._setup_resources()
    
    def _setup_resources(self):
        """Initialize toolkit resources (databases, models, etc.)"""
        pass
```

### Step 2: Implement Tool Functions

```python
def my_analysis_function(
    self,
    input_data: str,
    analysis_type: str = "comprehensive"
) -> str:
    r"""Perform custom analysis on input data.
    
    Args:
        input_data (str): Input data to analyze
        analysis_type (str): Type of analysis to perform
        
    Returns:
        str: Analysis results in JSON format
    """
    try:
        # Perform your custom analysis logic
        results = self._perform_analysis(input_data, analysis_type)
        
        # Return structured results
        return json.dumps({
            "success": True,
            "analysis_type": analysis_type,
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        self.logger.error(f"Analysis failed: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "analysis_type": analysis_type
        })

def _perform_analysis(self, data: str, analysis_type: str) -> dict:
    """Internal analysis logic"""
    # Implement your analysis algorithm here
    return {"processed_data": f"analyzed_{data}"}
```

### Step 3: Register Tools

```python
def get_tools(self) -> List:
    r"""Get the list of tools provided by this toolkit.
    
    Returns:
        List: List of tool functions available to agents
    """
    return [
        self.my_analysis_function,
        # Add more tool functions here
    ]
```

### Step 4: Add Error Handling and Logging

```python
import json
import logging
from datetime import datetime
from pathlib import Path

class MyCustomToolkit(BaseToolkit):
    def __init__(self, output_dir: str = "./custom_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def save_results(self, results: dict, filename: str):
        """Save analysis results to workspace"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to: {output_path}")
        return str(output_path)
```

### Step 5: Integration with Wildfire Agent

```python
# In examples/run_qwen_wildfire.py, add your toolkit:

from owl.utils.my_custom_toolkit import MyCustomToolkit

def construct_society(question: str, workspace_dir: pathlib.Path = None) -> RolePlaying:
    # ... existing code ...
    
    # Add your custom toolkit to the tools list
    tools = [
        # ... existing toolkits ...
        *MyCustomToolkit(output_dir=str(workspace_paths['analysis_results'])).get_tools(),
    ]
    
    # ... rest of the function ...
```

### Toolkit Development Best Practices

1. **Follow naming conventions**: Use descriptive names ending with "Toolkit"
2. **Implement proper error handling**: Always return JSON with success/error status
3. **Add comprehensive logging**: Use the logging module for debugging
4. **Save results to workspace**: Integrate with the workspace directory structure
5. **Write docstrings**: Document all public methods and parameters
6. **Handle dependencies gracefully**: Check for optional dependencies and provide fallbacks
7. **Return structured data**: Use consistent JSON response formats

### Example: Weather Data Toolkit

```python
class WeatherDataToolkit(BaseToolkit):
    r"""Toolkit for retrieving weather data relevant to wildfire risk."""
    
    def __init__(self, api_key: str, output_dir: str = "./weather_data"):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_fire_weather_forecast(self, latitude: float, longitude: float) -> str:
        r"""Get weather forecast for fire risk assessment.
        
        Args:
            latitude (float): Latitude coordinate
            longitude (float): Longitude coordinate
            
        Returns:
            str: Weather forecast data in JSON format
        """
        try:
            # API call logic here
            weather_data = self._fetch_weather_data(latitude, longitude)
            
            # Calculate fire risk indicators
            fire_risk = self._calculate_fire_risk(weather_data)
            
            results = {
                "location": {"lat": latitude, "lon": longitude},
                "weather": weather_data,
                "fire_risk": fire_risk,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to workspace
            output_file = f"weather_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.save_results(results, output_file)
            
            return json.dumps(results)
            
        except Exception as e:
            return json.dumps({"error": str(e), "success": False})
    
    def get_tools(self) -> List:
        return [self.get_fire_weather_forecast]
```

## 🧪 Writing Tests

### Test Structure

We use Python's unittest framework with automatic dependency checking:

```python
import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from owl.utils.my_custom_toolkit import MyCustomToolkit

class TestMyCustomToolkit(unittest.TestCase):
    """Test cases for MyCustomToolkit"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.toolkit = MyCustomToolkit(output_dir=str(self.temp_dir))
    
    def tearDown(self):
        """Clean up after tests"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_toolkit_initialization(self):
        """Test toolkit initializes correctly"""
        self.assertIsInstance(self.toolkit, MyCustomToolkit)
        self.assertTrue(self.temp_dir.exists())
    
    def test_analysis_function_success(self):
        """Test successful analysis"""
        result = self.toolkit.my_analysis_function("test_data")
        result_data = json.loads(result)
        
        self.assertTrue(result_data["success"])
        self.assertIn("results", result_data)
    
    def test_analysis_function_error_handling(self):
        """Test error handling"""
        # Test with invalid input
        result = self.toolkit.my_analysis_function("")
        result_data = json.loads(result)
        
        self.assertFalse(result_data["success"])
        self.assertIn("error", result_data)
    
    @patch('owl.utils.my_custom_toolkit.external_api_call')
    def test_external_dependency(self, mock_api):
        """Test with mocked external dependencies"""
        mock_api.return_value = {"status": "success", "data": "test"}
        
        result = self.toolkit.my_analysis_function("test")
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
```

### Running Tests

```bash
# Run all tool tests
python tests/run_tool_tests.py

# Run specific toolkit tests
python tests/run_tool_tests.py --filter test_my_custom_toolkit

# Include LLM-dependent tests (requires API keys)
python tests/run_tool_tests.py --include-llm

# List available test modules
python tests/run_tool_tests.py --list
```

### Test Coverage Requirements

1. **Initialization testing**: Verify toolkit setup and configuration
2. **Function testing**: Test all public methods with valid inputs
3. **Error handling**: Test error conditions and edge cases
4. **Dependency testing**: Mock external dependencies and APIs
5. **Integration testing**: Test toolkit integration with agent framework
6. **File operations**: Test workspace file creation and management

### Adding New Test Modules

1. Create `tests/tool_tests/test_your_toolkit.py`
2. Follow the pattern in existing test files
3. Add dependency requirements to `tests/run_tool_tests.py`
4. Update the dependency map for automatic checking

## 💻 Contribution Guidelines

### Types of Contributions

We welcome several types of contributions:

1. **New Toolkits**: Custom tools for specialized wildfire analysis
2. **Bug Fixes**: Improvements to existing functionality
3. **Documentation**: Better guides, examples, and API documentation
4. **Tests**: Expanded test coverage for existing and new features
5. **Performance**: Optimizations and efficiency improvements
6. **UI/UX**: Web interface enhancements and user experience improvements

### Development Workflow

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`
3. **Set up development environment** using conda
4. **Make your changes** following our coding standards
5. **Write tests** for new functionality
6. **Run the test suite** to ensure nothing breaks
7. **Update documentation** as needed
8. **Submit a pull request** with a clear description

### Branch Naming Convention

- `feature/toolkit-name` - New toolkit development
- `fix/issue-description` - Bug fixes
- `docs/section-name` - Documentation updates
- `test/toolkit-name` - Test improvements

### Commit Message Format

```
Add YOLO wildfire detection toolkit with comprehensive testing

- Implement WildfireYOLOToolkit for object detection
- Add fire, smoke, and infrastructure detection capabilities  
- Include automated risk assessment and recommendations
- Add comprehensive unittest suite with 100% pass rate
- Update documentation with usage examples

Fixes #123
```

## 📏 Code Style and Standards

### Python Code Style

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and returns
- Write **comprehensive docstrings** for all public methods
- Use **descriptive variable names** and avoid abbreviations
- **Maximum line length**: 88 characters (Black formatter)

### Docstring Format

```python
def analyze_wildfire_image(
    self,
    image_path: str,
    save_results: bool = True
) -> str:
    r"""Analyze wildfire-related objects in satellite/aerial imagery.
    
    Args:
        image_path (str): Path to the wildfire image
        save_results (bool): Whether to save results. Defaults to True.
        
    Returns:
        str: Detailed wildfire analysis results in JSON format
        
    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If image format is not supported
        
    Example:
        >>> toolkit = WildfireYOLOToolkit()
        >>> result = toolkit.analyze_wildfire_image("maui_fire.jpg")
        >>> data = json.loads(result)
        >>> print(data["risk_assessment"]["overall_risk_level"])
        HIGH
    """
```

### Error Handling Standards

```python
def your_function(self, input_data: str) -> str:
    """Function with proper error handling"""
    try:
        # Main logic here
        results = self._process_data(input_data)
        
        return json.dumps({
            "success": True,
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except FileNotFoundError as e:
        self.logger.error(f"File not found: {e}")
        return json.dumps({
            "success": False,
            "error": f"File not found: {str(e)}",
            "error_type": "FileNotFoundError"
        })
    except Exception as e:
        self.logger.error(f"Unexpected error: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        })
```

### Import Organization

```python
# Standard library imports
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import cv2
import numpy as np
from ultralytics import YOLO

# Local imports
from camel.toolkits.base import BaseToolkit
from owl.utils.common import setup_logging
```

## 🚀 Submitting Contributions

### Pull Request Checklist

Before submitting your pull request, ensure:

- [ ] **Code follows style guidelines** (PEP 8, type hints, docstrings)
- [ ] **All tests pass** (`python tests/run_tool_tests.py`)
- [ ] **New functionality includes tests** (aim for >90% coverage)
- [ ] **Documentation is updated** (README, docstrings, examples)
- [ ] **Commit messages are descriptive** and follow our format
- [ ] **No API keys or secrets** are committed to the repository
- [ ] **Dependencies are properly declared** in requirements or conda environment

### Pull Request Template

```markdown
## Description
Brief description of the changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## New Toolkit Checklist (if applicable)
- [ ] Inherits from BaseToolkit
- [ ] Implements get_tools() method
- [ ] Includes comprehensive error handling
- [ ] Has proper logging integration
- [ ] Saves results to workspace
- [ ] Includes unit tests with >90% coverage
- [ ] Updated integration example

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Code is well-documented with docstrings
- [ ] README updated if needed
- [ ] Examples provided for new functionality
```

### Review Process

1. **Automated checks**: GitHub Actions will run tests and style checks
2. **Code review**: Maintainers will review your code for quality and style
3. **Testing**: We'll test your changes in different environments
4. **Documentation review**: Ensure documentation is clear and complete
5. **Integration testing**: Verify changes work with existing functionality

### Getting Help

- **GitHub Issues**: Report bugs or request features
- **GitHub Discussions**: Ask questions about development
- **Documentation**: Check existing docs and examples first

## 🎯 Specific Contribution Areas

### High Priority Areas

1. **Additional YOLO Models**: Integration with specialized fire detection models
2. **GIS Toolkits**: PostGIS, QGIS, GRASS GIS integration
3. **Weather APIs**: Real-time weather data integration
4. **Satellite Data**: Additional satellite imagery providers
5. **Evacuation Planning**: Advanced routing and population analysis
6. **Risk Modeling**: Fire behavior prediction models

### Example Contribution Ideas

- **Fire Behavior Toolkit**: Physics-based fire spread modeling
- **Population Analytics Toolkit**: Census and demographic analysis
- **Infrastructure Assessment Toolkit**: Critical infrastructure vulnerability analysis
- **Weather Integration Toolkit**: Real-time meteorological data
- **Satellite Data Toolkit**: MODIS, Landsat, Sentinel imagery processing
- **Emergency Communications Toolkit**: Alert system integration

Thank you for contributing to Wildfire Agent! Your contributions help build better tools for wildfire management and emergency response. Together, we can make communities safer and more resilient to wildfire threats. 🔥🚒

---

**Questions?** Feel free to open an issue or start a discussion on GitHub.