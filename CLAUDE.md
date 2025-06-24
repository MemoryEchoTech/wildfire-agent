# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Wildfire Agent** is a specialized AI system for wildfire disaster management built on the OWL (Optimized Workforce Learning) framework from CAMEL-AI. The system uses a multi-agent architecture where specialized agents collaborate to handle complex wildfire management tasks including satellite imagery analysis, evacuation planning, risk assessment, and emergency response coordination.

## Architecture

### Core Framework
- **Base**: Built on `camel-ai[owl]==0.2.57` multi-agent framework
- **Pattern**: RolePlaying society with User Agent + Assistant Agent + specialized sub-agents
- **Specialization**: Domain-specific adaptation of the general-purpose OWL framework for wildfire management

### Key Components
- **`/owl/utils/enhanced_role_playing.py`**: `OwlRolePlaying` class that extends CAMEL's base RolePlaying with enhanced capabilities
- **`/owl/utils/document_toolkit.py`**: `DocumentProcessingToolkit` for multi-format document processing (images, PDFs, Excel, audio, zip)
- **`/owl/webapp.py`**: Gradio web interface with real-time logging and model selection
- **`/examples/`**: Multiple entry points for different usage patterns (CLI, web, programmatic, model-specific)

### Multi-Agent Coordination
Agents communicate through the RolePlaying society pattern where:
- Complex tasks are decomposed and distributed among specialized agents
- Each agent has access to specific toolkits (Browser, Code Execution, Search, etc.)
- Results are synthesized through inter-agent communication

## Development Commands

### Basic Usage
```bash
# Run with default settings
python examples/run.py

# Interactive CLI with model selection
python examples/run_cli.py

# Web interface (recommended for complex tasks)
python owl/webapp.py
# Access at http://localhost:7860
```

### Environment Setup
```bash
# Configure API keys
cp owl/.env_template owl/.env
# Edit .env with: OPENAI_API_KEY, QWEN_API_KEY, GOOGLE_API_KEY, etc.
```

### Model-Specific Execution
```bash
# OpenAI models
python examples/run.py

# Claude models
python examples/run_claude.py

# Qwen models (good for vision tasks)
python examples/run_qwen_zh.py

# MCP integration
python examples/run_mcp.py
```

### Testing & Validation
- No formal test suite currently exists
- Validation is done through example tasks and community use cases
- Web interface provides real-time feedback and debugging

## Toolkit Integration

### Core Toolkits Available
When configuring agents, you can include these toolkits:
```python
from camel.toolkits import (
    BrowserToolkit,          # Web automation via Playwright
    VideoAnalysisToolkit,    # Video processing
    AudioAnalysisToolkit,    # Audio transcription
    CodeExecutionToolkit,    # Safe code execution
    ImageAnalysisToolkit,    # Vision Language Models
    SearchToolkit,           # Multiple search engines
    ExcelToolkit,           # Spreadsheet processing
    FileWriteToolkit,       # File operations
)
```

### Wildfire-Specific Extensions (Planned)
The roadmap includes integration with:
- **PostGIS**: Spatial database operations
- **QGIS**: Professional GIS analysis
- **GRASS GIS**: Geospatial modeling
- **GDAL/OGR**: Geospatial data processing
- **Remote Sensing**: VLM-powered satellite imagery analysis

## Key Configuration Patterns

### Society Construction
```python
def construct_society(question: str) -> RolePlaying:
    # Configure models for different roles
    models = {
        "assistant": ModelFactory.create(...),
        "user": ModelFactory.create(...),
        "browsing": ModelFactory.create(...),
        # ... other specialized models
    }
    
    # Configure toolkits
    tools = [
        *BrowserToolkit().get_tools(),
        *CodeExecutionToolkit().get_tools(),
        # ... other toolkits
    ]
    
    # Create specialized agents
    society = OwlRolePlaying(
        assistant_agent_kwargs={"model": models["assistant"], "tools": tools},
        user_agent_kwargs={"model": models["user"]},
        # ... other configurations
    )
    
    return society
```

### Model Selection Strategy
- **OpenAI GPT-4**: Recommended for complex reasoning and tool use
- **Qwen VL**: Excellent for vision/image analysis tasks
- **Claude**: Good alternative for reasoning tasks
- **Gemini**: Multimodal capabilities
- **Local Models**: Ollama for privacy-sensitive deployments

## Wildfire Domain Specialization

### Current Capabilities
- Multi-agent task decomposition
- Document and image processing
- Web search and information gathering
- Code execution for data analysis
- Multi-modal understanding (text, images, audio, video)

### Planned Wildfire Features
Based on `/my_notes/agents_wildfire_plan.md`:
- **Fire Analysis**: Burn area mapping, spread rate calculation, severity assessment
- **Emergency Planning**: Evacuation route optimization, population analysis
- **Risk Assessment**: Infrastructure vulnerability, ecological impact
- **Resource Allocation**: Emergency response coordination

### Example Queries for Testing
```python
# Fire analysis
"How large is the area currently burning in the latest satellite image?"
"What is the rate of fire spread and which direction is it moving?"

# Emergency planning  
"How many people need immediate evacuation from the threatened area?"
"What are the safest evacuation routes for the affected communities?"

# Risk assessment
"What wildlife habitats and protected areas are impacted?"
"Which critical infrastructure is at risk from the current fire?"
```

## Community Extensions

The `/community_usecase/` directory contains examples of how to specialize the framework:
- **Stock Analysis**: Multi-agent investment analysis with debate rooms
- **Resume Analysis**: Automated screening with MCP integration
- **Excel Analysis**: Data analysis workflows
- **Interview Preparation**: AI-powered coaching systems

These serve as architectural patterns for building domain-specific applications.

## Important Notes

### Performance Considerations
- OpenAI models generally provide the best performance for complex multi-step reasoning
- For vision tasks, consider Qwen VL models
- Browser automation can be resource-intensive; use headless mode when possible
- Large document processing may require chunking strategies

### Environment Variables
Critical API keys needed:
- `OPENAI_API_KEY`: For GPT models and audio processing
- `QWEN_API_KEY`: For vision analysis capabilities
- `GOOGLE_API_KEY`: For search functionality
- Model-specific keys for alternative providers

### Debugging
- Use the web interface (`owl/webapp.py`) for interactive debugging
- Enable debug logging with `set_log_level(level="DEBUG")`
- Check agent communication flow in the RolePlaying society logs
- Toolkit errors often indicate missing API keys or dependencies