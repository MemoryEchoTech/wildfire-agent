# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

# Wildfire Agent - Specialized version for wildfire management tasks
# To run this file, you need to configure the Qwen API key
# You can obtain your API key from Bailian platform: bailian.console.aliyun.com
# Set it as QWEN_API_KEY="your-api-key" in your .env file or add it to your environment variables

import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.toolkits import (
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    BrowserToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.societies import RolePlaying

from owl.utils import run_society, DocumentProcessingToolkit

# Try to import YOLO toolkit
try:
    from owl.utils.wildfire_yolo_toolkit import WildfireYOLOToolkit
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLO toolkit not available. Install with: conda install ultralytics opencv pytorch torchvision -y")

from camel.logger import set_log_level


import pathlib

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))


def find_maui_image() -> str:
    """
    Automatically find the Maui wildfire image in the project.
    
    Returns:
        str: Path to the Maui wildfire image
    """
    possible_paths = [
        base_dir / "Maui Wildfires Image.jpg",  # Root directory
        base_dir / "workspace" / "Maui Wildfires Image.jpg",  # Workspace subdirectory
        base_dir / "wildfire_workspace" / "latest" / "satellite_imagery" / "Maui Wildfires Image.jpg",  # Latest workspace
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    # If not found, return the expected root path as default
    return str(base_dir / "Maui Wildfires Image.jpg")

set_log_level(level="DEBUG")


def setup_wildfire_workspace() -> pathlib.Path:
    """
    Create and setup a dedicated workspace directory for wildfire analysis.
    
    Returns:
        pathlib.Path: Path to the wildfire workspace directory
    """
    # Create workspace directory structure
    workspace_root = base_dir / "wildfire_workspace"
    
    # Create timestamped session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = workspace_root / f"session_{timestamp}"
    
    # Create subdirectories for different types of analysis
    subdirs = [
        "satellite_imagery",      # For downloaded/processed satellite images
        "analysis_results",       # For analysis outputs and reports
        "maps_and_visualizations", # For generated maps and charts
        "code_execution",         # For generated analysis code
        "documents",              # For reports and documentation
        "evacuation_plans",       # For evacuation route planning
        "risk_assessments",       # For risk analysis outputs
        "temp"                    # For temporary files
    ]
    
    # Create all directories
    for subdir in subdirs:
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Create a README file explaining the workspace structure
    readme_content = f"""# Wildfire Analysis Workspace - Session {timestamp}

This workspace contains all files and analysis results for the current wildfire management session.

## Directory Structure:
- `satellite_imagery/`: Downloaded and processed satellite images
- `analysis_results/`: Analysis outputs, calculations, and findings
- `maps_and_visualizations/`: Generated maps, charts, and visual outputs
- `code_execution/`: Python scripts and analysis code
- `documents/`: Reports, summaries, and documentation
- `evacuation_plans/`: Evacuation route planning and logistics
- `risk_assessments/`: Risk analysis and vulnerability assessments
- `temp/`: Temporary files (cleaned up automatically)

## Usage:
All wildfire agent operations will automatically save results to the appropriate subdirectories.
File paths will be provided in analysis outputs for easy access to generated content.
"""
    
    with open(session_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    # Create latest symlink for easy access
    latest_link = workspace_root / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(session_dir.name)
    
    print(f"🔥 Wildfire workspace created: {session_dir}")
    print(f"📁 Access via: {latest_link}")
    
    return session_dir


def get_workspace_paths(workspace_dir: pathlib.Path) -> dict:
    """
    Get all workspace subdirectory paths for easy access.
    
    Args:
        workspace_dir: Main workspace directory
        
    Returns:
        dict: Dictionary mapping subdirectory names to paths
    """
    return {
        "satellite_imagery": workspace_dir / "satellite_imagery",
        "analysis_results": workspace_dir / "analysis_results", 
        "maps_and_visualizations": workspace_dir / "maps_and_visualizations",
        "code_execution": workspace_dir / "code_execution",
        "documents": workspace_dir / "documents",
        "evacuation_plans": workspace_dir / "evacuation_plans",
        "risk_assessments": workspace_dir / "risk_assessments",
        "temp": workspace_dir / "temp",
        "root": workspace_dir
    }


def construct_society(question: str, workspace_dir: pathlib.Path = None) -> RolePlaying:
    """
    Construct a society of agents specialized for wildfire management tasks.

    Args:
        question (str): The wildfire-related task or question to be addressed by the society.
        workspace_dir (pathlib.Path, optional): Workspace directory for file operations. 
                                               If None, a new workspace will be created.

    Returns:
        RolePlaying: A configured society of agents ready to address wildfire management questions.
    """
    
    # Setup workspace if not provided
    if workspace_dir is None:
        workspace_dir = setup_wildfire_workspace()
    
    # Get workspace paths for different operations
    workspace_paths = get_workspace_paths(workspace_dir)
    
    # Create enhanced question with workspace context
    enhanced_question = f"""{question}

🔥 WILDFIRE AGENT WORKSPACE INSTRUCTIONS:
You are working in a dedicated wildfire analysis workspace. All file operations should use these directories:

📁 WORKSPACE STRUCTURE:
- Satellite imagery: {workspace_paths['satellite_imagery']}
- Analysis results: {workspace_paths['analysis_results']}
- Maps & visualizations: {workspace_paths['maps_and_visualizations']}
- Code execution: {workspace_paths['code_execution']}
- Documents & reports: {workspace_paths['documents']}
- Evacuation plans: {workspace_paths['evacuation_plans']}
- Risk assessments: {workspace_paths['risk_assessments']}
- Temporary files: {workspace_paths['temp']}

💡 IMPORTANT: Always save files to the appropriate workspace subdirectory. Provide full file paths in your responses so users can easily access generated content.

🤖 ADVANCED CAPABILITIES:
- YOLO Object Detection: Use analyze_wildfire_image() for automated fire, smoke, and infrastructure detection
- Vision Language Models: Analyze satellite imagery for detailed wildfire assessment
- Multi-agent Coordination: Leverage specialized agents for different analysis tasks
"""

    # Create models for different components - optimized for wildfire analysis
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_MAX,
            model_config_dict={"temperature": 0},
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_MAX,
            model_config_dict={"temperature": 0},
        ),
        "browsing": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_VL_MAX,  # Vision model for satellite imagery
            model_config_dict={"temperature": 0},
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_MAX,
            model_config_dict={"temperature": 0},
        ),
        "video": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_VL_MAX,  # For aerial footage analysis
            model_config_dict={"temperature": 0},
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_VL_MAX,  # Critical for satellite/drone imagery
            model_config_dict={"temperature": 0},
        ),
        "document": ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_VL_MAX,
            model_config_dict={"temperature": 0},
        ),
    }

    # Configure toolkits for wildfire management with workspace integration
    tools = [
        *BrowserToolkit(
            headless=False,  # Set to True for headless mode (e.g., on remote servers)
            web_agent_model=models["browsing"],
            planning_agent_model=models["planning"],
            output_language="English",  # Changed to English for wildfire management
        ).get_tools(),
        *VideoAnalysisToolkit(model=models["video"]).get_tools(),  # For aerial footage analysis
        *CodeExecutionToolkit(
            sandbox="subprocess", 
            verbose=True
        ).get_tools(),  # For GIS calculations and analysis scripts
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),  # Critical for satellite imagery
        SearchToolkit().search_duckduckgo,  # For real-time fire and weather data
        SearchToolkit().search_google,  # For weather and fire information
        SearchToolkit().search_wiki,  # For background information
        # Note: Future versions will include PostGIS, QGIS, GRASS GIS toolkits
        *ExcelToolkit().get_tools(),  # For population/infrastructure data analysis
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),  # For processing reports and documents
        *FileWriteToolkit(output_dir=str(workspace_paths['analysis_results'])).get_tools(),  # For analysis reports
        *FileWriteToolkit(output_dir=str(workspace_paths['maps_and_visualizations'])).get_tools(),  # For maps and charts
        *FileWriteToolkit(output_dir=str(workspace_paths['documents'])).get_tools(),  # For documentation
        *FileWriteToolkit(output_dir=str(workspace_paths['evacuation_plans'])).get_tools(),  # For evacuation planning
        *FileWriteToolkit(output_dir=str(workspace_paths['risk_assessments'])).get_tools(),  # For risk assessments
    ]
    
    # Add YOLO toolkit if available
    if YOLO_AVAILABLE:
        try:
            yolo_toolkit = WildfireYOLOToolkit(
                model_path="yolo11n.pt",  # Nano model for faster inference
                confidence_threshold=0.3,  # Lower threshold for wildfire detection
                output_dir=str(workspace_paths['satellite_imagery'])  # Save YOLO results to satellite imagery folder
            )
            tools.extend(yolo_toolkit.get_tools())
            print("🔥 YOLO wildfire detection toolkit integrated successfully!")
        except Exception as e:
            print(f"⚠️  Failed to initialize YOLO toolkit: {e}")
    else:
        print("⚠️  YOLO toolkit not available - wildfire object detection disabled")

    # Configure agent roles and parameters for wildfire management
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters with workspace-enhanced question
    task_kwargs = {
        "task_prompt": enhanced_question,
        "with_task_specify": False,
    }

    # Create and return the society specialized for wildfire management
    society = RolePlaying(
        **task_kwargs,
        user_role_name="Emergency Manager",  # Specialized role for wildfire context
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="Wildfire AI Agent",  # Specialized assistant role
        assistant_agent_kwargs=assistant_agent_kwargs,
        output_language="English",  # Changed to English for international compatibility
    )

    # Store workspace info in society for later access
    society.wildfire_workspace = workspace_dir
    society.workspace_paths = workspace_paths

    return society


def main():
    r"""Main function to run the Wildfire Agent system with an example question."""
    print("🔥 Starting Wildfire Agent System...")
    
    # Example wildfire management question with Maui satellite image analysis
    # Automatically find the Maui image path
    maui_image_path = find_maui_image()
    default_task = f"🖼️ Analyze the Maui wildfire satellite image at {maui_image_path}. Identify fire hotspots, assess burn areas, and evaluate risks to Lahaina community. Generate a comprehensive analysis report with evacuation recommendations."

    # Override default task if command line argument is provided
    task = sys.argv[1] if len(sys.argv) > 1 else default_task

    print(f"📋 Task: {task}")
    
    # Construct and run the society
    society = construct_society(task)
    
    print(f"📁 Workspace: {society.wildfire_workspace}")
    print(f"🔗 Quick access: {society.wildfire_workspace.parent / 'latest'}")
    
    answer, chat_history, token_count = run_society(society)

    # Output the result and workspace info
    print(f"\n🔥 WILDFIRE AGENT ANALYSIS COMPLETE 🔥")
    print(f"📋 Answer: {answer}")
    print(f"\n📁 ALL FILES SAVED TO WORKSPACE:")
    print(f"   📂 Main workspace: {society.wildfire_workspace}")
    print(f"   🔗 Quick access: {society.wildfire_workspace.parent / 'latest'}")
    print(f"\n📊 Workspace structure:")
    for name, path in society.workspace_paths.items():
        if name != 'root':
            print(f"   - {name}: {path}")
    
    print(f"\n💡 Check the workspace directories for generated files, maps, reports, and analysis results!")


if __name__ == "__main__":
    main()