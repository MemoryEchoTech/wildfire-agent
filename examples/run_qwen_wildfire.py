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

from camel.logger import set_log_level


import pathlib

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")


def construct_society(question: str) -> RolePlaying:
    """
    Construct a society of agents specialized for wildfire management tasks.

    Args:
        question (str): The wildfire-related task or question to be addressed by the society.

    Returns:
        RolePlaying: A configured society of agents ready to address wildfire management questions.
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

    # Configure toolkits for wildfire management
    tools = [
        *BrowserToolkit(
            headless=False,  # Set to True for headless mode (e.g., on remote servers)
            web_agent_model=models["browsing"],
            planning_agent_model=models["planning"],
            output_language="English",  # Changed to English for wildfire management
        ).get_tools(),
        *VideoAnalysisToolkit(model=models["video"]).get_tools(),  # For aerial footage
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),  # For GIS calculations
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),  # Critical for satellite imagery
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_google,  # For weather and fire data
        SearchToolkit().search_wiki,
        # Note: Future versions will include PostGIS, QGIS, GRASS GIS toolkits
        *ExcelToolkit().get_tools(),  # For population/infrastructure data
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),  # For reports and maps
    ]

    # Configure agent roles and parameters for wildfire management
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
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

    return society


def main():
    r"""Main function to run the Wildfire Agent system with an example question."""
    # Example wildfire management question
    default_task = "Analyze the current wildfire situation: How large is the area currently burning, what is the rate of spread, and which communities need immediate evacuation?"

    # Override default task if command line argument is provided
    task = sys.argv[1] if len(sys.argv) > 1 else default_task

    # Construct and run the society
    society = construct_society(task)
    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mWildfire Agent Answer: {answer}\033[0m")


if __name__ == "__main__":
    main()