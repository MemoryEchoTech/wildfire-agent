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
# Import from the correct module path
from utils import run_society
import os
import gradio as gr
import time
import json
import logging
import datetime
from typing import Tuple
import importlib
from dotenv import load_dotenv, set_key, find_dotenv, unset_key
import threading
import queue
import re

os.environ["PYTHONIOENCODING"] = "utf-8"


# Configure logging system
def setup_logging():
    """Configure logging system to output logs to file, memory queue, and console"""
    # Create logs directory (if it doesn't exist)
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Generate log filename (using current date)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = os.path.join(logs_dir, f"gradio_log_{current_date}.txt")

    # Configure root logger (captures all logs)
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("Logging system initialized, log file: %s", log_file)
    return log_file


# Global variables
LOG_FILE = None
LOG_QUEUE: queue.Queue = queue.Queue()  # Log queue
STOP_LOG_THREAD = threading.Event()
CURRENT_PROCESS = None  # Used to track the currently running process
STOP_REQUESTED = threading.Event()  # Used to mark if stop was requested


# Log reading and updating functions
def log_reader_thread(log_file):
    """Background thread that continuously reads the log file and adds new lines to the queue"""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            # Move to the end of file
            f.seek(0, 2)

            while not STOP_LOG_THREAD.is_set():
                line = f.readline()
                if line:
                    LOG_QUEUE.put(line)  # Add to conversation record queue
                else:
                    # No new lines, wait for a short time
                    time.sleep(0.1)
    except Exception as e:
        logging.error(f"Log reader thread error: {str(e)}")


def get_workspace_images():
    """Get all images from the current workspace for Gradio Gallery
    
    Returns:
        list: List of image file paths for Gradio Gallery component
    """
    import os
    import glob
    from datetime import datetime
    
    image_paths = []
    
    # Define workspace paths to search
    workspace_paths = [
        "workspace",
        "wildfire_workspace/latest",
        "wildfire_workspace/latest/satellite_imagery",
        "wildfire_workspace/latest/maps_and_visualizations",
        "wildfire_workspace/latest/analysis_results",
    ]
    
    # Also search all recent workspace sessions
    workspace_root = "wildfire_workspace"
    if os.path.exists(workspace_root):
        # Get all session directories
        session_dirs = glob.glob(os.path.join(workspace_root, "session_*"))
        for session_dir in sorted(session_dirs, reverse=True)[:5]:  # Last 5 sessions
            workspace_paths.extend([
                os.path.join(session_dir, "satellite_imagery"),
                os.path.join(session_dir, "maps_and_visualizations"),
                os.path.join(session_dir, "analysis_results"),
            ])
    
    # Image extensions to search for
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    
    images_with_time = []
    
    for workspace_path in workspace_paths:
        if not os.path.exists(workspace_path):
            continue
            
        for ext in image_extensions:
            pattern = os.path.join(workspace_path, ext)
            found_images = glob.glob(pattern)
            
            for img_path in found_images:
                try:
                    # Get file stats for sorting
                    stat = os.stat(img_path)
                    modified_time = datetime.fromtimestamp(stat.st_mtime)
                    
                    # Use absolute path for Gradio Gallery
                    abs_path = os.path.abspath(img_path)
                    images_with_time.append((abs_path, modified_time))
                except Exception as e:
                    continue
    
    # Sort by modification time (newest first) and return just the paths
    images_with_time.sort(key=lambda x: x[1], reverse=True)
    image_paths = [img[0] for img in images_with_time]
    
    return image_paths


def format_image_gallery() -> str:
    """Create formatted gallery display of workspace images
    
    Returns:
        str: HTML/Markdown formatted gallery
    """
    images = get_workspace_images()
    
    if not images:
        return """
# 📸 Image Gallery

No images found in workspace yet. Run an analysis to generate images!

**Tip:** The gallery will show:
- 🛰️ Input satellite imagery
- 🎯 YOLO detection results  
- 📊 Analysis visualizations
- 🗺️ Generated maps and charts
"""
    
    # Group images by type
    image_groups = {
        'satellite_input': [],
        'yolo_output': [],
        'analysis_output': [],
        'map_visualization': [],
        'general': []
    }
    
    for img in images:
        image_groups[img['type']].append(img)
    
    # Generate gallery HTML
    gallery_html = "# 📸 Workspace Image Gallery\n\n"
    gallery_html += f"**Found {len(images)} images** • Last updated: {images[0]['modified'].strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Type icons and names
    type_info = {
        'satellite_input': ('🛰️', 'Satellite Imagery'),
        'yolo_output': ('🎯', 'YOLO Detection Results'),
        'analysis_output': ('📊', 'Analysis Results'),
        'map_visualization': ('🗺️', 'Maps & Visualizations'),
        'general': ('📁', 'General Images')
    }
    
    for img_type, (icon, title) in type_info.items():
        type_images = image_groups[img_type]
        if not type_images:
            continue
            
        gallery_html += f"## {icon} {title} ({len(type_images)})\n\n"
        
        for img in type_images:
            # Format file size
            size_mb = img['size'] / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{img['size'] / 1024:.0f} KB"
            
            gallery_html += f"### 📷 {img['filename']}\n\n"
            
            # Try to display the image using Gradio's file serving
            try:
                gallery_html += f"<img src='/file={img['full_path']}' alt='{img['filename']}' style='max-width: 400px; max-height: 300px; border-radius: 8px; margin: 10px 0;'>\n\n"
            except Exception:
                gallery_html += f"*Image could not be displayed*\n\n"
            
            gallery_html += f"**Details:**\n"
            gallery_html += f"- 📁 Path: `{img['path']}`\n"
            gallery_html += f"- 📏 Size: {size_str}\n"
            gallery_html += f"- 🕒 Modified: {img['modified'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            gallery_html += f"- 📂 Workspace: `{img['workspace']}`\n\n"
            gallery_html += "---\n\n"
    
    return gallery_html


def get_latest_logs(max_lines=100, queue_source=None):
    """Get the latest log lines from the queue, or read directly from the file if the queue is empty

    Args:
        max_lines: Maximum number of lines to return
        queue_source: Specify which queue to use, default is LOG_QUEUE

    Returns:
        str: Log content
    """
    logs = []
    log_queue = queue_source if queue_source else LOG_QUEUE

    # Create a temporary queue to store logs so we can process them without removing them from the original queue
    temp_queue = queue.Queue()
    temp_logs = []

    try:
        # Try to get all available log lines from the queue
        while not log_queue.empty() and len(temp_logs) < max_lines:
            log = log_queue.get_nowait()
            temp_logs.append(log)
            temp_queue.put(log)  # Put the log back into the temporary queue
    except queue.Empty:
        pass

    # Process conversation records
    logs = temp_logs

    # If there are no new logs or not enough logs, try to read the last few lines directly from the file
    if len(logs) < max_lines and LOG_FILE and os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                all_lines = f.readlines()
                # If there are already some logs in the queue, only read the remaining needed lines
                remaining_lines = max_lines - len(logs)
                file_logs = (
                    all_lines[-remaining_lines:]
                    if len(all_lines) > remaining_lines
                    else all_lines
                )

                # Add file logs before queue logs
                logs = file_logs + logs
        except Exception as e:
            error_msg = f"Error reading log file: {str(e)}"
            logging.error(error_msg)
            if not logs:  # Only add error message if there are no logs
                logs = [error_msg]

    # If there are still no logs, return a prompt message
    if not logs:
        return "Initialization in progress..."

    # Filter logs, only keep logs with 'camel.agents.chat_agent - INFO'
    filtered_logs = []
    for log in logs:
        if "camel.agents.chat_agent - INFO" in log:
            filtered_logs.append(log)

    # If there are no logs after filtering, return a prompt message
    if not filtered_logs:
        return "No conversation records yet."

    # Process log content, extract the latest user and assistant messages
    simplified_logs = []

    # Use a set to track messages that have already been processed, to avoid duplicates
    processed_messages = set()

    def process_message(role, content):
        # Create a unique identifier to track messages
        msg_id = f"{role}:{content}"
        if msg_id in processed_messages:
            return None

        processed_messages.add(msg_id)
        content = content.replace("\\n", "\n")
        lines = [line.strip() for line in content.split("\n")]
        content = "\n".join(lines)

        role_emoji = "🙋" if role.lower() == "user" else "🤖"
        return f"""### {role_emoji} {role.title()} Agent

{content}"""

    for log in filtered_logs:
        formatted_messages = []
        # Try to extract message array
        messages_match = re.search(
            r"Model (.*?), index (\d+), processed these messages: (\[.*\])", log
        )

        if messages_match:
            try:
                messages = json.loads(messages_match.group(3))
                for msg in messages:
                    if msg.get("role") in ["user", "assistant"]:
                        formatted_msg = process_message(
                            msg.get("role"), msg.get("content", "")
                        )
                        if formatted_msg:
                            formatted_messages.append(formatted_msg)
            except json.JSONDecodeError:
                pass

        # If JSON parsing fails or no message array is found, try to extract conversation content directly
        if not formatted_messages:
            user_pattern = re.compile(r"\{'role': 'user', 'content': '(.*?)'\}")
            assistant_pattern = re.compile(
                r"\{'role': 'assistant', 'content': '(.*?)'\}"
            )

            for content in user_pattern.findall(log):
                formatted_msg = process_message("user", content)
                if formatted_msg:
                    formatted_messages.append(formatted_msg)

            for content in assistant_pattern.findall(log):
                formatted_msg = process_message("assistant", content)
                if formatted_msg:
                    formatted_messages.append(formatted_msg)

        if formatted_messages:
            simplified_logs.append("\n\n".join(formatted_messages))

    # Format log output, ensure appropriate separation between each conversation record
    formatted_logs = []
    for i, log in enumerate(simplified_logs):
        # Remove excess whitespace characters from beginning and end
        log = log.strip()

        formatted_logs.append(log)

        # Ensure each conversation record ends with a newline
        if not log.endswith("\n"):
            formatted_logs.append("\n")

    return "\n".join(formatted_logs)


# Dictionary containing module descriptions - Wildfire Agent Configuration
MODULE_DESCRIPTIONS = {
    "run_qwen_wildfire": "🔥 Wildfire Agent: Specialized Qwen VL model for wildfire management, satellite imagery analysis, and emergency response planning",
}


# Default environment variable template
DEFAULT_ENV_TEMPLATE = """#===========================================
# MODEL & API 
# (See https://docs.camel-ai.org/key_modules/models.html#)
#===========================================

# OPENAI API (https://platform.openai.com/api-keys)
OPENAI_API_KEY='Your_Key'
# OPENAI_API_BASE_URL=""

# Azure OpenAI API
# AZURE_OPENAI_BASE_URL=""
# AZURE_API_VERSION=""
# AZURE_OPENAI_API_KEY=""
# AZURE_DEPLOYMENT_NAME=""


# Qwen API (https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key)
QWEN_API_KEY='Your_Key'

# DeepSeek API (https://platform.deepseek.com/api_keys)
DEEPSEEK_API_KEY='Your_Key'

#===========================================
# Tools & Services API
#===========================================

# Google Search API (https://coda.io/@jon-dallas/google-image-search-pack-example/search-engine-id-and-google-api-key-3)
GOOGLE_API_KEY='Your_Key'
SEARCH_ENGINE_ID='Your_ID'

# Chunkr API (https://chunkr.ai/)
CHUNKR_API_KEY='Your_Key'

# Firecrawl API (https://www.firecrawl.dev/)
FIRECRAWL_API_KEY='Your_Key'
#FIRECRAWL_API_URL="https://api.firecrawl.dev"
"""


def validate_input(question: str) -> bool:
    """Validate if user input is valid

    Args:
        question: User question

    Returns:
        bool: Whether the input is valid
    """
    # Check if input is empty or contains only spaces
    if not question or question.strip() == "":
        return False
    return True


def extract_agent_summary(chat_history: list, society=None) -> str:
    """Extract a summary of what the agent accomplished from chat history
    
    Args:
        chat_history: List of chat messages from the agent session
        society: Society object with workspace information
        
    Returns:
        str: Formatted summary of agent actions
    """
    if not chat_history:
        return "No actions recorded in this session."
    
    summary_parts = []
    summary_parts.append("# 🔥 Agent Session Summary\n")
    
    # Add workspace information if available
    if society and hasattr(society, 'wildfire_workspace'):
        workspace_path = society.wildfire_workspace
        summary_parts.append(f"**📁 Workspace:** `{workspace_path}`\n")
        summary_parts.append(f"**🔗 Quick Access:** `{workspace_path.parent / 'latest'}`\n\n")
    
    # Track different types of actions
    actions = {
        'files_created': [],
        'images_analyzed': [],
        'tools_used': [],
        'analyses_performed': [],
        'errors_encountered': []
    }
    
    # Parse chat history for actions
    for message in chat_history:
        content = str(message).lower()
        
        # Track file operations
        if 'saved' in content or 'created' in content or 'generated' in content:
            if '.jpg' in content or '.png' in content:
                actions['files_created'].append('🖼️ Image file')
            elif '.json' in content:
                actions['files_created'].append('📄 JSON results')
            elif '.docx' in content or '.pdf' in content:
                actions['files_created'].append('📋 Report document')
            elif 'file' in content:
                actions['files_created'].append('📁 Analysis file')
        
        # Track image analysis
        if 'analyze' in content and ('image' in content or 'satellite' in content):
            actions['images_analyzed'].append('🛰️ Satellite imagery')
        if 'maui' in content and 'wildfire' in content:
            actions['images_analyzed'].append('🔥 Maui wildfire satellite image')
        
        # Track tool usage
        if 'yolo' in content:
            actions['tools_used'].append('🎯 YOLO object detection')
        if 'code execution' in content or 'python' in content:
            actions['tools_used'].append('💻 Code execution')
        if 'search' in content:
            actions['tools_used'].append('🔍 Web search')
        if 'browser' in content:
            actions['tools_used'].append('🌐 Web browsing')
        
        # Track analyses
        if 'risk assessment' in content:
            actions['analyses_performed'].append('⚠️ Risk assessment')
        if 'evacuation' in content:
            actions['analyses_performed'].append('🚨 Evacuation planning')
        if 'fire detection' in content or 'hotspot' in content:
            actions['analyses_performed'].append('🔥 Fire detection')
        if 'infrastructure' in content:
            actions['analyses_performed'].append('🏗️ Infrastructure analysis')
        
        # Track errors
        if 'error' in content or 'failed' in content:
            actions['errors_encountered'].append('❌ Processing error')
    
    # Build summary sections
    if actions['images_analyzed']:
        summary_parts.append("## 🖼️ Images Analyzed\n")
        for item in set(actions['images_analyzed']):
            summary_parts.append(f"- {item}\n")
        summary_parts.append("\n")
    
    if actions['tools_used']:
        summary_parts.append("## 🛠️ Tools Used\n")
        for item in set(actions['tools_used']):
            summary_parts.append(f"- {item}\n")
        summary_parts.append("\n")
    
    if actions['analyses_performed']:
        summary_parts.append("## 📊 Analyses Performed\n")
        for item in set(actions['analyses_performed']):
            summary_parts.append(f"- {item}\n")
        summary_parts.append("\n")
    
    if actions['files_created']:
        summary_parts.append("## 📁 Files Created\n")
        for item in set(actions['files_created']):
            summary_parts.append(f"- {item}\n")
        summary_parts.append("\n")
    
    if actions['errors_encountered']:
        summary_parts.append("## ⚠️ Issues Encountered\n")
        for item in set(actions['errors_encountered']):
            summary_parts.append(f"- {item}\n")
        summary_parts.append("\n")
    
    # Add session stats
    summary_parts.append("## 📈 Session Statistics\n")
    summary_parts.append(f"- **Total Messages:** {len(chat_history)}\n")
    summary_parts.append(f"- **Tools Used:** {len(set(actions['tools_used']))}\n")
    summary_parts.append(f"- **Files Generated:** {len(set(actions['files_created']))}\n")
    summary_parts.append(f"- **Analyses Completed:** {len(set(actions['analyses_performed']))}\n")
    
    return "".join(summary_parts)


def run_owl(question: str, example_module: str) -> Tuple[str, str, str, str]:
    """Run the OWL system and return results

    Args:
        question: User question
        example_module: Example module name to import (e.g., "run_terminal_zh" or "run_deep")

    Returns:
        Tuple[str, str, str, str]: Answer, token count, status, summary
    """
    global CURRENT_PROCESS

    # Validate input
    if not validate_input(question):
        logging.warning("User submitted invalid input")
        return (
            "Please enter a valid question",
            "0",
            "❌ Error: Invalid input question",
            "No summary available due to invalid input.",
        )

    try:
        # Ensure environment variables are loaded
        load_dotenv(find_dotenv(), override=True)
        logging.info(
            f"Processing question: '{question}', using module: {example_module}"
        )

        # Check if the module is in MODULE_DESCRIPTIONS
        if example_module not in MODULE_DESCRIPTIONS:
            logging.error(f"User selected an unsupported module: {example_module}")
            return (
                f"Selected module '{example_module}' is not supported",
                "0",
                "❌ Error: Unsupported module",
                "No summary available due to unsupported module.",
            )

        # Dynamically import target module
        module_path = f"examples.{example_module}"
        try:
            logging.info(f"Importing module: {module_path}")
            module = importlib.import_module(module_path)
        except ImportError as ie:
            logging.error(f"Unable to import module {module_path}: {str(ie)}")
            return (
                f"Unable to import module: {module_path}",
                "0",
                f"❌ Error: Module {example_module} does not exist or cannot be loaded - {str(ie)}",
                "No summary available due to import error.",
            )
        except Exception as e:
            logging.error(
                f"Error occurred while importing module {module_path}: {str(e)}"
            )
            return (
                f"Error occurred while importing module: {module_path}",
                "0",
                f"❌ Error: {str(e)}",
                "No summary available due to module error.",
            )

        # Check if it contains the construct_society function
        if not hasattr(module, "construct_society"):
            logging.error(
                f"construct_society function not found in module {module_path}"
            )
            return (
                f"construct_society function not found in module {module_path}",
                "0",
                "❌ Error: Module interface incompatible",
                "No summary available due to interface error.",
            )

        # Build society simulation
        try:
            logging.info("Building society simulation...")
            society = module.construct_society(question)

        except Exception as e:
            logging.error(f"Error occurred while building society simulation: {str(e)}")
            return (
                f"Error occurred while building society simulation: {str(e)}",
                "0",
                f"❌ Error: Build failed - {str(e)}",
                "No summary available due to build error.",
            )

        # Run society simulation
        try:
            logging.info("Running society simulation...")
            answer, chat_history, token_info = run_society(society)
            logging.info("Society simulation completed")
        except Exception as e:
            logging.error(f"Error occurred while running society simulation: {str(e)}")
            return (
                f"Error occurred while running society simulation: {str(e)}",
                "0",
                f"❌ Error: Run failed - {str(e)}",
                "No summary available due to runtime error.",
            )

        # Safely get token count
        if not isinstance(token_info, dict):
            token_info = {}

        completion_tokens = token_info.get("completion_token_count", 0)
        prompt_tokens = token_info.get("prompt_token_count", 0)
        total_tokens = completion_tokens + prompt_tokens

        logging.info(
            f"Processing completed, token usage: completion={completion_tokens}, prompt={prompt_tokens}, total={total_tokens}"
        )

        # Generate agent summary
        agent_summary = extract_agent_summary(chat_history, society)
        
        return (
            answer,
            f"Completion tokens: {completion_tokens:,} | Prompt tokens: {prompt_tokens:,} | Total: {total_tokens:,}",
            "✅ Successfully completed",
            agent_summary,
        )

    except Exception as e:
        logging.error(
            f"Uncaught error occurred while processing the question: {str(e)}"
        )
        return (f"Error occurred: {str(e)}", "0", f"❌ Error: {str(e)}", "No summary available due to uncaught error.")


def update_module_description(module_name: str) -> str:
    """Return the description of the selected module"""
    return MODULE_DESCRIPTIONS.get(module_name, "No description available")


# Store environment variables configured from the frontend
WEB_FRONTEND_ENV_VARS: dict[str, str] = {}


def init_env_file():
    """Initialize .env file if it doesn't exist"""
    dotenv_path = find_dotenv()
    if not dotenv_path:
        with open(".env", "w") as f:
            f.write(DEFAULT_ENV_TEMPLATE)
        dotenv_path = find_dotenv()
    return dotenv_path


def load_env_vars():
    """Load environment variables and return as dictionary format

    Returns:
        dict: Environment variable dictionary, each value is a tuple containing value and source (value, source)
    """
    dotenv_path = init_env_file()
    load_dotenv(dotenv_path, override=True)

    # Read environment variables from .env file
    env_file_vars = {}
    with open(dotenv_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_file_vars[key.strip()] = value.strip().strip("\"'")

    # Get from system environment variables
    system_env_vars = {
        k: v
        for k, v in os.environ.items()
        if k not in env_file_vars and k not in WEB_FRONTEND_ENV_VARS
    }

    # Merge environment variables and mark sources
    env_vars = {}

    # Add system environment variables (lowest priority)
    for key, value in system_env_vars.items():
        env_vars[key] = (value, "System")

    # Add .env file environment variables (medium priority)
    for key, value in env_file_vars.items():
        env_vars[key] = (value, ".env file")

    # Add frontend configured environment variables (highest priority)
    for key, value in WEB_FRONTEND_ENV_VARS.items():
        env_vars[key] = (value, "Frontend configuration")
        # Ensure operating system environment variables are also updated
        os.environ[key] = value

    return env_vars


def save_env_vars(env_vars):
    """Save environment variables to .env file

    Args:
        env_vars: Dictionary, keys are environment variable names, values can be strings or (value, source) tuples
    """
    try:
        dotenv_path = init_env_file()

        # Save each environment variable
        for key, value_data in env_vars.items():
            if key and key.strip():  # Ensure key is not empty
                # Handle case where value might be a tuple
                if isinstance(value_data, tuple):
                    value = value_data[0]
                else:
                    value = value_data

                set_key(dotenv_path, key.strip(), value.strip())

        # Reload environment variables to ensure they take effect
        load_dotenv(dotenv_path, override=True)

        return True, "Environment variables have been successfully saved!"
    except Exception as e:
        return False, f"Error saving environment variables: {str(e)}"


def add_env_var(key, value, from_frontend=True):
    """Add or update a single environment variable

    Args:
        key: Environment variable name
        value: Environment variable value
        from_frontend: Whether it's from frontend configuration, default is True
    """
    try:
        if not key or not key.strip():
            return False, "Variable name cannot be empty"

        key = key.strip()
        value = value.strip()

        # If from frontend, add to frontend environment variable dictionary
        if from_frontend:
            WEB_FRONTEND_ENV_VARS[key] = value
            # Directly update system environment variables
            os.environ[key] = value

        # Also update .env file
        dotenv_path = init_env_file()
        set_key(dotenv_path, key, value)
        load_dotenv(dotenv_path, override=True)

        return True, f"Environment variable {key} has been successfully added/updated!"
    except Exception as e:
        return False, f"Error adding environment variable: {str(e)}"


def delete_env_var(key):
    """Delete environment variable"""
    try:
        if not key or not key.strip():
            return False, "Variable name cannot be empty"

        key = key.strip()

        # Delete from .env file
        dotenv_path = init_env_file()
        unset_key(dotenv_path, key)

        # Delete from frontend environment variable dictionary
        if key in WEB_FRONTEND_ENV_VARS:
            del WEB_FRONTEND_ENV_VARS[key]

        # Also delete from current process environment
        if key in os.environ:
            del os.environ[key]

        return True, f"Environment variable {key} has been successfully deleted!"
    except Exception as e:
        return False, f"Error deleting environment variable: {str(e)}"


def is_api_related(key: str) -> bool:
    """Determine if an environment variable is API-related

    Args:
        key: Environment variable name

    Returns:
        bool: Whether it's API-related
    """
    # API-related keywords
    api_keywords = [
        "api",
        "key",
        "token",
        "secret",
        "password",
        "openai",
        "qwen",
        "deepseek",
        "google",
        "search",
        "hf",
        "hugging",
        "chunkr",
        "firecrawl",
    ]

    # Check if it contains API-related keywords (case insensitive)
    return any(keyword in key.lower() for keyword in api_keywords)


def get_api_guide(key: str) -> str:
    """Return the corresponding API guide based on the environment variable name

    Args:
        key: Environment variable name

    Returns:
        str: API guide link or description
    """
    key_lower = key.lower()
    if "openai" in key_lower:
        return "https://platform.openai.com/api-keys"
    elif "qwen" in key_lower or "dashscope" in key_lower:
        return "https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key"
    elif "deepseek" in key_lower:
        return "https://platform.deepseek.com/api_keys"
    elif "ppio" in key_lower:
        return "https://ppinfra.com/settings/key-management?utm_source=github_owl"
    elif "google" in key_lower:
        return "https://coda.io/@jon-dallas/google-image-search-pack-example/search-engine-id-and-google-api-key-3"
    elif "search_engine_id" in key_lower:
        return "https://coda.io/@jon-dallas/google-image-search-pack-example/search-engine-id-and-google-api-key-3"
    elif "chunkr" in key_lower:
        return "https://chunkr.ai/"
    elif "firecrawl" in key_lower:
        return "https://www.firecrawl.dev/"
    elif "novita" in key_lower:
        return "https://novita.ai/settings/key-management?utm_source=github_owl&utm_medium=github_readme&utm_campaign=github_link"
    else:
        return ""


def update_env_table():
    """Update environment variable table display, only showing API-related environment variables"""
    env_vars = load_env_vars()
    # Filter out API-related environment variables
    api_env_vars = {k: v for k, v in env_vars.items() if is_api_related(k)}
    # Convert to list format to meet Gradio Dataframe requirements
    # Format: [Variable name, Variable value, Guide link]
    result = []
    for k, v in api_env_vars.items():
        guide = get_api_guide(k)
        # If there's a guide link, create a clickable link
        guide_link = (
            f"<a href='{guide}' target='_blank' class='guide-link'>🔗 Get</a>"
            if guide
            else ""
        )
        result.append([k, v[0], guide_link])
    return result


def save_env_table_changes(data):
    """Save changes to the environment variable table

    Args:
        data: Dataframe data, possibly a pandas DataFrame object

    Returns:
        str: Operation status information, containing HTML-formatted status message
    """
    try:
        logging.info(
            f"Starting to process environment variable table data, type: {type(data)}"
        )

        # Get all current environment variables
        current_env_vars = load_env_vars()
        processed_keys = set()  # Record processed keys to detect deleted variables

        # Process pandas DataFrame object
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            # Get column name information
            columns = data.columns.tolist()
            logging.info(f"DataFrame column names: {columns}")

            # Iterate through each row of the DataFrame
            for index, row in data.iterrows():
                # Use column names to access data
                if len(columns) >= 3:
                    # Get variable name and value (column 0 is name, column 1 is value)
                    key = row[0] if isinstance(row, pd.Series) else row.iloc[0]
                    value = row[1] if isinstance(row, pd.Series) else row.iloc[1]

                    # Check if it's an empty row or deleted variable
                    if (
                        key and str(key).strip()
                    ):  # If key name is not empty, add or update
                        logging.info(
                            f"Processing environment variable: {key} = {value}"
                        )
                        add_env_var(key, str(value))
                        processed_keys.add(key)
        # Process other formats
        elif isinstance(data, dict):
            logging.info(f"Dictionary format data keys: {list(data.keys())}")
            # If dictionary format, try different keys
            if "data" in data:
                rows = data["data"]
            elif "values" in data:
                rows = data["values"]
            elif "value" in data:
                rows = data["value"]
            else:
                # Try using dictionary directly as row data
                rows = []
                for key, value in data.items():
                    if key not in ["headers", "types", "columns"]:
                        rows.append([key, value])

            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, list) and len(row) >= 2:
                        key, value = row[0], row[1]
                        if key and str(key).strip():
                            add_env_var(key, str(value))
                            processed_keys.add(key)
        elif isinstance(data, list):
            # 列表格式
            for row in data:
                if isinstance(row, list) and len(row) >= 2:
                    key, value = row[0], row[1]
                    if key and str(key).strip():
                        add_env_var(key, str(value))
                        processed_keys.add(key)
        else:
            logging.error(f"Unknown data format: {type(data)}")
            return f"❌ Save failed: Unknown data format {type(data)}"

        # Process deleted variables - check if there are variables in current environment not appearing in the table
        api_related_keys = {k for k in current_env_vars.keys() if is_api_related(k)}
        keys_to_delete = api_related_keys - processed_keys

        # Delete variables no longer in the table
        for key in keys_to_delete:
            logging.info(f"Deleting environment variable: {key}")
            delete_env_var(key)

        return "✅ Environment variables have been successfully saved"
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logging.error(f"Error saving environment variables: {str(e)}\n{error_details}")
        return f"❌ Save failed: {str(e)}"


def get_env_var_value(key):
    """Get the actual value of an environment variable

    Priority: Frontend configuration > .env file > System environment variables
    """
    # Check frontend configured environment variables
    if key in WEB_FRONTEND_ENV_VARS:
        return WEB_FRONTEND_ENV_VARS[key]

    # Check system environment variables (including those loaded from .env)
    return os.environ.get(key, "")


def create_ui():
    """Create enhanced Gradio interface"""

    def clear_log_file():
        """Clear log file content"""
        try:
            if LOG_FILE and os.path.exists(LOG_FILE):
                # Clear log file content instead of deleting the file
                open(LOG_FILE, "w").close()
                logging.info("Log file has been cleared")
                # Clear log queue
                while not LOG_QUEUE.empty():
                    try:
                        LOG_QUEUE.get_nowait()
                    except queue.Empty:
                        break
                return ""
            else:
                return ""
        except Exception as e:
            logging.error(f"Error clearing log file: {str(e)}")
            return ""

    # Create a real-time log update function
    def process_with_live_logs(question, module_name):
        """Process questions and update logs in real-time"""
        global CURRENT_PROCESS

        # Clear log file
        clear_log_file()

        # Create a background thread to process the question
        result_queue = queue.Queue()

        def process_in_background():
            try:
                result = run_owl(question, module_name)
                result_queue.put(result)
            except Exception as e:
                result_queue.put(
                    (f"Error occurred: {str(e)}", "0", f"❌ Error: {str(e)}", "No summary available due to error.")
                )

        # Start background processing thread
        bg_thread = threading.Thread(target=process_in_background)
        CURRENT_PROCESS = bg_thread  # Record current process
        bg_thread.start()

        # While waiting for processing to complete, update logs once per second
        while bg_thread.is_alive():
            # Update conversation record display
            logs2 = get_latest_logs(100, LOG_QUEUE)

            # Always update status
            yield (
                "0",
                "<span class='status-indicator status-running'></span> Processing...",
                logs2,
                "Agent is currently processing your request...",
            )

            time.sleep(1)

        # Processing complete, get results
        if not result_queue.empty():
            result = result_queue.get()
            answer, token_count, status, summary = result

            # Final update of conversation record
            logs2 = get_latest_logs(100, LOG_QUEUE)

            # Set different indicators based on status
            if "Error" in status:
                status_with_indicator = (
                    f"<span class='status-indicator status-error'></span> {status}"
                )
            else:
                status_with_indicator = (
                    f"<span class='status-indicator status-success'></span> {status}"
                )

            yield token_count, status_with_indicator, logs2, summary
        else:
            logs2 = get_latest_logs(100, LOG_QUEUE)
            yield (
                "0",
                "<span class='status-indicator status-error'></span> Terminated",
                logs2,
                "Session was terminated before completion.",
            )

    with gr.Blocks(title="🔥 Wildfire Agent", theme=gr.themes.Soft(primary_hue="red")) as app:
        gr.Markdown(
            """
                # 🔥 Wildfire Agent - AI Emergency Management System

                Specialized AI system for wildfire disaster management, emergency response, and geospatial analysis. Built on the proven OWL/CAMEL framework with integrated remote sensing and GIS capabilities.

                **Key Features:** Satellite imagery analysis • Fire spread modeling • Evacuation planning • Risk assessment • Infrastructure protection
                
                This specialized wildfire management system uses advanced multi-agent collaboration to provide emergency managers with real-time analysis and response planning.
                """
        )

        # Add custom CSS
        gr.HTML("""
            <style>
            /* Chat container style */
            .chat-container .chatbot {
                height: 500px;
                overflow-y: auto;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }
            

            /* Improved tab style */
            .tabs .tab-nav {
                background-color: #f5f5f5;
                border-radius: 8px 8px 0 0;
                padding: 5px;
            }
            
            .tabs .tab-nav button {
                border-radius: 5px;
                margin: 0 3px;
                padding: 8px 15px;
                font-weight: 500;
            }
            
            .tabs .tab-nav button.selected {
                background-color: #2c7be5;
                color: white;
            }
            
            /* Status indicator style */
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 5px;
            }
            
            .status-running {
                background-color: #ffc107;
                animation: pulse 1.5s infinite;
            }
            
            .status-success {
                background-color: #28a745;
            }
            
            .status-error {
                background-color: #dc3545;
            }
            
            /* Log display area style */
            .log-display {
                height: 500px !important;
                max-height: 500px !important;
                overflow-y: auto !important;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
                background-color: #f8f9fa;
                font-family: monospace;
                font-size: 0.9em;
                white-space: pre-wrap;
                line-height: 1.4;
                word-wrap: break-word;
            }
            
            .log-display pre {
                margin: 0;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            
            .log-display code {
                background-color: transparent;
                padding: 0;
                color: inherit;
            }
            
            /* Fix scrolling for Gradio Markdown component in conversation record */
            .log-display .prose {
                height: 450px !important;
                max-height: 450px !important;
                overflow-y: auto !important;
                padding-right: 10px;
            }
            
            /* Custom scrollbar styling */
            .log-display .prose::-webkit-scrollbar {
                width: 8px;
            }
            
            .log-display .prose::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 4px;
            }
            
            .log-display .prose::-webkit-scrollbar-thumb {
                background: #c1c1c1;
                border-radius: 4px;
            }
            
            .log-display .prose::-webkit-scrollbar-thumb:hover {
                background: #a8a8a8;
            }
            
            /* Ensure conversation text wraps properly */
            .log-display .prose p, .log-display .prose div {
                word-wrap: break-word;
                overflow-wrap: break-word;
                margin-bottom: 10px;
            }
            
            /* Environment variable management style */
            .env-manager-container {
                border-radius: 10px;
                padding: 15px;
                background-color: #f9f9f9;
                margin-bottom: 20px;
            }
            
            .env-controls, .api-help-container {
                border-radius: 8px;
                padding: 15px;
                background-color: white;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
                height: 100%;
            }
            
            .env-add-group, .env-delete-group {
                margin-top: 20px;
                padding: 15px;
                border-radius: 8px;
                background-color: #f5f8ff;
                border: 1px solid #e0e8ff;
            }
            
            .env-delete-group {
                background-color: #fff5f5;
                border: 1px solid #ffe0e0;
            }
            
            .env-buttons {
                justify-content: flex-start;
                gap: 10px;
                margin-top: 10px;
            }
            
            .env-button {
                min-width: 100px;
            }
            
            .delete-button {
                background-color: #dc3545;
                color: white;
            }
            
            .env-table {
                margin-bottom: 15px;
            }
            
            /* Improved environment variable table style */
            .env-table table {
                border-collapse: separate;
                border-spacing: 0;
                width: 100%;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            
            .env-table th {
                background-color: #f0f7ff;
                padding: 12px 15px;
                text-align: left;
                font-weight: 600;
                color: #2c7be5;
                border-bottom: 2px solid #e0e8ff;
            }
            
            .env-table td {
                padding: 10px 15px;
                border-bottom: 1px solid #f0f0f0;
            }
            
            .env-table tr:hover td {
                background-color: #f9fbff;
            }
            
            .env-table tr:last-child td {
                border-bottom: none;
            }
            
            /* Status icon style */
            .status-icon-cell {
                text-align: center;
                font-size: 1.2em;
            }
            
            /* Link style */
            .guide-link {
                color: #2c7be5;
                text-decoration: none;
                cursor: pointer;
                font-weight: 500;
            }
            
            .guide-link:hover {
                text-decoration: underline;
            }
            
            .env-status {
                margin-top: 15px;
                font-weight: 500;
                padding: 10px;
                border-radius: 6px;
                transition: all 0.3s ease;
            }
            
            .env-status-success {
                background-color: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }
            
            .env-status-error {
                background-color: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }
            
            .api-help-accordion {
                margin-bottom: 8px;
                border-radius: 6px;
                overflow: hidden;
            }
            

            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            </style>
            """)

        with gr.Row():
            with gr.Column(scale=0.5):
                question_input = gr.Textbox(
                    lines=5,
                    placeholder="Enter your wildfire management question here...",
                    label="🔥 Wildfire Management Query",
                    elem_id="question_input",
                    show_copy_button=True,
                    value="🖼️ Analyze the Maui wildfire satellite image at /Users/kang/GitHub/wildfire-agent/Maui Wildfires Image.jpg. Identify fire hotspots, assess burn areas, and evaluate risks to Lahaina community.",
                )

                # Enhanced module selection dropdown
                # Only includes modules defined in MODULE_DESCRIPTIONS
                module_dropdown = gr.Dropdown(
                    choices=list(MODULE_DESCRIPTIONS.keys()),
                    value="run_qwen_wildfire",
                    label="🔥 Wildfire Agent Module",
                    interactive=False,  # Make it non-interactive since we only have one module
                )

                # Module description text box
                module_description = gr.Textbox(
                    value=MODULE_DESCRIPTIONS["run_qwen_wildfire"],
                    label="Module Description",
                    interactive=False,
                    elem_classes="module-info",
                )

                with gr.Row():
                    run_button = gr.Button(
                        "Run", variant="primary", elem_classes="primary"
                    )

                status_output = gr.HTML(
                    value="<span class='status-indicator status-success'></span> Ready",
                    label="Status",
                )
                token_count_output = gr.Textbox(
                    label="Token Count", interactive=False, elem_classes="token-count"
                )

                # Example questions
                examples = [
                    "🖼️ Analyze the Maui wildfire satellite image at /Users/kang/GitHub/wildfire-agent/Maui Wildfires Image.jpg. Identify fire hotspots, assess burn areas, and evaluate risks to Lahaina community.",
                    "🎯 Use YOLO object detection to identify and analyze all objects in the Maui wildfire image, then combine with VLM analysis for comprehensive assessment.",
                    "🔥 Apply YOLO wildfire detection to identify fire indicators and generate emergency recommendations for the Maui image.",
                    "Analyze the current wildfire situation: How large is the area currently burning, what is the rate of spread, and which communities need immediate evacuation?",
                    "What are the safest evacuation routes for communities threatened by the active fire?",
                    "Assess the wildfire risk to critical infrastructure including hospitals, schools, and power stations.",
                    "Generate a comprehensive evacuation plan for affected communities with timeline and resource requirements.",
                ]

                gr.Examples(examples=examples, inputs=question_input)

                gr.HTML("""
                        <div class="footer" id="about">
                            <h3>🔥 About Wildfire Agent</h3>
                            <p>Wildfire Agent is an AI-powered wildfire management system built on the OWL/CAMEL framework, specialized for emergency response, satellite imagery analysis, and evacuation planning.</p>
                            <p>Capabilities: Fire spread analysis • Evacuation planning • Risk assessment • Infrastructure protection • Remote sensing integration</p>
                            <p>© 2025 Based on OWL/CAMEL-AI framework under Apache License 2.0</p>
                            <p><a href="https://github.com/your-username/wildfire-agent" target="_blank">GitHub</a></p>
                        </div>
                    """)

            with gr.Tabs():  # Set conversation record as the default selected tab
                with gr.TabItem("Conversation Record"):
                    # Add conversation record display area
                    with gr.Group():
                        log_display2 = gr.Markdown(
                            value="No conversation records yet.",
                            elem_classes="log-display",
                        )

                    with gr.Row():
                        refresh_logs_button2 = gr.Button("Refresh Record")
                        auto_refresh_checkbox2 = gr.Checkbox(
                            label="Auto Refresh", value=True, interactive=True
                        )
                        clear_logs_button2 = gr.Button(
                            "Clear Record", variant="secondary"
                        )

                with gr.TabItem("Agent Summary"):
                    # Add agent summary display area
                    with gr.Group():
                        summary_display = gr.Markdown(
                            value="No agent actions recorded yet. Run a query to see what the agent accomplishes.",
                            elem_classes="log-display",
                        )

                with gr.TabItem("Image Gallery"):
                    # Add image gallery display area using proper Gradio Gallery
                    with gr.Group():
                        image_gallery_display = gr.Gallery(
                            value=get_workspace_images(),
                            label="🔥 Workspace Image Gallery",
                            show_label=True,
                            elem_id="workspace_gallery",
                            columns=3,
                            rows=2,
                            object_fit="contain",
                            height="auto",
                            allow_preview=True,
                            show_share_button=False,
                            show_download_button=True,
                        )

                    with gr.Row():
                        refresh_gallery_button = gr.Button("🔄 Refresh Gallery", variant="primary")
                        auto_refresh_gallery = gr.Checkbox(
                            label="Auto Refresh", value=False, interactive=True
                        )

                with gr.TabItem("Environment Variable Management", id="env-settings"):
                    with gr.Group(elem_classes="env-manager-container"):
                        gr.Markdown("""
                            ## Environment Variable Management
                            
                            Set model API keys and other service credentials here. This information will be saved in a local `.env` file, ensuring your API keys are securely stored and not uploaded to the network. Correctly setting API keys is crucial for the functionality of the OWL system. Environment variables can be flexibly configured according to tool requirements.
                            """)

                        # Main content divided into two-column layout
                        with gr.Row():
                            # Left column: Environment variable management controls
                            with gr.Column(scale=3):
                                with gr.Group(elem_classes="env-controls"):
                                    # Environment variable table - set to interactive for direct editing
                                    gr.Markdown("""
                                    <div style="background-color: #e7f3fe; border-left: 6px solid #2196F3; padding: 10px; margin: 15px 0; border-radius: 4px;">
                                      <strong>Tip:</strong> Please make sure to run cp .env_template .env to create a local .env file, and flexibly configure the required environment variables according to the running module
                                    </div>
                                    """)

                                    # Enhanced environment variable table, supporting adding and deleting rows
                                    env_table = gr.Dataframe(
                                        headers=[
                                            "Variable Name",
                                            "Value",
                                            "Retrieval Guide",
                                        ],
                                        datatype=[
                                            "str",
                                            "str",
                                            "html",
                                        ],  # Set the last column as HTML type to support links
                                        row_count=10,  # Increase row count to allow adding new variables
                                        col_count=(3, "fixed"),
                                        value=update_env_table,
                                        label="API Keys and Environment Variables",
                                        interactive=True,  # Set as interactive, allowing direct editing
                                        elem_classes="env-table",
                                    )

                                    # Operation instructions
                                    gr.Markdown(
                                        """
                                    <div style="background-color: #fff3cd; border-left: 6px solid #ffc107; padding: 10px; margin: 15px 0; border-radius: 4px;">
                                    <strong>Operation Guide</strong>:
                                    <ul style="margin-top: 8px; margin-bottom: 8px;">
                                      <li><strong>Edit Variable</strong>: Click directly on the "Value" cell in the table to edit</li>
                                      <li><strong>Add Variable</strong>: Enter a new variable name and value in a blank row</li>
                                      <li><strong>Delete Variable</strong>: Clear the variable name to delete that row</li>
                                      <li><strong>Get API Key</strong>: Click on the link in the "Retrieval Guide" column to get the corresponding API key</li>
                                    </ul>
                                    </div>
                                    """,
                                        elem_classes="env-instructions",
                                    )

                                    # Environment variable operation buttons
                                    with gr.Row(elem_classes="env-buttons"):
                                        save_env_button = gr.Button(
                                            "💾 Save Changes",
                                            variant="primary",
                                            elem_classes="env-button",
                                        )
                                        refresh_button = gr.Button(
                                            "🔄 Refresh List", elem_classes="env-button"
                                        )

                                    # Status display
                                    env_status = gr.HTML(
                                        label="Operation Status",
                                        value="",
                                        elem_classes="env-status",
                                    )

                    # 连接事件处理函数
                    save_env_button.click(
                        fn=save_env_table_changes,
                        inputs=[env_table],
                        outputs=[env_status],
                    ).then(fn=update_env_table, outputs=[env_table])

                    refresh_button.click(fn=update_env_table, outputs=[env_table])

        # Set up event handling
        run_button.click(
            fn=process_with_live_logs,
            inputs=[question_input, module_dropdown],
            outputs=[token_count_output, status_output, log_display2, summary_display],
        )

        # Module selection updates description
        module_dropdown.change(
            fn=update_module_description,
            inputs=module_dropdown,
            outputs=module_description,
        )

        # Conversation record related event handling
        refresh_logs_button2.click(
            fn=lambda: get_latest_logs(100, LOG_QUEUE), outputs=[log_display2]
        )

        clear_logs_button2.click(fn=clear_log_file, outputs=[log_display2])

        # Auto refresh control
        def toggle_auto_refresh(enabled):
            if enabled:
                return gr.update(every=3)
            else:
                return gr.update(every=0)

        auto_refresh_checkbox2.change(
            fn=toggle_auto_refresh,
            inputs=[auto_refresh_checkbox2],
            outputs=[log_display2],
        )

        # Image gallery refresh functionality
        refresh_gallery_button.click(
            fn=get_workspace_images,
            outputs=[image_gallery_display]
        )

        # Auto refresh for image gallery
        def toggle_gallery_auto_refresh(enabled):
            if enabled:
                return gr.update(every=5)  # Refresh every 5 seconds for images
            else:
                return gr.update(every=0)

        auto_refresh_gallery.change(
            fn=toggle_gallery_auto_refresh,
            inputs=[auto_refresh_gallery],
            outputs=[image_gallery_display],
        )

        # No longer automatically refresh logs by default

    return app


# Main function
def main():
    try:
        # Initialize logging system
        global LOG_FILE
        LOG_FILE = setup_logging()
        logging.info("🔥 Wildfire Agent Web application started")

        # Start log reading thread
        log_thread = threading.Thread(
            target=log_reader_thread, args=(LOG_FILE,), daemon=True
        )
        log_thread.start()
        logging.info("Log reading thread started")

        # Initialize .env file (if it doesn't exist)
        init_env_file()
        app = create_ui()

        app.queue()
        app.launch(
            share=False,
            favicon_path=os.path.join(
                os.path.dirname(__file__), "assets", "owl-favicon.ico"
            ),
            allowed_paths=[
                ".",  # Current directory
                "workspace",  # Workspace directory  
                "wildfire_workspace",  # Wildfire workspace directory
                "/Users/kang/GitHub/wildfire-agent",  # Project root
            ],
        )
    except Exception as e:
        logging.error(f"Error occurred while starting the application: {str(e)}")
        print(f"Error occurred while starting the application: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Ensure log thread stops
        STOP_LOG_THREAD.set()
        STOP_REQUESTED.set()
        logging.info("Application closed")


if __name__ == "__main__":
    main()
