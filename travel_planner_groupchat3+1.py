import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Tuple
import uuid
import time
from typing import Annotated

# Azure AI SDK imports
from azure.identity import ClientSecretCredential
from azure.identity import DefaultAzureCredential
from azure.ai.projects.aio import AIProjectClient as AsyncAIProjectClient  # Async version for Semantic Kernel
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import CodeInterpreterTool, RequiredFunctionToolCall, SubmitToolOutputsAction, ToolOutput, MessageRole

# Semantic Kernel imports
from semantic_kernel.agents import AgentGroupChat, AzureAIAgent, AzureAIAgentSettings
from semantic_kernel.agents.strategies import KernelFunctionTerminationStrategy, TerminationStrategy, SelectionStrategy
from semantic_kernel.contents import AuthorRole, ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelFunctionFromPrompt
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from openai import AzureOpenAI

# Weather API imports
import requests

# Local imports
from util import AzureBlobManager
from tools import WeatherPlugin, DALLEPlugin, LocationPlugin

# Configure logging
import logging
# 設置更高的日誌等級，只顯示警告和錯誤
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# define agent names
PLANNER_NAME = "TravelPlannerAgent"
WEATHER_NAME = "WeatherAdvisorAgent"
ROUTE_NAME = "RouteMasterAgent"
FILE_NAME = "FileGenerationAgent"

class TravelPlanningTerminationStrategy(TerminationStrategy):
    """A strategy for determining when the travel planning process should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        if not history:
            return False
        
        last_message = history[-1]
        
        # Terminate if the last agent (the RouteMasterAgent) has responded
        if agent.name in [ROUTE_NAME, WEATHER_NAME] and len(history) > 6:
            return True
            
        # Also terminate if we reach a certain number of iterations
        if len(history) > 20:  # Prevent infinite loops
            return True
            
        return False
    
class CustomSelectionStrategy(SelectionStrategy):
    """A custom selection strategy for the travel planning agents."""
    
    async def select_agent(self, agents, history):
        if not history:
            return agents[0]
        
        last_message = history[-1]
        
        if hasattr(last_message, 'name'):
            last_agent_name = last_message.name
            
            if last_agent_name == PLANNER_NAME:
                for agent in agents:
                    if agent.name == ROUTE_NAME:
                        return agent
                        
            elif last_agent_name == ROUTE_NAME:
                for agent in agents:
                    if agent.name == WEATHER_NAME:
                        return agent
                        
            elif last_agent_name == WEATHER_NAME:
                for agent in agents:
                    if agent.name == ROUTE_NAME:
                        return agent
        
        # default
        return agents[0]

class TravelPlanningSystem:
    """Multi-agent travel planning system using Semantic Kernel and Azure AI Foundry."""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.blob_manager = AzureBlobManager(connection_string=config["AZURE_STORAGE_CONNECTION_STRING"])

        self.dalle_client = AzureOpenAI(
            api_key=self.config["AZURE_OPENAI_API_KEY"],
            api_version="2024-02-01",
            azure_endpoint = self.config["AZURE_OPENAI_ENDPOINT"],
        )
        self.weather_plugin = WeatherPlugin(self.config["GEOCODING_API_KEY"])
        # Create credentials
        self.credential = DefaultAzureCredential()
    
    async def create_travel_planning_agents(self, client, current_date: str):
        """Create all the agents needed for travel planning."""
        # weather_plugin = WeatherPlugin(self.config["GEOCODING_API_KEY"])
        # 1. Travel Planner Agent
        planner_definition = await client.agents.create_agent(
            model=self.config["MODEL_DEPLOYMENT_NAME"],
            name=PLANNER_NAME,
            instructions=f"""
            Today is {current_date}. You are a helpful assistant that can suggest a travel plan for a user based on their request.
            When generating the itinerary, calculate and include the actual calendar date for each day of the trip.
            In addition, for each destination or attraction in the itinerary, please include the specific area or neighborhood name in Tokyo 
            (e.g., Asakusa, Shinjuku, Roppongi, Ueno, etc.).
            """
        )
        planner_agent = AzureAIAgent(client=client, definition=planner_definition)
        
        # 2. RouteMaster Agent
        routemaster_definition = await client.agents.create_agent(
            model=self.config["MODEL_DEPLOYMENT_NAME"],
            name=ROUTE_NAME,
            instructions="""
            你是一位在地經驗豐富的包車司機，熟悉當地所有熱門與隱藏版景點，擁有超過10年的旅遊接待與路線規劃經驗。
            你擅長依照旅客需求設計順暢、有效率且舒適的旅遊路線，並靈活運用包車的優勢，避開人潮與節省交通時間。

            你的任務是根據使用者提供的旅遊天數與偏好，規劃**全程以包車為主的觀光路線**，包含景點順序、停留時間建議與每日路線動線。
            請以在地導遊兼司機的角度，提出貼心、實用的安排，並特別留意以下幾點：

            1. **路線應避免重複與繞路**，確保每天動線順暢、不折返，並合理分配景點密度與車程時間。
            2. **考慮旅客體力與舒適度**，避免安排過長車程或連續密集行程，適當安排午餐與休息點。
            3. **根據不同天氣與時段調整景點順序**，例如將戶外活動安排在早上，避開午後高溫或人潮。

            最終輸出請清楚列出每日預定行程、主要景點、車程時間預估與司機建議，並以親切專業的語氣說明規劃邏輯。
            """
        )
        routemaster_agent = AzureAIAgent(client=client, definition=routemaster_definition)

        # 3. Weather Advisor Agent (with tools)
        weather_advisor_definition = await client.agents.create_agent(
            model=self.config["MODEL_DEPLOYMENT_NAME"],
            name=WEATHER_NAME,
            instructions=f"""
            你是一位擁有超過20年經驗的氣象專業知識旅遊顧問，擅長根據即時與預測天氣，
            協助使用者動態調整每日行程安排。你的任務是：
            1. 使用get_weather_forecast並根據指定**日期與實際地點(請使用英文地名)**的天氣資訊，提出合適的行程建議。
            2. **請避免使用籠統的地名查詢天氣**，而應以每日實際行程地點為查詢依據。    
            若天氣不佳（如下雨、強風、酷暑），請建議更換為室內景點，
            若天氣良好，則可推薦戶外行程。"
            建議內容需具體、合理，並配合使用者原本的動線
            """,
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather_forecast",
                    "description": "Get weather forecast for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location_name": {
                                "type": "string",
                                "description": "The name of the location"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Start date in YYYY-MM-DD format"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date in YYYY-MM-DD format"
                            }
                        },
                        "required": ["location_name", "start_date", "end_date"]
                    }
                }
            }]
        )

        
        weather_advisor_agent = AzureAIAgent(
            client=client, 
            definition=weather_advisor_definition,
            plugins=[self.weather_plugin]  # 在這裡添加插件
        )
        
        return {
            "planner": planner_agent,
            "routemaster": routemaster_agent,
            "weather_advisor": weather_advisor_agent
        }
    
    async def run_travel_planning(self, task_query: str, user_id: str = None) -> Tuple[str, List[str]]:
        """Run the travel planning using Azure AI Foundry."""
        
        if not user_id:
            user_id = str(uuid.uuid4())
        
        
        # Check or create thread
        thread_id = self.blob_manager.read_thread(user_id)
        current_date = datetime.now().strftime("%Y-%m-%d")
        group_chat_responses = []
       
        
        # Create AI Project client
        async with AsyncAIProjectClient(
            endpoint=self.config["AIPROJECT_ENDPOINT"],
            subscription_id=self.config["AIPROJECT_SUBSCRIPTION_ID"],
            resource_group_name=self.config["AIPROJECT_RESOURCE_GROUP_NAME"],
            project_name=self.config["AIPROJECT_PROJECT_NAME"],
            credential=self.credential
        ) as client:
            # Create all agents
            agents = await self.create_travel_planning_agents(client, current_date)
            
            selection_strategy = CustomSelectionStrategy()
            termination_strategy = TravelPlanningTerminationStrategy()

            # Create the group chat
            chat = AgentGroupChat(
                agents=[
                    agents["planner"],
                    agents["routemaster"],
                    agents["weather_advisor"]
                ],
                selection_strategy=selection_strategy,
                termination_strategy=termination_strategy
            )
            
            try:
                # Add the task to the group chat
                await chat.add_chat_message(message=task_query)
                print(f"# {AuthorRole.USER}: '{task_query}'")
                group_chat_responses.append({"role": "user", "content": task_query})
                
                # Run the chat
                async for content in chat.invoke():
                    print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
                    group_chat_responses.append({
                        "role": content.role,
                        "name": content.name,
                        "content": content.content
                    })

                instruction = "Please combine the above into a complete travel plan, including everyday itinerary, accommodation, transportation and weather. And you must use code interpreter to generate the final travel plan into an HTML file."
                group_chat_responses.append({
                    "role": "user",
                    "content": instruction
                })
                
                # Now create the FileGenerationAgent outside the group chat
                final_response, urls = self.run_file_generation_agent(
                    group_chat_responses, 
                    current_date
                )
                
                return final_response, urls
                    
            finally:
                print("Completed travel planning run")
    
    def run_file_generation_agent(self, group_chat_responses, current_date):
        """Synchronous version of the file generation agent"""
        dalle_plugin = DALLEPlugin(self.dalle_client)
        location_plugin = LocationPlugin(self.config["GEOCODING_API_KEY"])

        # Create a consolidated message
        consolidated_message = "# Travel Planning Group Chat Results\n\n"
        for resp in group_chat_responses:
            role = resp.get("role", "")
            name = resp.get("name", "")
            content = resp.get("content", "")
            
            if role == "user":
                consolidated_message += f"## User Query\n{content}\n\n"
            elif role == "assistant":
                consolidated_message += f"## {name} Response\n{content}\n\n"
            elif role == "tool":
                consolidated_message += f"## Tool Response\n{content}\n\n"

        print(f"Consolidated message: {consolidated_message}")
        # Create synchronous AI Project client
        project_client = AIProjectClient(
            endpoint=self.config["AIPROJECT_ENDPOINT"],
            subscription_id=self.config["AIPROJECT_SUBSCRIPTION_ID"],
            resource_group_name=self.config["AIPROJECT_RESOURCE_GROUP_NAME"],
            project_name=self.config["AIPROJECT_PROJECT_NAME"],
            credential=self.credential
        )
        
        try:
            # Create the FileGenerationAgent
            code_interpreter = CodeInterpreterTool()
            file_gen_definition = project_client.agents.create_agent(
                model=self.config["MODEL_DEPLOYMENT_NAME"],
                name=FILE_NAME,
                instructions="""
                - Take a complete multi-day travel itinerary and additional adjustment suggestions (e.g., based on weather, transportation).
                - The output should match the input language.   

                HTML Output Requirements:
                - Generate the travel plan as a complete HTML document, including <html>, <head>, and <body> sections. Save as a file.

                - For each day in the itinerary:
                    - **MUST** include:
                        - Weather forecast (temperature and conditions)
                        - Transportation types
                        - Hotel name (bolded for emphasis)        
                    - Use the get_lat_long(location_name) function to retrieve the latitude and longitude required for the iframe.
                    - Then, use those coordinates to build the iframe using the following format:
                        <iframe
                        width="600"
                        height="450"
                        style="border:0"
                        loading="lazy"
                        allowfullscreen
                        referrerpolicy="no-referrer-when-downgrade"
                        src="https://www.google.com/maps?q={lat},{lon}&hl=zh-TW&output=embed">
                        </iframe>>

                Images:
                - Use the dalle function to generate only **one** representative image should be generated for the **entire trip** based on the overall travel theme.
                - Use the original image URL directly, **do not save it locally**.
                - Insert the image using standard `<img src="...">` HTML syntax, for example:
                ```html
                <img src="https://dalleprodsec.blob.core.windows.net/your_image_path.png?...sas_token..." alt="Trip Cover Image" style="width:50%;height:auto;">
                ```
                File Handling:
                - Save the final HTML content as a file, for example "travel_plan.html".
                - Provide a Markdown download link for the file:
                ```
                [Download Travel Plan](sandbox:/mnt/data/travel_plan.html)
                ```  

                Style and Layout:
                - Organize the content with clear headings:
                - Use bullet points `<ul><li>...</li></ul>` for listing places or meals.
                - Highlight key information like hotel names and transportation types with **bold text**.
                - Maintain a clean, readable layout. You may optionally embed simple `<style>` CSS inside the HTML for better visuals (e.g., spacing, font-size).

                PDF Generation:
                - When generating PDFs that include Chinese content, always use built-in fonts to ensure compatibility and avoid font rendering issues.
                - Follow these rules:
                    - Use `canvas.Canvas()` from ReportLab to create the PDF.
                    - Register a built-in CJK font using `UnicodeCIDFont`. Use `"STSong-Light"` as the default for Traditional Chinese.
                    - Do NOT use external `.ttf` or `.ttc` fonts unless explicitly instructed to.
                """,
                tools=code_interpreter.definitions + dalle_plugin.definitions + location_plugin.definitions,
                tool_resources=code_interpreter.resources
            )
        
            # Create a new thread
            thread = project_client.agents.create_thread()
            
            # Add the consolidated message to the thread
            message = project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=consolidated_message
            )
            
            # Run the agent
            run = project_client.agents.create_run(
                thread_id=thread.id,
                agent_id=file_gen_definition.id
            )
            
            print(f"File Generation Agent run created with ID: {run.id}")
            
            # Wait for the run to complete
            while run.status in ["queued", "in_progress", "requires_action"]:
                print(f"Run status: {run.status}")
                time.sleep(2)
                run = project_client.agents.get_run(thread_id=thread.id, run_id=run.id)
                
                if run.status == "requires_action" and isinstance(run.required_action, SubmitToolOutputsAction):
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls
                    if not tool_calls:
                        print("No tool calls provided - cancelling run")
                        project_client.agents.cancel_run(
                            thread_id=thread.id, run_id=run.id)
                        break
                    
                    tool_outputs = []
                    for tool_call in tool_calls:
                        if isinstance(tool_call, RequiredFunctionToolCall):
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                            
                            print(f"Executing tool call: {function_name} with args: {function_args}")
                            
                            try:
                                # Handle DALLE image generation
                                if function_name == "generate_image":
                                    prompt = function_args.get("prompt")
                                    image_url = dalle_plugin.generate_image(prompt)
                                    tool_outputs.append(
                                        ToolOutput(
                                            tool_call_id=tool_call.id,
                                            output=image_url
                                        )
                                    )
                                    print(f"Generated image URL: {image_url}")
                                # Handle location service requests
                                elif function_name == "get_lat_long":
                                    location_name = function_args.get("location_name")
                                    location_data = location_plugin.get_lat_long(location_name)
                                    tool_outputs.append(
                                        ToolOutput(
                                            tool_call_id=tool_call.id,
                                            output=location_data
                                        )
                                    )
                                    print(f"Generated location data: {location_data}")
                            except Exception as e:
                                error_msg = f"Error executing tool_call {tool_call.id}: {e}"
                                print(error_msg)
                                # Optional: Return error message to the model
                                tool_outputs.append(
                                    ToolOutput(
                                        tool_call_id=tool_call.id,
                                        output=f"Error: {str(e)}"
                                    )
                                )
                    
                    print(f"Tool outputs: {tool_outputs}")
                    if tool_outputs:
                        project_client.agents.submit_tool_outputs_to_run(
                            thread_id=thread.id, run_id=run.id, tool_outputs=tool_outputs
                        )
                
            print(f"File Generation Agent run completed with status: {run.status}")
            messages = project_client.agents.list_messages(thread_id=thread.id)
            
            # Process file path annotations and get final response
            file_urls = []
            final_response = ""
            
            # Get the assistant's message
            try:
                # Find the assistant's message
                for msg in messages.data:
                    if msg.role == "assistant":
                        # Get the content from the message
                        for content_item in msg.content:
                            if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                                final_response = content_item.text.value
                                break
            except Exception as e:
                print(f"Error getting assistant message: {e}")
                final_response = "Could not retrieve the final travel plan."
            
            # Process file path annotations to get files
            try:
                # Print details for debugging
                print(f"Looking for file path annotations...")
                
                if hasattr(messages, 'file_path_annotations'):
                    annotations = messages.file_path_annotations
                    print(f"Found {len(annotations)} file path annotations")
                    
                    for file_path_annotation in annotations:
                        try:
                            print(f"Processing file: {file_path_annotation.text}")
                            file_id = file_path_annotation.file_path.file_id
                            print(f"Getting file content for {file_id}")
                            
                            # Synchronous file content retrieval
                            data_bytes_chunks = project_client.agents.get_file_content(file_id=file_id)
                            data_bytes = b''.join(data_bytes_chunks)
                            
                            print(f"Successfully read file, size: {len(data_bytes)} bytes")
                            
                            # Upload to blob storage
                            file_name = os.path.basename(file_path_annotation.text)
                            timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                            new_file_name = f"{timestamp}-{file_name}"
                            
                            # Get MIME type
                            file_ext = file_name.split('.')[-1]
                            content_type = self._get_mime_type(file_ext)
                            
                            # Upload file to blob storage
                            image_url = self.blob_manager.upload_blob(
                                image_data=data_bytes,
                                file_name=new_file_name,
                                content_type=content_type
                            )
                            
                            file_urls.append(image_url)
                            print(f"Successfully uploaded file to {image_url}")
                            
                            # Replace sandbox paths with actual URLs in the response
                            final_response = final_response.replace(
                                file_path_annotation.text, 
                                image_url
                            )
                            
                        except Exception as e:
                            print(f"Error processing file {file_path_annotation.text}: {str(e)}")
                            import traceback
                            traceback.print_exc()
                else:
                    print("No file_path_annotations found in messages object")
                    # Print message object details for debugging
                    print(f"Messages object attributes: {dir(messages)}")
            except Exception as e:
                print(f"Error processing file path annotations: {e}")
                import traceback
                traceback.print_exc()
            
            # Add file URLs to response if any were generated
            if file_urls:
                final_response += "\n\n## Generated Files:\n"
                for i, url in enumerate(file_urls, 1):
                    final_response += f"\n{i}. [View File {i}]({url})"
            else:
                final_response += "\n\nNo files were generated or there was an error in file processing."
            
            # Remove TERMINATE if present
            if "TERMINATE" in final_response:
                final_response = final_response.replace("TERMINATE", "").strip()
                
            return final_response, file_urls
        
        finally:
            print("completed run for file generation agent")
    
    def _get_mime_type(self, extension: str) -> str:
        """Get MIME type from file extension."""
        import mimetypes
        if not extension.startswith('.'):
            extension = '.' + extension
        mime_type, _ = mimetypes.guess_type('file' + extension)
        return mime_type or 'application/octet-stream'  # fallback

async def main():
    """Main execution function."""
    # Load configuration from environment variables
    config = {
        "GEOCODING_API_KEY": "7df67dd4bedc45538e04cb7c3c7fbc72",
        "MODEL_DEPLOYMENT_NAME": "gpt-4o",
        "AIPROJECT_ENDPOINT": "https://swedencentral.api.azureml.ms",
        "AIPROJECT_SUBSCRIPTION_ID": "d598fcd6-0db7-43df-adb8-7b537c178a92",
        "AIPROJECT_RESOURCE_GROUP_NAME": "adeline",
        "AIPROJECT_PROJECT_NAME": "a-adelineyu-semantic",
        "AZURE_STORAGE_CONNECTION_STRING": "DefaultEndpointsProtocol=https;AccountName=adeline7238714506;AccountKey=IHKpIcym2hjyBMCnZ6Xhb3on47QXnzzZKNGyjXAyhQQA3Xbgf9Xk0Z0Wy+FeSZcWf14RHxRYgEQ8+AStpkjnXA==;EndpointSuffix=core.windows.net",
        "AZURE_OPENAI_API_KEY": "FMyPlo9BIi5W4xmhSio4eS3faMr2YRZ6g5cOBmYJ2k2iVfEOxkrUJQQJ99BCACfhMk5XJ3w3AAABACOGTzmA",
        "AZURE_OPENAI_ENDPOINT": "https://adeline0415openai.openai.azure.com/",
    }
    
    # Initialize the travel planning system
    travel_system = TravelPlanningSystem(config)
    
    # Task query
    task_query = """
    明天要出發去東京,請規劃一個七天六夜從台北出發的東京行程,以下是幾個重點
    1.每天晚上安排不同的飯店入住(請提供飯店名稱)
    2.全程坐大眾運輸交通工具(請標示需搭東京Metro、都營地鐵或JR)
    3.其中有一天要去迪士尼樂園
    4.出入境皆為成田機場(20:30抵達) (08:30起飛)
    """
    
    # Run the travel planning
    final_plan, file_urls = await travel_system.run_travel_planning(task_query)
    
    print("\n=== FINAL TRAVEL PLAN ===\n")
    print(final_plan)
    
    if file_urls:
        print("\n=== GENERATED FILES ===\n")
        for url in file_urls:
            print(url)

if __name__ == "__main__":
    asyncio.run(main())