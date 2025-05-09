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
from semantic_kernel.agents.strategies import TerminationStrategy
from semantic_kernel.contents import AuthorRole
from semantic_kernel.functions import kernel_function
from semantic_kernel import Kernel
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

class TravelPlanningTerminationStrategy(TerminationStrategy):
    """A strategy for determining when the travel planning process should terminate."""

    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        if not history:
            return False
        
        last_message = history[-1]
        
        # Terminate if the last agent (the RouteMasterAgent or WeatherAdvisorAgent) has responded
        if agent.name in ["RouteMasterAgent", "WeatherAdvisorAgent"] and len(history) > 6:
            return True
            
        # Also terminate if we reach a certain number of iterations
        if len(history) > 40:  # Prevent infinite loops
            return True
            
        return False

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
            name="TravelPlannerAgent",
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
            name="RouteMasterAgent",
            instructions="""
            你是一位擁有超過20年自由行經驗的旅遊達人，熟悉所有地鐵與JR路線，
            精通景點動線安排、住宿推薦與行李轉移策略。你的任務是根據使用者提供的旅遊行程，
            從專家的角度提供批判性回饋，指出潛在問題與可優化之處，並給出具體可執行的改善方案，
            包含：每日交通順暢度、住宿選擇合理性、體力負擔與景點分布安排。
            請特別留意以下幾點：
            1. 避免頻繁的地鐵轉乘與跨公司轉線，例如東京Metro、都營地鐵與JR間的跳線移動，因為這會增加交通費與等待時間。
            2. 檢查每晚飯店與當天行程之間的地理順暢性與合理距離，避免來回折返與不必要的長距離移動。
            3. 若行程安排不夠合理，請提供具體修改建議，包含推薦替代景點、飯店名稱與順路路線，
            確保全程以大眾交通工具為主，並盡量簡化交通移動路徑。
            """
        )
        routemaster_agent = AzureAIAgent(client=client, definition=routemaster_definition)

        # 3. Weather Advisor Agent (with tools)
        weather_advisor_definition = await client.agents.create_agent(
            model=self.config["MODEL_DEPLOYMENT_NAME"],
            name="WeatherAdvisorAgent",
            instructions=f"""
            Today is {current_date}. 你是一位擁有超過20年經驗的氣象專業知識旅遊顧問，擅長根據即時與預測天氣，
            協助使用者動態調整每日行程安排。你的任務是：
            1. 你**必須**使用 `get_weather_forecast` 工具根據實際地點（如：新宿、淺草、六本木、迪士尼樂園）來查詢天氣資料，不能自行假設天氣。
            2. 請避免使用籠統的地名如「Tokyo」查詢天氣，而應以每日實際行程地點為查詢依據。
            3. 若天氣不佳（如下雨、強風、酷暑），請建議更換為室內景點，
            4. 若天氣良好，則可推薦戶外行程。
            5. 請根據使用者的行程安排，提供具體的建議與替代方案，並附上天氣預報資料。

            建議內容需具體、合理，並配合使用者原本的動線與住宿位置，避免增加移動成本與轉乘。
            **Important:** You must always call the `get_weather_forecast` tool whenever you need weather information, and never fabricate weather data yourself.
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
        file_urls = []
        
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
            
            # Create the group chat
            chat = AgentGroupChat(
                agents=[
                    agents["planner"],
                    agents["routemaster"],
                    agents["weather_advisor"]
                ],
                termination_strategy=TravelPlanningTerminationStrategy()
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
                    
                print("Group chat responses:", group_chat_responses)
                # Now create the FileGenerationAgent outside the group chat
                final_response, urls = self.run_file_generation_agent(
                    group_chat_responses, 
                    current_date
                )
                
                return final_response, urls
                    
            finally:
                # Cleanup
                await chat.reset()
                for agent in agents.values():
                    await client.agents.delete_agent(agent.id)
    
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
                name="FileGenerationAgent",
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
            # Clean up the agent
            try:
                project_client.agents.delete_agent(file_gen_definition.id)
            except Exception as e:
                print(f"Error deleting agent: {e}")
    
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