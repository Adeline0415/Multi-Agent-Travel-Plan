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
from azure.ai.projects.models import CodeInterpreterTool

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
from tools import WeatherPlugin, DALLEPlugin

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
                    
                    # # Handle tool calls for weather agent
                    # try:
                    #     if content.name == "WeatherAdvisorAgent" and hasattr(content, 'tool_calls') and content.tool_calls:
                    #         for tool_call in content.tool_calls:
                    #             if tool_call.function.name == "get_weather_forecast":
                    #                 args = json.loads(tool_call.function.arguments)
                    #                 # Use the WeatherPlugin from tools.py
                    #                 weather_result = self.weather_plugin.get_weather_forecast(
                    #                     location_name=args.get("location_name"),
                    #                     start_date=args.get("start_date"),
                    #                     end_date=args.get("end_date")
                    #                 )
                    #                 # Add tool result back to chat
                    #                 await chat.add_chat_message(
                    #                     message=weather_result,
                    #                     role="tool",
                    #                     tool_call_id=tool_call.id
                    #                 )
                    #                 group_chat_responses.append({
                    #                     "role": "tool",
                    #                     "content": weather_result
                    #                 })
                    # except AttributeError as e:
                    #     print(f"處理工具調用時出錯: {e}")
                print("Group chat responses:", group_chat_responses)
                # Now create the FileGenerationAgent outside the group chat
                # 修改這一行
                final_response, urls = await self.run_file_generation_agent(
                    group_chat_responses, 
                    current_date
                )
                
                return final_response, urls
                    
            finally:
                # Cleanup
                await chat.reset()
                for agent in agents.values():
                    await client.agents.delete_agent(agent.id)
    
    async def run_file_generation_agent(self, group_chat_responses, current_date):
        """Asynchronous version of the file generation agent with robust file handling"""
        dalle_plugin = DALLEPlugin(self.dalle_client)

        # 創建整合的訊息
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

        print(f"Consolidated message prepared, length: {len(consolidated_message)}")
        
        # 創建異步 AI Project 客戶端
        async with AsyncAIProjectClient(
            endpoint=self.config["AIPROJECT_ENDPOINT"],
            subscription_id=self.config["AIPROJECT_SUBSCRIPTION_ID"],
            resource_group_name=self.config["AIPROJECT_RESOURCE_GROUP_NAME"],
            project_name=self.config["AIPROJECT_PROJECT_NAME"],
            credential=self.credential
        ) as project_client:
            try:
                # 創建 FileGenerationAgent
                code_interpreter = CodeInterpreterTool()
                file_gen_definition = await project_client.agents.create_agent(
                    model=self.config["MODEL_DEPLOYMENT_NAME"],
                    name="FileGenerationAgent",
                    instructions="""
                   You are a travel plan formatter that creates beautiful HTML presentations of travel itineraries.
                
                Take the complete multi-day travel itinerary and additional adjustment suggestions (e.g., based on weather, transportation) from the other agents and format it into a cohesive, visually appealing HTML document.
                
                Your output must:
                - Match the language of the input (Japanese/Chinese/English)
                - Include a full HTML document with <html>, <head>, and <body> sections
                - Insert an embedded Google Map <iframe> for each main location or day
                - Complete multi-day travel plan included weather information, transportation suggestions, and any other relevant details (like luanch and dinner options if available)
                
                Images:
                - Generate ONE representative image for the ENTIRE trip based on the overall travel theme using the generate_image tool
                - Use the image URL directly in an <img> tag, for example:
                <img src="[image_url]" alt="Trip Cover Image" style="width:50%;height:auto;">
                
                File Handling:
                - Save the final HTML content to a file named "travel_plan.html" using the code interpreter
                - Your code should look like:
                ```python
                with open('travel_plan.html', 'w', encoding='utf-8') as f:
                    f.write(html_content)
                print("HTML file created: travel_plan.html")
                ```
                
                Style and Layout:
                - Use clear headings and structure
                - Use bullet points for listing places and activities
                - Highlight important information like hotel names and transportation
                - Include a clean, readable CSS style inside the HTML
                
                IMPORTANT:
                1. First analyze all advice from other agents
                2. Then use generate_image ONCE to create a trip cover image
                3. Create the complete HTML with embedded maps and the image
                4. Save the HTML to a file using the code interpreter
                5. Print the EXACT file paths of any files you create
                6. Only after all these steps are complete, add "TERMINATE" to your message

                CRITICAL NOTE: Your primary purpose is to create and save the HTML file. If you fail to save a proper HTML file, you have failed your mission. No exceptions.
                    """,
                    tools=code_interpreter.definitions,
                    tool_resources=code_interpreter.resources
                )
            
                # 創建新執行緒
                thread = await project_client.agents.create_thread()
                
                # 添加整合訊息至執行緒
                await project_client.agents.create_message(
                    thread_id=thread.id,
                    role="user",
                    content=consolidated_message
                )
                
                # 執行代理
                run = await project_client.agents.create_run(
                    thread_id=thread.id,
                    agent_id=file_gen_definition.id
                )
                
                print(f"File Generation Agent run created with ID: {run.id}")
                
                # 等待執行完成，使用擴展間隔以避免速率限制
                last_check_time = time.time()
                min_check_interval = 5
                timeout = 600  # 5 分鐘超時
                start_time = time.time()
                
                while run.status in ["queued", "in_progress", "requires_action"]:
                    current_time = time.time()
                    
                    # 檢查超時
                    if current_time - start_time > timeout:
                        print(f"Run timed out after {timeout} seconds")
                        break
                    
                    # 強制最小檢查間隔
                    elapsed = current_time - last_check_time
                    if elapsed < min_check_interval:
                        await asyncio.sleep(min_check_interval - elapsed)
                    
                    print(f"Run status: {run.status}")
                    last_check_time = time.time()
                    
                    run = await project_client.agents.get_run(thread_id=thread.id, run_id=run.id)
                
                print(f"File Generation Agent run completed with status: {run.status}")
                
                print("Waiting for file registration to complete...")
                await asyncio.sleep(40)
                
                # 獲取執行緒訊息
                messages = await project_client.agents.list_messages(thread_id=thread.id)
                
                # 處理檔案註解和取得最終回應
                file_urls = []
                final_response = ""
                
                # 獲取助理訊息
                for msg in messages.data:
                    if msg.role == "assistant":
                        for content_item in msg.content:
                            if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                                final_response = content_item.text.value
                                break
                        break
                
                # 處理檔案註解
                try:
                    print("Looking for file path annotations...")
                    
                    # 檢查是否有檔案註解
                    if hasattr(messages, 'file_path_annotations') and messages.file_path_annotations:
                        annotations = messages.file_path_annotations
                        print(f"Found {len(annotations)} file path annotations")
                        
                        for file_path_annotation in annotations:
                            file_id = file_path_annotation.file_path.file_id
                            
                            # 使用重試邏輯獲取檔案
                            data_bytes = await self.get_file_with_retry(project_client, file_id)
                            
                            if data_bytes:
                                # 處理檔案
                                file_name = os.path.basename(file_path_annotation.text)
                                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                new_file_name = f"{timestamp}-{file_name}"
                                
                                file_ext = file_name.split('.')[-1]
                                content_type = self._get_mime_type(file_ext)
                                
                                image_url = self.blob_manager.upload_blob(
                                    image_data=data_bytes,
                                    file_name=new_file_name,
                                    content_type=content_type
                                )
                                
                                file_urls.append(image_url)
                                print(f"Successfully uploaded file to {image_url}")
                                
                                # 替換回應中的沙盒路徑
                                final_response = final_response.replace(
                                    file_path_annotation.text, 
                                    image_url
                                )
                    else:
                        print("No file_path_annotations found, checking message content for file paths...")
                        
                        # 從訊息內容提取檔案路徑
                        sandbox_file_path = None
                        for msg in messages.data:
                            if msg.role == "assistant":
                                for content_item in msg.content:
                                    if hasattr(content_item, 'text') and hasattr(content_item.text, 'value'):
                                        content = content_item.text.value
                                        file_path = self.extract_file_path_from_content(content)
                                        if file_path:
                                            sandbox_file_path = file_path
                                            print(f"Found file path in message: {sandbox_file_path}")
                                            break
                        
                        # 如果找到了檔案路徑但沒有檔案註解，則從訊息提取 HTML 內容
                        if sandbox_file_path or "<!DOCTYPE html>" in final_response or "<html" in final_response:
                            html_content = self.extract_html_from_content(final_response)
                            
                            if html_content:
                                print(f"Extracted HTML content, length: {len(html_content)}")
                                
                                # 手動創建和上傳檔案
                                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                file_name = f"{timestamp}-travel_plan.html"
                                
                                image_url = self.blob_manager.upload_blob(
                                    image_data=html_content.encode('utf-8'),
                                    file_name=file_name,
                                    content_type="text/html"
                                )
                                
                                file_urls.append(image_url)
                                print(f"Manually created and uploaded HTML file to {image_url}")
                            else:
                                print("Could not extract HTML content from the response")
                except Exception as e:
                    print(f"Error processing file annotations: {e}")
                    import traceback
                    traceback.print_exc()
                
                # 將檔案 URL 添加到回應中
                if file_urls:
                    final_response += "\n\n## Generated Files:\n"
                    for i, url in enumerate(file_urls, 1):
                        final_response += f"\n{i}. [View File {i}]({url})"
                else:
                    final_response += "\n\nNo files were generated or there was an error in file processing."
                
                # 刪除 "TERMINATE" 字串如果存在
                if "TERMINATE" in final_response:
                    final_response = final_response.replace("TERMINATE", "").strip()
                    
                return final_response, file_urls
            
            finally:
                # 清理代理
                try:
                    await project_client.agents.delete_agent(file_gen_definition.id)
                except Exception as e:
                    print(f"Error deleting agent: {e}")
    
    async def get_file_with_retry(self, project_client, file_id, max_retries=5):
        """使用重試邏輯獲取檔案內容"""
        for attempt in range(max_retries):
            try:
                print(f"Getting file content attempt {attempt+1}/{max_retries}")
                
                if attempt > 0:
                    await asyncio.sleep(2 * attempt)
                
                async with asyncio.timeout(30):
                    data_bytes_chunks = await project_client.agents.get_file_content(file_id=file_id)
                    data_bytes = b''.join(data_bytes_chunks)
                    
                    if len(data_bytes) > 0:
                        print(f"Successfully retrieved file, size: {len(data_bytes)} bytes")
                        return data_bytes
                    else:
                        print(f"Warning: Empty file content on attempt {attempt+1}")
                        if attempt == max_retries - 1:
                            return None
            
            except asyncio.TimeoutError:
                print(f"Timeout getting file on attempt {attempt+1}")
                if attempt == max_retries - 1:
                    return None
                    
            except Exception as e:
                print(f"Error getting file on attempt {attempt+1}: {str(e)}")
                if attempt == max_retries - 1:
                    return None
        
        return None
    
    def extract_file_path_from_content(self, content):
        """從訊息內容中提取檔案路徑"""
        patterns = [
            r'File saved:\s+([^\s]+)',
            r'HTML file created:\s+([^\s]+)',
            r'file created:\s+([^\s]+)',
            r'Created file:\s+([^\s]+)',
            r'(?:saved|written) to [\'"]?([^\'"\s]+)[\'"]?',
            r'Output file:\s+([^\s]+)',
        ]
        
        for pattern in patterns:
            import re
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                path = matches[0].strip().strip(r'`\'"()')
                if path.startswith('/mnt/data/'):
                    path = path[len('/mnt/data/'):]
                return path
        
        return None
    
    def extract_html_from_content(self, content):
        """從訊息內容中提取 HTML 代碼"""
        # 嘗試多種 HTML 開頭標籤
        html_starts = ["<!DOCTYPE html>", "<html", "<HTML"]
        
        for start_tag in html_starts:
            html_start = content.find(start_tag)
            if html_start >= 0:
                # 搜尋結束標籤
                html_end = content.find("</html>", html_start)
                if html_end < 0:
                    html_end = content.find("</HTML>", html_start)
                
                if html_end > html_start:
                    return content[html_start:html_end + 7]  # 包含結尾標籤
        
        return None
    
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