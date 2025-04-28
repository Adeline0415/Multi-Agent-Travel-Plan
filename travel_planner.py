import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Tuple
import uuid
from typing import Annotated

# Azure AI SDK imports
from azure.identity import ClientSecretCredential
from azure.identity import DefaultAzureCredential
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import CodeInterpreterTool

# Semantic Kernel imports
from semantic_kernel.agents import AgentGroupChat, AzureAIAgent, AzureAIAgentSettings
from semantic_kernel.agents.strategies import TerminationStrategy
from semantic_kernel.contents import AuthorRole
from semantic_kernel.functions import kernel_function
from semantic_kernel import Kernel

# Weather API imports
import requests

# Local imports
from util import AzureBlobManager

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
        
        # Terminate if the summary agent outputs a complete plan with 'TERMINATE'
        if agent.name == "TravelSummaryAgent" and "TERMINATE" in last_message.content:
            return True
            
        # # Also terminate if we reach a certain number of iterations
        # if len(history) > 40:  # Prevent infinite loops
        #     return True
            
        return False

class WeatherService:
    """Service for weather-related functions"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def get_lat_long(self, location_name: str) -> Tuple[float, float]:
        """Get latitude and longitude from a location name."""
        API_URL = 'https://api.opencagedata.com/geocode/v1/json'   
        
        params = {
            'q': location_name,
            'key': self.api_key
        }
        
        response = requests.get(API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            lat = data['results'][0]['geometry']['lat']
            lon = data['results'][0]['geometry']['lng']
            return lat, lon
        else:
            raise Exception("Geocoding API request failed")

    def get_weather_forecast(self, location_name: str, start_date: str, end_date: str) -> str:
        """Get the weather forecast for a given location and date range."""
        try:
            latitude, longitude = self.get_lat_long(location_name)
            logger.info(f"latitude:{latitude}, longitude:{longitude}")
            
            base_url = 'https://api.open-meteo.com/v1/forecast'
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': start_date,
                'end_date': end_date,
                'daily': 'temperature_2m_min,temperature_2m_max,rain_sum,weather_code',
                'timezone': 'GMT'
            }
            
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                forecast_data = response.json()
                
                forecast = forecast_data['daily']
                summary = f"Weather forecast for {location_name} from {start_date} to {end_date}:"
                for date, temp_min, temp_max, rain, weather_code in zip(
                    forecast['time'], 
                    forecast['temperature_2m_min'], 
                    forecast['temperature_2m_max'], 
                    forecast['rain_sum'], 
                    forecast['weather_code']
                ):
                    summary += (
                        f"\nDate: {date}, Min Temp: {temp_min}°C, Max Temp: {temp_max}°C, "
                        f"Rain: {rain}mm, Weather Code: {weather_code}"
                    )
                return summary
            else:
                raise Exception("Open-Meteo API request failed")
        except Exception as e:
            return str(e)

class TravelPlanningSystem:
    """Multi-agent travel planning system using Semantic Kernel and Azure AI Foundry."""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.blob_manager = AzureBlobManager(connection_string=config["AZURE_STORAGE_CONNECTION_STRING"])
        self.weather_service = WeatherService(config["GEOCODING_API_KEY"])
        
        # Create credentials
        self.credential = DefaultAzureCredential()
    
    async def create_travel_planning_agents(self, client, current_date: str):
        """Create all the agents needed for travel planning."""
        
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
        
        class WeatherPlugin:
            def __init__(self, weather_service):
                self.weather_service = weather_service
                
            @kernel_function(
                description="Get weather forecast for a location",
                name="get_weather_forecast"
            )
            def get_weather_forecast(self, 
                                    location_name: Annotated[str, "The name of the location"],
                                    start_date: Annotated[str, "Start date in YYYY-MM-DD format"],
                                    end_date: Annotated[str, "End date in YYYY-MM-DD format"]) -> str:
                """Get the weather forecast for a given location and date range."""
                return self.weather_service.get_weather_forecast(location_name, start_date, end_date)

        # 創建插件實例
        weather_plugin = WeatherPlugin(self.weather_service)

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
            plugins=[weather_plugin]  # 在這裡添加插件
        )

        # 4. Travel Summary Agent (with code interpreter)
        code_interpreter = CodeInterpreterTool()
        summary_agent_definition = await client.agents.create_agent(
            model=self.config["MODEL_DEPLOYMENT_NAME"],
            name="TravelSummaryAgent",
            instructions="""
            You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan.
            You must ensure that the final plan is integrated and complete.
            若天氣不佳（如下雨、強風、酷暑），請更換為室內景點，
            若天氣良好，則可更換為戶外行程。
            行程內容需具體、合理，並配合使用者原本的動線與住宿位置，避免增加移動成本與轉乘。
            You can use the code interpreter tool to generate visual itineraries, maps, charts, or any other helpful visualizations.
            YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.
            """,
            tools=code_interpreter.definitions,
            tool_resources=code_interpreter.resources
        )
        summary_agent = AzureAIAgent(client=client, definition=summary_agent_definition)
        
        return {
            "planner": planner_agent,
            "routemaster": routemaster_agent,
            "weather_advisor": weather_advisor_agent,
            "summary": summary_agent
        }
    
    async def run_travel_planning(self, task_query: str, user_id: str = None) -> Tuple[str, List[str]]:
        """Run the travel planning using Azure AI Foundry."""
        
        if not user_id:
            user_id = str(uuid.uuid4())
        
        # Check or create thread
        thread_id = self.blob_manager.read_thread(user_id)
        current_date = datetime.now().strftime("%Y-%m-%d")
        file_urls = []
        
        # Create AI Project client
        async with AIProjectClient(
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
                    agents["weather_advisor"],
                    agents["summary"]
                ],
                termination_strategy=TravelPlanningTerminationStrategy(
                    agents=[agents["summary"]]
                )
            )
            
            try:
                # Add the task to the group chat
                await chat.add_chat_message(message=task_query)
                print(f"# {AuthorRole.USER}: '{task_query}'")
                
                # Run the chat
                async for content in chat.invoke():
                    print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
                    
                    # Handle tool calls for weather agent
                    try:
                        if content.name == "WeatherAdvisorAgent" and hasattr(content, 'tool_calls') and content.tool_calls:
                            for tool_call in content.tool_calls:
                                if tool_call.function.name == "get_weather_forecast":
                                    args = json.loads(tool_call.function.arguments)
                                    weather_result = self.weather_service.get_weather_forecast(
                                        location_name=args.get("location_name"),
                                        start_date=args.get("start_date"),
                                        end_date=args.get("end_date")
                                    )
                                    # Add tool result back to chat
                                    await chat.add_chat_message(
                                        message=weather_result,
                                        role="tool",
                                        tool_call_id=tool_call.id
                                    )
                    except AttributeError as e:
                        print(f"處理工具調用時出錯: {e}")
                    
                    # Handle code interpreter results
                    try:
                        if hasattr(content, 'file_path_annotations') and content.file_path_annotations:
                            for file_path_annotation in content.file_path_annotations:
                                file_ext = file_path_annotation.text.split('.')[-1]
                                data_bytes = b''.join(client.agents.get_file_content(
                                    file_id=file_path_annotation.file_path.file_id
                                ))
                                
                                file_name = os.path.basename(file_path_annotation.text)
                                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                                new_file_name = f"{timestamp}-{file_name}"
                                
                                image_url = self.blob_manager.upload_blob(
                                    data=data_bytes,
                                    file_name=new_file_name,
                                    content_type=self._get_mime_type(file_ext)
                                )
                                file_urls.append(image_url)
                    except AttributeError:
                        # 如果沒有file_path_annotations屬性，跳過這部分
                        pass
                    
                    # Store the final response
                    if content.name == "TravelSummaryAgent" and "TERMINATE" in content.content:
                        final_response = content.content.replace("TERMINATE", "").strip()
                        return final_response, file_urls
                
                # If we reach here without a final response
                if chat.history:
                    return chat.history[-1].content, file_urls
                else:
                    return "No travel plan could be generated.", file_urls
                    
            finally:
                # Cleanup
                await chat.reset()
                for agent in agents.values():
                    await client.agents.delete_agent(agent.id)
    
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