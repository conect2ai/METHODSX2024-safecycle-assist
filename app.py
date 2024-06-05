import os
import sys

import numpy as np
import pandas as pd

import osmnx as ox
import networkx as nx

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool

from typing import Type
from pydantic import BaseModel, Field


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Raio da Terra em quilômetros

    return c * r

def find_nearest_risk_lane(lat, lon, data):
    distances = data.apply(lambda row: haversine(lat, lon, row['lat'], row['lon']), axis=1)

    nearest_index = distances.idxmin()
    
    return data.loc[nearest_index, 'RL']


class RiskPointInput(BaseModel):
    lat: float = Field(..., title="Latitude", description="Latitude of the point")
    lon: float = Field(..., title="Longitude", description="Longitude of the point")

class RiskTripInput(BaseModel):
    start_lat: float = Field(..., title="Latitude of the starting point", description="Latitude of the starting point")
    start_lon: float = Field(..., title="Longitude of the starting point", description="Longitude of the starting point")
    end_lat: float = Field(..., title="Latitude of the ending point", description="Latitude of the ending point")
    end_lon: float = Field(..., title="Longitude of the ending point", description="Longitude of the ending point")

class RiskPointTool(BaseTool):
    name = "RiskPointTool"
    description = "A tool that computes the risk of a point in the map"
    args_schema: Type[BaseModel] = RiskPointInput

    def _run(
        self,
        lat: float,
        lon: float
    ) -> str:
        data = pd.read_csv('risk_map.csv', sep=";")
        risk_lane = find_nearest_risk_lane(lat, lon, data)

        return risk_lane

class RiskTripTool(BaseTool):
    name = "RiskTripTool"
    description = "A tool that computes the risk of a trip in the map"
    args_schema: Type[BaseModel] = RiskTripInput

    def _run(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float
    ) -> str:
        data = pd.read_csv('risk_map.csv', sep=";")
        place_name = "Porto, Portugal"
        graph = ox.graph_from_place(place_name, network_type='all')

        start = (start_lat, start_lon)
        end = (end_lat, end_lon)

        start_node = ox.distance.nearest_nodes(graph, start[1], start[0])
        end_node = ox.distance.nearest_nodes(graph, end[1], end[0])

        route = nx.shortest_path(graph, start_node, end_node, weight='length')

        risk = {
            1: 0,
            2: 0,
            3: 0
        }

        for node in route:
            node_data = graph.nodes[node]
            risk_lane = find_nearest_risk_lane(node_data['y'], node_data['x'], data)
            risk[risk_lane] += 1

        sum_risk = risk[1] + risk[2] + risk[3]

        return f"Low risk percentage: {risk[1]/sum_risk*100:.2f}%, Medium risk percentage: {risk[2]/sum_risk*100:.2f}%, High risk percentage: {risk[3]/sum_risk*100:.2f}%"

openai_api_key = os.getenv("OPENAI_API_KEY")

chat_open_ai = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, openai_api_key=openai_api_key)

risk_point_tool = RiskPointTool()
risk_trip_tool = RiskTripTool()

tools = [
    risk_point_tool,
    risk_trip_tool
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an agent responsible for assisting users in computing cycling routes based on risk levels. Your task is to analyze the computed risk levels and provide comprehensive and well-tailored responses to user inquiries.

            Risk Levels:
            - (1) Low Risk: Routes that are safe for most cyclists, including beginners. These routes are well within the city's emergency mitigation capacity, with close proximity to emergency services.
            - (2) Medium Risk: Routes that may have moderate distance from emergency services. These routes are suitable for experienced cyclists who are comfortable with occasional risks and may require emergency services less readily available.
            - (3) High Risk: Routes that are far from emergency services and other emergency-related amenities. These routes involve significant hazards and are recommended only for very experienced cyclists or those seeking a challenge.

            Main Instructions:
            1. Input Interpretation:
            - Receive the computed risk level (1, 2, or 3) from the risk analysis system.
            - Understand the context and specifics of each risk level as detailed above, focusing on the city's emergency mitigation capacity.

            2. Response Generation:
            - Provide clear and detailed descriptions of the route based on the given risk level.
            - Include relevant safety tips and precautions specific to the risk level.
            - Suggest the user to look for safer alternative routes if the computed risk level is considered high (the route passes througout many high-risk areas).

            3. Example Responses:
            - Low Risk (1): "The route from point A to point B is considered low risk, making it ideal for beginners. The path is within close proximity to emergency services. Ensure you stay on designated cycling lanes for maximum safety."
            - Medium Risk (2): "The route from point A to point B has a medium risk level, suitable for experienced cyclists. While not immediately near emergency services, they are accessible with some delay. Be cautious at intersections and consider wearing high-visibility gear."
            - High Risk (3): "The route from point A to point B is high risk, recommended only for very experienced cyclists. The path includes heavy traffic and is far from emergency services. For a safer alternative, consider another route and provide the information that I will assist you in evaluating it to assure you stay safe."

            4. General Tips:
            - Always remind users to wear appropriate safety gear, such as helmets and reflective clothing - reinforce additional gear if the route has high risk.
            - Encourage users to check weather conditions before starting their journey.
            - Advise users to carry a basic repair kit and stay hydrated.
            - Suggest carrying a mobile phone and sharing route details with someone in case of an emergency.

            By following these guidelines, you will ensure that users receive precise, helpful, and contextually appropriate information for their cycling routes. Your goal is to enhance their cycling experience while prioritizing their safety based on the city's emergency mitigation capacity.
            """
        ),
        (
            "user",
            "{input}"
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm_with_tools = chat_open_ai.bind_tools(tools)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

user_input = sys.argv[sys.argv.index("--prompt") + 1]

result = list(
    agent_executor.stream(
        {"input": user_input}
    )
)

print(result[-1]["messages"][0].content)
