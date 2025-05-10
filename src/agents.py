"""
Fraud Suraksha - Specialized Agents
This module defines specialized agents for fraud detection and analysis.
"""

from crewai import Agent
from typing import Any, List
from crewai.tools import SerperDevTool
from langchain.tools import StructuredTool

def create_agents(llm: Any, tools: List[StructuredTool]):
    """
    Create specialized agents for fraud detection and analysis.
    
    Args:
        llm: The language model to use for the agents
        tools: Additional tools for the agents to use
        
    Returns:
        A dictionary containing the agents
    """
    # Initialize tools
    search_tool = SerperDevTool()
    
    # Create tools list with search tool and RAG tools
    researcher_tools = [search_tool] + tools
    analyzer_tools = tools  # Analyzer uses only RAG tools by default
    
    # Create specialized agents
    researcher = Agent(
        role="Fraud Research Specialist",
        goal="Find comprehensive information about potential fraud schemes and scams",
        backstory="""You are an expert researcher with deep knowledge of fraud patterns.
        Your expertise lies in analyzing user-reported suspicious activity and finding relevant
        information from both knowledge bases and online sources. You systematically gather
        details about potential scams, paying special attention to patterns, known schemes,
        and emerging threats. You are meticulous and thorough in your research.""",
        verbose=True,
        allow_delegation=True,
        tools=researcher_tools,
        llm=llm
    )
    
    analyzer = Agent(
        role="Fraud Pattern Analyzer",
        goal="Analyze patterns in suspicious activity to identify fraud risk",
        backstory="""You are a fraud analyst with years of experience identifying fraud patterns.
        Your specialty is recognizing subtle cues that indicate potential scams. You carefully
        review research findings and user reports to identify risk factors and assign a fraud
        risk score (Low/Medium/High). You also calculate a confidence level in your assessment.
        You are detail-oriented and methodical in your analysis, carefully considering all
        evidence before making an assessment.""",
        verbose=True,
        allow_delegation=True,
        tools=analyzer_tools,
        llm=llm
    )
    
    advisor = Agent(
        role="Fraud Prevention Advisor",
        goal="Provide clear, actionable advice to protect users from fraud",
        backstory="""You are a fraud prevention expert who specializes in creating clear,
        personalized advice for users. You take complex fraud analysis and translate it into
        specific, actionable steps that anyone can understand and follow. You ensure users
        understand the nature of the risk and exactly what steps to take next. You are
        empathetic and user-focused, always ensuring your advice is practical, non-alarmist,
        and tailored to the specific situation.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
    
    return {
        "researcher": researcher,
        "analyzer": analyzer,
        "advisor": advisor
    }