"""
Fraud Suraksha - Task Definitions
This module defines the tasks for the fraud detection and analysis crew.
"""

from crewai import Task
from typing import Dict, Any, List

def create_tasks(agents: Dict[str, Any], context: Dict[str, Any] = None):
    """
    Create tasks for the fraud detection and analysis crew.
    
    Args:
        agents: Dictionary containing the agents
        context: Additional context for the tasks
        
    Returns:
        A list of tasks
    """
    # Extract agents
    researcher = agents["researcher"]
    analyzer = agents["analyzer"]
    advisor = agents["advisor"]
    
    # Create task context
    user_query = context.get("user_query", "") if context else ""
    chat_history = context.get("chat_history", "") if context else ""
    
    # Create tasks with detailed instructions
    research_task = Task(
        description=f"""Research the suspicious activity described by the user:
        
        User Query: {user_query}
        
        Previous Chat History: {chat_history}
        
        Your job is to:
        1. Search for similar scams and fraud patterns in our database
        2. Use web search to find additional information about this type of scam
        3. Look for technical details about how this specific scam or fraud works
        4. Identify common red flags and warning signs
        5. Find specific information about this particular case
        
        Be thorough and specific in your research, focusing on:
        - Known fraud patterns similar to this case
        - Technical details about how this fraud operates
        - Common red flags present in this scenario
        - Recent reports of similar scams
        
        Be specific and fact-based in your findings, avoiding speculation.
        """,
        expected_output="""Detailed research findings about the potential fraud, including:
        1. Similar known scams and their patterns
        2. Technical details of how this fraud typically operates
        3. Common red flags associated with this type of fraud
        4. Current status of similar scams (active/reported)
        5. Specific details relevant to this case
        
        Format your findings in a clear, organized manner with specific sections for each category above.
        """,
        agent=researcher
    )
    
    analysis_task = Task(
        description="""Analyze the research findings and user report to determine fraud risk level.
        
        Review the research findings and user report carefully to:
        1. Identify specific red flags present in this case
        2. Assess the severity of each red flag
        3. Detect patterns that match known fraud schemes
        4. Determine the overall fraud risk level
        5. Evaluate your confidence in this assessment
        
        Consider both the available evidence and any gaps in information that affect your confidence.
        """,
        expected_output="""Comprehensive fraud risk analysis containing:
        1. List of identified red flags, with severity rating for each
        2. Overall fraud risk score (Low/Medium/High)
        3. Confidence assessment (percentage) in your conclusion
        4. Specific evidence supporting your risk assessment
        5. Any information gaps that affect confidence level
        
        Format your analysis with clear sections and bullet points.
        """,
        agent=analyzer,
        context=[research_task]
    )
    
    advisory_task = Task(
        description="""Create personalized advice for the user based on the fraud analysis.
        
        Using the research findings and fraud analysis:
        1. Craft specific, actionable advice for this exact situation
        2. Provide clear steps the user should take to protect themselves
        3. Explain what specific actions to avoid
        4. Offer guidance on what to do if they've already engaged with the potential fraud
        5. Suggest preventative measures for the future
        
        Ensure your advice is practical, easy to follow, and tailored to the user's specific situation.
        """,
        expected_output="""Clear, actionable advice for the user with:
        1. A brief summary of the situation and risk level
        2. Specific, step-by-step actions they should take now
        3. Actions to avoid
        4. What to do if already victimized
        5. Preventative measures for the future
        
        Format your advice in a clear, reassuring tone with bullet points for actions.
        """,
        agent=advisor,
        context=[research_task, analysis_task]
    )
    
    return [research_task, analysis_task, advisory_task]