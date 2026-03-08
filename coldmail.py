import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ScrapeWebsiteTool

# Page config
st.set_page_config(page_title="Cold Email Generator", page_icon="📧", layout="wide")

# Disable CrewAI telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# API Key handling - Streamlit Cloud secrets or local .env
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

# Initialize LLM
try:
    llm = LLM(
        model="gemini/gemini-2.0-flash-exp",
        api_key=api_key
    )
except Exception as e:
    # Fallback to LiteLLM if native provider fails
    import litellm
    llm = LLM(
        model="gemini/gemini-1.5-flash",
        api_key=api_key
    )

# Agency services knowledge base
agency_services = """
1. SEO Optimization Service: Best for companies with good products but low traffic. We increase organic reach.
2. Custom Web Development: Best for companies with outdated, ugly or slow websites. We build modern React/Python sites.
3. AI Automation: Best for companies with manual, repetitive tasks. We build agents to save time.
"""

# Initialize scraper tool
scrape_tool = ScrapeWebsiteTool()

# Create agents
def create_agents():
    researcher = Agent(
        role='Business Intelligence Analyst',
        goal='Analyze the target company website and identify their core business and potential weaknesses.',
        backstory="You are an expert at analyzing businesses just by looking at their landing page. You look for what they do and where they might be struggling.",
        tools=[scrape_tool],
        verbose=False,
        allow_delegation=False,
        memory=False,
        llm=llm
    )

    strategist = Agent(
        role='Agency Strategist',
        goal='Match the target company needs with ONE of our agency services.',
        backstory=f"""You work for a top-tier digital agency.
        Your goal is to read the analysis of a prospect and decide which of OUR services to pitch.

        OUR SERVICES KNOWLEDGE BASE:

        {agency_services}

        You must pick the SINGLE best service for this specific client and explain why.""",
        verbose=False,
        memory=False,
        llm=llm
    )

    writer = Agent(
        role='Senior Sales Copywriter',
        goal='Write a personalized cold email that sounds human and professional.',
        backstory="""You write emails that get replies. You never sound robotic.
        You mention specific details found by the Researcher to prove we actually looked at their site.""",
        verbose=False,
        llm=llm
    )

    return researcher, strategist, writer

# Streamlit UI
st.title("📧 AI Cold Email Generator")
st.markdown("Generate personalized cold emails for any company using AI agents.")

# Input section
with st.container():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_url = st.text_input(
            "🌐 Company Website URL",
            placeholder="https://example.com",
            help="Enter the full URL of the company you want to target"
        )
    
    with col2:
        recipient_name = st.text_input(
            "👤 Recipient Name (optional)",
            placeholder="CEO",
            help="Who should the email be addressed to?"
        )

# Generate button
if st.button("🚀 Generate Cold Email", type="primary", use_container_width=True):
    if not target_url:
        st.error("⚠️ Please enter a company website URL")
    elif not target_url.startswith("http"):
        st.error("⚠️ URL must start with http:// or https://")
    else:
        with st.spinner("🤖 AI agents are analyzing the website and crafting your email..."):
            try:
                # Create agents
                researcher, strategist, writer = create_agents()
                
                # Create tasks
                task_analyze = Task(
                    description=f"Scrape the website {target_url}. Summarize what the company does and identify 1 key area where they could improve (e.g., design, traffic, automation).",
                    expected_output="A brief summary of the company and their potential pain points.",
                    agent=researcher
                )

                task_strategize = Task(
                    description="Based on the analysis, pick ONE service from our Agency Knowledge Base that solves their problem. Explain the match.",
                    expected_output="The selected service and the reasoning for the match.",
                    agent=strategist,
                    context=[task_analyze]
                )

                recipient = recipient_name if recipient_name else "the CEO"
                task_write = Task(
                    description=f"Draft a cold email to {recipient} of the target company. Pitch the selected service. Keep it under 150 words.",
                    expected_output="A professional cold email ready to send.",
                    agent=writer,
                    context=[task_analyze, task_strategize]
                )

                # Create crew and run
                sales_crew = Crew(
                    agents=[researcher, strategist, writer],
                    tasks=[task_analyze, task_strategize, task_write],
                    process=Process.sequential,
                    verbose=False
                )

                result = sales_crew.kickoff()

                # Display result
                st.success("✅ Cold email generated successfully!")
                
                with st.container():
                    st.markdown("### 📨 Your Cold Email")
                    st.markdown("---")
                    st.markdown(result.raw)
                    st.markdown("---")
                    
                    # Copy button
                    st.code(result.raw, language="text")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Tip: Make sure the website allows scraping and is accessible.")

# Footer
st.markdown("---")
st.markdown("🔒 Powered by CrewAI + Gemini | Deployed on Streamlit Cloud")
