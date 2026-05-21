from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

texte = [
    "douaa is a software engineer with 5 years of experience and a salary of 100000 dollars",
]

embeddings_model = OpenAIEmbeddings()

vector_store = Chroma.from_texts(
    texts=texte,
    embedding=embeddings_model,
    collection_name="employee_info"
)

retriever = vector_store.as_retriever()

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="employee_info_retriever",
    description="use this tool to retrieve information about employees"
)


@tool
def get_employee_info(name: str):
    """
    Get information about a given employee (name, salary, seniority)
    """
    print("get_employee_info called")
    return {"name": name, "salary": 100000, "seniority": 5}


@tool
def send_email(email: str, subject: str, content: str):
    """
    Send an email to a given email address with a subject and content
    """
    print(f"Email sent to {email} with subject '{subject}' and content '{content}'")
    return f"Email sent successfully to {email}, {subject}, {content}"


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

tools = [get_employee_info, send_email, retriever_tool]

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

resp = agent_executor.invoke(
    {
        "input": "nom, prenom, diplomes du rihab assouli et les informations sur l'entreprise"
    }
)

print(resp["output"])