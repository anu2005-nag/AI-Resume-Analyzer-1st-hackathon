import os
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from prompts.prompt_reader import get_format_prompt
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=groq_key,
    temperature=0,
)


def format_resume_chain(resume):
    system_prompt = get_format_prompt()
    resume_json   = resume.model_dump_json(indent=2)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate the HTML resume for the following data:\n\n{resume_json}")
    ]
    llm.model_kwargs["max_tokens"] = 4096
    response   = llm.invoke(messages)
    html_output = response.content

    if not isinstance(html_output, str):
        html_output = str(html_output)

    if "<!DOCTYPE html>" in html_output:
        html_output = html_output[html_output.index("<!DOCTYPE html>"):]

    return html_output