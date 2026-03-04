import streamlit as st
import uuid
from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace,HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
import requests
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.tools import tool
import json
from dotenv import load_dotenv
import os

API_KEY=os.getenv(key='EXCHANGE_API_KEY')

load_dotenv()

system = SystemMessage(
    content="""
You are a currency assistant.

Use tools ONLY when the user asks for currency conversion.

If the user says hello, hi, or asks general questions,
reply normally and DO NOT call any tool.

MOST IMPORTANT-: and first use get_conversion_factor tool then only use convert tool as first fetch the conversion rate for base currency to target then convert base currency to target 
by using conversion rate
"""
)


llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",

    temperature=0.4,
    max_new_tokens=256

)

llm=ChatHuggingFace(llm=llm)

@tool
def get_conversion_factor(base_currency: str,target_currency:str)-> dict:
  """Get the currency conversion rate between two currencies like USD to INR."""
  response=requests.get(f"https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{base_currency}/{target_currency}")
  return response.json()

# yaha niche main conversion rate injected tool arg m kyon aanotate ya toh video dekh le 44:00 se satrt hoke
# ya sun jaise pehle get conversion and convert ko model ne call but model as usse currecy val toh 10 mil gayi humari query se
#  but conversion rate thodi mila hoga nechere ko so vo apne hisab se funct m ek no behj dega jo sahi ni so injected usse bata ki
#  iss para ko set mat karna bas ye hi
@tool
def convert(base_currency_val:int,conversion_rate:Annotated[float,InjectedToolArg])->float:
  """Multiply currency value with conversion rate"""

  return base_currency_val*conversion_rate



llm_with_tools=llm.bind_tools([get_conversion_factor,convert])




def tooli(ai_msg,message):

    convert_rate=None

    for tool_call in ai_msg.tool_calls:

        if tool_call["name"]=="get_conversion_factor":

            result = get_conversion_factor.invoke(tool_call["args"])

            if isinstance(result, str):
                result = json.loads(result)

            convert_rate = result["conversion_rate"]

            message.append(
                SystemMessage(content=f"Conversion rate is {convert_rate}")
            )
            print(f"********conversion rate{convert_rate}")


        if tool_call["name"]=="convert":

            if convert_rate is None:
                continue

            tool_call["args"]["conversion_rate"] = convert_rate

            result = convert.invoke(tool_call["args"])

            message.append(
                SystemMessage(content=f"Converted value is {result}")
            )





# chats={
# chat_id=[
# {}
# ]
# 
# }

st.set_page_config(page_title="currency",layout="wide")


if "chats" not in st.session_state:
    st.session_state.chats={}

if "current_session_id" not in st.session_state:
    id=str(uuid.uuid4())
    st.session_state.current_session_id=id
    st.session_state.chats[id]=[]

if "msg_history" not in st.session_state:
    st.session_state.msg_history={}
    st.session_state.msg_history[st.session_state.current_session_id]=[system.content]
     




#sidebar

if st.sidebar.button("New Chat"):
    id=str(uuid.uuid4())
    st.session_state.current_session_id=id
    st.session_state.chats[id]=[]
    st.session_state.msg_history[id]=[system.content]

for chat_id in st.session_state.chats.keys():
    if st.sidebar.button(f"Chat {chat_id[:8]}",key=chat_id):
        st.session_state.current_session_id=chat_id



#main

curr=st.session_state.current_session_id
message=st.session_state.msg_history[curr]
for msg in st.session_state.chats[curr]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#input
prompt=st.chat_input("type the message")

if prompt:
    message.append(HumanMessage(content=prompt))
    st.session_state.chats[curr].append({
        "role":"user",
        "content":prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)
    ai_msg=llm_with_tools.invoke(message)

    message.append(ai_msg)
    print(ai_msg)
 
    if ai_msg.tool_calls:
        print(f"*******tool call")
        tooli(ai_msg,message)
        print(message)
        ai_msg=llm_with_tools.invoke(message)
        print(ai_msg)

    st.session_state.chats[curr].append({
        "role":"assistant",
        "content":ai_msg.content
    })
    with st.chat_message("assistant"):
        st.markdown(ai_msg.content)