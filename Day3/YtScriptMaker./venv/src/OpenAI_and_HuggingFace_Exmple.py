import os
import streamlit as st
# from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain import LLMChain, HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv

load_dotenv()

# os.environ['OPENAI_API_KEY'] = ""


# Streamlit App Framework
st.title('YouTube Script Generator')
st.image('/home/talha/August Series/GuidedProjectsAugustSeries/Day 3/YtScriptMaker/.venv/images/Youtube.jpeg')
prompt = st.text_input('Enter your prompt here')

# Prompt Templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Give me a youtube video title on {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'Write me a YouTube video on this Title : {title}, while levaraging the following wikipedia research : {wikipedia_research}'
)


title_memory = ConversationBufferMemory(
    input_key = 'topic',
    memory_key = 'chat_history'
)

script_memory = ConversationBufferMemory(
    input_key = 'title',
    memory_key = 'chat_history'
)

#LLMS
# llm = OpenAI(temperature = 0.9)
llm = HuggingFaceHub(repo_id = "google/flan-t5-small")

title_chain = LLMChain(llm = llm,
                       prompt = title_template,
                       verbose = True,
                       output_key = 'title',
                       memory = title_memory)

script_chain = LLMChain(llm =llm,
                        prompt = script_template,
                        verbose = True,
                        output_key = 'script',
                        memory = script_memory)


wiki = WikipediaAPIWrapper()

# Print the script if there is an input prompt
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title = title,
                              wikipedia_research = wiki_research)

    st.write(title)
    st.write(script)
    st.write(wiki_research)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wiki Research'):
        st.info(wiki_research)
