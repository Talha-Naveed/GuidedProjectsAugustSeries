from gradio_client import Client
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper

# Streamlit App Framework
st.title('YouTube Script Generator by Talha')
st.image('images/Youtube.jpeg')
# st.sidebar.markdown("[![Title](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg) (https://www.linkedin.com/in/talhanaveedakhtar/)")
st.sidebar.markdown(
    """
    <a href="https://www.linkedin.com/in/talhanaveedakhtar/" target="_blank">
        <img src="https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg" width="50" height="50">
        <span style="font-size: 20px; color: #0072b1;">LinkedIn</span>
    </a>
    """,
    unsafe_allow_html=True
)
prompt = st.text_input('Enter your prompt here')






#llm
client = Client("Be-Bo/llama-3-chatbot_70b")

# wiki = WikipediaAPIWrapper()
try:
	if prompt:

		# wiki_research = wiki.run(prompt)

		script = client.predict(
			message=f"Write me a YouTube video on {prompt}",
			api_name="/chat"
		)

		st.write(script)

		with st.expander("Script History"):
			st.info()

		with st.expander("Wiki HIstory"):
			st.info(wiki_research)
except TypeError:
	pass