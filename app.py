import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

MAX_TOKENS = os.getenv("MAX_TOKENS")
if MAX_TOKENS is None:
    MAX_TOKENS = "8192"

MAX_TOKENS = int(MAX_TOKENS)

VERSION = "v1.2.0"

question_template = """
Answer the following questions as best you can.
Question: {input}
Answer: Respond in the language which above question is using.
"""

GROQ_MODELS = [
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "llama3-8b-8192",
]


async def predict(message, history, model_name: str, temperature, max_tokens):
    llm = ChatGroq(model_name=model_name, temperature=temperature,
                   streaming=True, max_tokens=max_tokens)
    langchain_history = []
    for human, ai in history:
        langchain_history.append(HumanMessage(content=human))
        langchain_history.append(AIMessage(content=ai))
    prompt_template = ChatPromptTemplate.from_messages(
        langchain_history + [("human", question_template)])

    chain = prompt_template | llm | StrOutputParser()
    msg = ''
    async for chunk in chain.astream({"input": message}):
        msg = msg + chunk
        yield msg


demo = gr.ChatInterface(
    fn=predict,
    description="A Gradio chatbot powered by GroqCloud and Langchain " + VERSION,
    additional_inputs_accordion=gr.Accordion(
        label="Parameters", render=False, open=False),
    additional_inputs=[
        gr.Dropdown(label="Model", choices=GROQ_MODELS, value=GROQ_MODELS[0]),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1,
                  value=0.6),
        gr.Number(label="Max Tokens", value=MAX_TOKENS,
                  step=1024, minimum=1024, maximum=32768),
    ],
    examples=[
        ["What is Decision Tree Regression"],
        ["Wite a love story with about 10000 words."],
        ["如何配置 Nginx 多域名和 SSL"],
        ["llm の事前トレーニングと微調整とは何ですか?またその違いは何ですか"],
    ],
    cache_examples=False,
    title="Qboot"
)
demo.launch()
