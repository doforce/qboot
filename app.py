import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage

PROMPTS = [
    ("Detect Language", """
Respond to the user's input language. If the user's input is in [Language 1], respond in [Language 1]. If the user's input is in [Language 2], respond in [Language 2]. If the user's input is in [Language 3], respond in [Language 3], and so on. Continue to adapt the response language to match the user's input language for each subsequent interaction."
Example:
Respond to the user's input language. If the user's input is in Chinese, respond in Chinese. If the user's input is in Spanish, respond in Spanish. If the user's input is in English, respond in English. If the user's input is in French, respond in French, and so on. Continue to adapt the response language to match the user's input language for each subsequent interaction.
""")
]


VERSION = "v1.1.1"

MAX_TOKENS = os.getenv("MAX_TOKENS")
if MAX_TOKENS is None:
    MAX_TOKENS = "8192"

MAX_TOKENS = int(MAX_TOKENS)

GROQ_MODELS = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
]


async def predict(message, history, model_name, temperature, max_tokens, prompt, custom_prompt: str):
    llm = ChatGroq(model_name=model_name, temperature=temperature,
                   streaming=True, max_tokens=max_tokens)
    langchain_history = []
    if custom_prompt.strip() != "":
        langchain_history.append(SystemMessage(content=custom_prompt))
    else:
        langchain_history.append(SystemMessage(content=prompt))
    for human, ai in history:
        langchain_history.append(HumanMessage(content=human))
        langchain_history.append(AIMessage(content=ai))
    langchain_history.append(HumanMessage(content=message))

    message = ""
    async for chunk in llm.astream(langchain_history):
        message = message + chunk.content
        yield message


demo = gr.ChatInterface(
    fn=predict,
    description=f"A Gradio chatbot powered by GroqCloud {VERSION}",
    additional_inputs_accordion=gr.Accordion(
        label="Parameters", render=False, open=False),
    additional_inputs=[
        gr.Dropdown(label="Model", choices=GROQ_MODELS, value=GROQ_MODELS[0]),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.1,
                  value=0.6),
        gr.Number(label="Max Tokens", value=2048,
                  step=1024, minimum=1024, maximum=MAX_TOKENS),
        gr.Dropdown(label="Prompt", choices=PROMPTS, value=PROMPTS[0][1]),
        gr.TextArea(label="Custom prompt",
                    placeholder="If leave it empty, use prompt above.")
    ],
    examples=[
        ["What is Decision Tree Regression?"],
        ["Wite a love story with about 10000 words."],
        ["Why should I care about fast inference?"],
        ["如何配置 Nginx 多域名和 SSL"],
    ],
    cache_examples=False,
    title="Qboot"
)
demo.launch()
