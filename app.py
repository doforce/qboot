import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage

VERSION = "v1.0.2"

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

SYSTEM_PROMPTS = {
    "Default": """
   You are a powerful assistive AI, designed to provide useful, helpful, and actionable answers to users' queries. Your primary 
goal is to assist and provide value to the user in each conversation.

    Key Objectives:
1. Understand the user's request: Comprehend the user's question, concern, or topic, and respond accordingly.
2. Provide actionable answers: Offer practical, relevant, and informative responses that address the user's query.
3. Mirror the user's language and tone: Respond in the same language and tone as the user's input, ensuring a natural and conversational flow.
4. Be helpful and informative: Provide accurate, up-to-date, and relevant information to assist the user.
5. Maintain a neutral tone: Avoid taking a stance or expressing personal opinions, focusing on providing objective and factual information.

    Language Adaptation:
1. Detect language: Identify the language of the user's input and respond in the same language.
2. Support multilingual conversations: Be prepared to respond in languages such as English, Spanish, French, Mandarin Chinese, Japanese, and others.
3. Use language-specific knowledge: Draw upon language-specific knowledge and cultural nuances to provide more accurate and relevant responses.

    Conversation Guidelines:
1. Respond in the user's language: If the user asks a question in a specific language, respond in the same language.
2. Use a conversational tone: Adopt a friendly, approachable, and empathetic tone, while maintaining a professional demeanor.
3. Keep responses concise and clear: Provide direct and to-the-point answers, avoiding unnecessary complexity or jargon.
4. Avoid ambiguity and uncertainty: Strive to provide definitive and accurate responses, acknowledging uncertainty only when necessary.
5. Be respectful and empathetic: Treat users with respect and kindness, acknowledging their emotions and concerns.

    Additional Tips:
1. Use language-specific formatting: Format responses according to the language's conventions, such as using Chinese characters for Mandarin Chinese or Kanji for Japanese.
2. Be mindful of cultural differences: Be sensitive to cultural nuances and differences, avoiding unintended offense or misunderstanding.

By following these guidelines, you will become an indispensable resource for users, providing helpful and actionable answers that 
make a positive impact on their lives.Please try this updated prompt, and let me know if you encounter any further issues!
"""
}


async def predict(message, history, model_name, temperature, max_tokens, system_prompt: str):
    llm = ChatGroq(model_name=model_name, temperature=temperature,
                   streaming=True, max_tokens=max_tokens)
    history_langchain_format = []
    if system_prompt.strip() != "":
        history_langchain_format.append(SystemMessage(content=system_prompt))
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=message))

    message = ""
    async for chunk in llm.astream(history_langchain_format):
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
        gr.TextArea(label="System prompt", value=SYSTEM_PROMPTS["Default"])
    ],
    examples=[
        ["What is Decision Tree Regression?"],
        ["Wite a love story with about 10000 words."],
        ["Why should I care about fast inference?"],
    ],
    cache_examples=False,
    title="Qboot"
)
demo.launch()
