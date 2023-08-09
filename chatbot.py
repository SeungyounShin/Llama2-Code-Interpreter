import gradio as gr
import random
import time
import argparse

from code_interpreter.LlamaCodeInterpreter import LlamaCodeInterpreter
from code_interpreter.GPTCodeInterpreter import GPTCodeInterpreter
from utils.const import *

LLAMA2_MODEL_PATH = "./ckpt/llama-2-13b-chat"

def load_model(model_path):
    print('++ Loading Model')
    return LlamaCodeInterpreter(model_path, load_in_4bit=True)
    #return GPTCodeInterpreter()

def main(model_path):
    # Create an instance of your custom interpreter
    code_interpreter = load_model(model_path)

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])

        def respond(message, chat_history):
            bot_message = code_interpreter.chat(message, VERBOSE=True)['content']
            chat_history.append((message, bot_message))
            time.sleep(2)
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

    demo.launch()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatbot")
    parser.add_argument("--model_path", type=str, default=LLAMA2_MODEL_PATH,
                        help="Path to the model. Default is './ckpt/llama-2-13b-chat'.")
    args = parser.parse_args()

    main(args.model_path)
