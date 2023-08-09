<p align="center" width="100%">
<img src="/assets/logo2.png" alt="llama2 code interprerter icon" style="width: 200px; height:200px; display: block; margin: auto; border-radius: 50%;">
</p>


# Llama2 Code Interpreter

This project allows LLM to generate code, execute it, receive feedback, debug, and answer questions based on the whole process. It is designed to be intuitive and versatile, capable of dealing with multiple languages and frameworks.

[The purpose and direction of the project](https://github.com/SeungyounShin/Llama2-Code-Interpreter/wiki)

## 🌟 Key Features

- 🚀 **Generation and Execution of Code** : Tailor-made codes spun into existence and executed smoothly. 
- 🔎 **Interactive Debugging & Feedback** : Real-time engagement in debugging, accompanied by constructive feedback for an improved coding experience. 
- 🧠 **Dynamic Q&A System** : A revolutionary question-answering mechanism powered by cutting-edge code generation, execution, real-time feedback, and interactive debugging.
- 🌐 **Internet Access Support** : Comprehensive functionality, powered by full support for internet access. Make the world wide web your sandbox!

## Examples



---
<div align="center">

***Llama2 in Action***

<p align="center" width="100%">
<img src="/assets/president_code.gif" alt="example1_president_search_with_code" style="width: 600px; display: block; margin: auto; border-radius: 50%;">
</p>

</div>

In response to the query, <span style="color:blue">"Who is the current president of South Korea?"</span>, Llama2 generates Python code for web scraping. This code targets the official website of the **South Korean government**, specifically extracting information from the exact URL.


The script retrieves the HTML title tag which contains the current president's name. The output, <span style="color:green">`'대한민국 대통령 > 윤석열 대통령 > 취임사'`</span>, allows Llama2 to discern that the incumbent president is <span style="color:red">윤석열 (Yoon Suk-yeol)</span>.



## Installation

1. Clone the repository:
```bash
git clone https://github.com/SeungyounShin/Llama2-Code-Interpreter.git
```

2. Change directory:
```bash
cd Llama2-Code-Interpreter.git
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

I see, you want to include the part about setting the `LLAMA_CI_PATH` environment variable in the setup instructions. Here's how you might write it:

### Setup

**Set the `LLAMA_CI_PATH` environment variable:** This script requires the `LLAMA_CI_PATH` environment variable to be set to the directory that contains the relevant code. You can set it to the current directory like this:

```bash
export LLAMA_CI_PATH=$(pwd)
```

Please note that this setting is only valid for the current shell session. If you want to make it permanent, you can add it to your shell's startup file (like `.bashrc` or `.bash_profile`).

### Run App

To start interacting with Llama2 via the Gradio UI:

```bash
python3 chatbot.py --mode_path <your-model-path>
```

Replace `<your-model-path>` with the path to the model file you want to use. (Usally I recommend you to use chat-type model e.g. `meta-llama/Llama-2-13b-chat`)

Please let me know if you need help with a specific part of this setup process.

## Contributions

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/SeungyounShin/Llama2-Code-Interpreter/issues). 

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Seungyoun, Shin - 2022021568@korea.ac.kr

## Acknowledgement

Here are some relevant and related projects that have contributed to the development of this work:

1. **llama2** : [GitHub Repository](https://github.com/facebookresearch/llama)
2. **yet-another-gpt-tutorial** : [GitHub Repository](https://github.com/sjchoi86/yet-another-gpt-tutorial/tree/main)

These projects have been instrumental in providing valuable insights and resources, and their contributions are highly appreciated.

---