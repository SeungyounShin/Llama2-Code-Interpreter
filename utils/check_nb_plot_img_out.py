import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
import base64
from io import BytesIO
import re 

def get_error_message(traceback_str):
    lines = traceback_str.split('\n')
    for line in lines:
        if 'Error:' in line:
            return line
    return None  # Return None if no error message is found


nb = nbformat.v4.new_notebook()

SITE_PKG_ERROR_PREFIX = 'File /usr/local/lib/python3.8/'

code_sample = """
import yfinance as yf
import matplotlib.pyplot as plt

# Get the data of the Tesla USD stock price
tsla = yf.Ticker("TSLA-USD")

# Get the historical prices for the last 3 months
tsla_hist = tsla.history(period="max", start="3 months ago")

# Plot the close prices
tsla_hist['Close'].plot(figsize=(16, 9))
plt.title('Tesla stock price last 3 months')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.show()
"""

# Add a cell with your code
code_cell = nbformat.v4.new_code_cell(source=code_sample)
nb.cells.append(code_cell)

# Execute the notebook
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
output_str, error_str = None, None

try:
    ep.preprocess(nb)
    if nb.cells[0].outputs:  # Check if there are any outputs
        for i,c in enumerate(nb.cells[-1].outputs):
            print(f'[{i+1}] : {c}')
                 
except CellExecutionError as e:
    error_str = e

    if error_str is not None:
        # Get the traceback, which is a list of strings, and join them into one string
        filtered_error_msg = error_str.__str__().split('An error occurred while executing the following cell')[-1].split("\n------------------\n")[-1]
        raw_error_msg = "".join(filtered_error_msg)
            
        # Remove escape sequences for colored text
        #print(raw_error_msg)
        ansi_escape = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
        error_msg = ansi_escape.sub('', raw_error_msg)

        error_msg_only_cell = error_msg.split(SITE_PKG_ERROR_PREFIX)
        for i,c in enumerate(error_msg_only_cell):
            if i ==0:
                print(f'[{i+1}]\n{c.strip()}\n---')
            if i==3:
                error_header = get_error_message(c)
                print(error_header)


        #error_msg = raw_error_msg.replace("\x1b[0m", "").replace("\x1b[0;31m", "").replace("\x1b[0;32m", "").replace("\x1b[1;32m", "").replace("\x1b[38;5;241m", "").replace("\x1b[38;5;28;01m", "").replace("\x1b[38;5;21m", "").replace("\x1b[38;5;28m", "").replace("\x1b[43m", "").replace("\x1b[49m", "").replace("\x1b[38;5;241;43m", "").replace("\x1b[39;49m", "").replace("\x1b[0;36m", "").replace("\x1b[0;39m", "")
        error_lines = error_msg.split("\n")
        
        # Only keep the lines up to (and including) the first line that contains 'Error' followed by a ':'
        error_lines = error_lines[:next(i for i, line in enumerate(error_lines) if 'Error:' in line) + 1]

        # Join the lines back into a single string
        error_msg = "\n".join(error_lines)