import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

nb = nbformat.v4.new_notebook()

# Add a cell with your code
code_cell = nbformat.v4.new_code_cell(source=f'import os\nprint(os.getcwd())')
nb.cells.append(code_cell)

# Execute the notebook
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
output_str, error_str = None, None

ep.preprocess(nb)
if nb.cells[0].outputs:  # Check if there are any outputs
    output = nb.cells[-1].outputs[0]

print(output)
# Repo path :: /home/seungyoun/llama_code_interpreter\n
