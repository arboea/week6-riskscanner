# Answer Questions CLI

Small Python utility to read questions from a file and produce answers.

Supported input formats:
- CSV: will use a column named `question` (case-insensitive) if present, otherwise the first column.
- XLSX/XLS: same behavior as CSV.
- TXT: each non-empty line is a question.

Quick start (Windows PowerShell):

1. Install dependencies (recommended in a virtualenv):

```powershell
python -m pip install -r requirements.txt
```

2. Set your OpenAI key (if using OpenAI backend):

```powershell
$env:OPENAI_API_KEY = 'sk-...'
```

3. Run the script:

```powershell
python .\answer_questions.py examples/questions.csv --backend openai --model gpt-3.5-turbo --out answers.csv

# or use echo mode (no API key needed)
python .\answer_questions.py examples/questions.txt --backend echo --out answers-echo.csv
```

Output: CSV file with `question` and `answer` columns.

Notes:
- For CSV/XLSX inputs install `pandas` and `openpyxl` (provided in `requirements.txt`).
- If you want another LLM backend, the script is organized so you can add a new `answer_with_<backend>` function.
