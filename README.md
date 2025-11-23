I used AI to develop a python script that answered questions found ina  csv file and provide a confidence level on the answer provided. I was able to find some factual question lists that I provided in two iterations for the script to respond to. While I found that it answered each question correctly in both requests, I found it interesting that the AI answered the questions correctly, but had a low confidence score for the majority of the questions. The answers should be simple to find for the AI and indicates that it cannot fully differenciate between what is factual and what is not. This highlights a risk in utilizing AI without human review or guidance. 

If I was a leader, I would certainly put rules in place to mitigate risks that come from using AI tools available. If leaning into AI use at higher levels, I would ensure there are the proper disclosures being communicated. 

While AI is a powerful tool for us to use, in its current state it needs to be supervised. 


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
