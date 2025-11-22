#!/usr/bin/env python3
"""
answer_questions.py

Reads questions from an input file (.csv, .xlsx, .txt) and produces answers.

Usage examples (PowerShell):
  $env:OPENAI_API_KEY = 'sk-...'
  python .\answer_questions.py examples/questions.csv --backend openai --model gpt-3.5-turbo

Or use echo mode which returns a placeholder answer:
  python .\answer_questions.py examples/questions.txt --backend echo

"""
import os
import argparse
import sys
import time
import logging
from typing import List, Dict

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import openai
except Exception:
    openai = None

logging.basicConfig(level=logging.INFO, format="%(message)s")


def read_questions(path: str) -> List[str]:
    path = os.path.abspath(path)
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext in ('.csv', '.txt', '.xlsx') and pd is None:
        raise RuntimeError('pandas is required to read csv/xlsx files. Please install requirements.')

    if ext == '.csv':
        df = pd.read_csv(path)
        if 'question' in df.columns.str.lower():
            # try case-insensitive match
            col = [c for c in df.columns if c.lower() == 'question'][0]
            return [str(x) for x in df[col].dropna().tolist()]
        # fallback to first column
        first_col = df.columns[0]
        return [str(x) for x in df[first_col].dropna().tolist()]

    if ext in ('.xls', '.xlsx'):
        df = pd.read_excel(path)
        if 'question' in df.columns.str.lower():
            col = [c for c in df.columns if c.lower() == 'question'][0]
            return [str(x) for x in df[col].dropna().tolist()]
        first_col = df.columns[0]
        return [str(x) for x in df[first_col].dropna().tolist()]

    if ext == '.txt':
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
        return [l for l in lines if l]

    raise ValueError(f'Unsupported extension: {ext}')


def answer_with_openai(question: str, model: str = 'gpt-3.5-turbo', api_key: str = None) -> str:
    if openai is None:
        raise RuntimeError('openai package not installed. Add it to requirements and install.')
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set in environment')

    openai.api_key = api_key
    # Use ChatCompletion API
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.2,
            max_tokens=1024,
        )
        return resp['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.warning(f'OpenAI request failed: {e}')
        raise


def answer_with_echo(question: str) -> str:
    return f"[echo answer] {question}"


def write_answers(answers: List[Dict], out_path: str):
    # write CSV via pandas if available, else simple CSV writer
    if pd is not None:
        df = pd.DataFrame(answers)
        df.to_csv(out_path, index=False)
        return

    import csv
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'answer'])
        writer.writeheader()
        writer.writerows(answers)


def main():
    parser = argparse.ArgumentParser(description='Answer questions from a file (.csv/.xlsx/.txt).')
    parser.add_argument('input', help='Input file path (.csv, .xlsx, .txt)')
    parser.add_argument('--backend', choices=['openai', 'echo'], default='openai', help='Answer backend')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='OpenAI model to use')
    parser.add_argument('--out', default='answers.csv', help='Output CSV file path')
    parser.add_argument('--apikey', default=None, help='OpenAI API key (optional; otherwise use OPENAI_API_KEY env var)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between requests to avoid rate limits')
    args = parser.parse_args()

    try:
        questions = read_questions(args.input)
    except Exception as e:
        logging.error(f'Failed to read questions: {e}')
        sys.exit(2)

    logging.info(f'Read {len(questions)} questions from {args.input}')

    answers = []
    for i, q in enumerate(questions, start=1):
        logging.info(f'[{i}/{len(questions)}] Question: {q[:80]}')
        try:
            if args.backend == 'openai':
                ans = answer_with_openai(q, model=args.model, api_key=args.apikey)
            else:
                ans = answer_with_echo(q)
        except Exception as e:
            ans = f'ERROR: {e}'
        answers.append({'question': q, 'answer': ans})
        time.sleep(args.delay)

    try:
        write_answers(answers, args.out)
        logging.info(f'Wrote answers to {args.out}')
    except Exception as e:
        logging.error(f'Failed to write answers: {e}')
        sys.exit(3)


if __name__ == '__main__':
    main()
