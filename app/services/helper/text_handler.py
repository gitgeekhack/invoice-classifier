import asyncio
import datefinder
import regex as re
import spacy

"""
This module contains functions for handling text data to extract information. The functions are:

find_date(text): extracts a date from the given text using various techniques including regex patterns and spacy's named
entity recognition (NER). It returns a string in the format of "YYYY-MM-DD" or "YYYY-DD-MM".

find_money_or_cardinal(text): extracts a money amount or a cardinal number from the given text using spacy's NER.
It returns a string.

text_handle_func(text, label_id): a general function that takes in a text and a label_id as arguments.
The label_id is used to determine which specific function to call. If label_id is 0, it calls find_date function,
if it's 1, it calls find_money_or_cardinal function, and if it's 2, it simply returns the input text (merchant name).
It returns the extracted information as a string or None if no information is found.
"""

nlp = spacy.load('en_core_web_sm')


async def find_date(text):
    cleaned_text = re.sub(r'[a-z]:\s', ' ', text.strip(), flags=re.IGNORECASE)
    if list(datefinder.find_dates(cleaned_text)):
        matches = list(datefinder.find_dates(cleaned_text))
        for match in matches:
            date_str = match.strftime("%Y-%m-%d")
            captures = re.findall(r'\d+', text.strip())
            if int(re.findall(r'\d+', date_str)[2]) == int(captures[0]):
                return date_str
            else:
                date_str = match.strftime("%Y-%d-%m")
                return date_str
    else:
        loop = asyncio.get_running_loop()
        doc = await loop.run_in_executor(None, nlp, text.strip())
        for ent in doc.ents:
            if ent.label_ == "DATE":
                return ent.text
            else:
                inv_dt = []
                str_pattern = [
                    "[0-9]{2}/{1}[0-9]{2}/{1}[0-9]{4}",
                    "\\d{1,2}-(January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|October|Oct|November|Nov|December|Dec)-\\d{4}",
                    "\\d{4}-\\d{1,2}-\\d{1,2}",
                    "[0-9]{1,2}\\s(January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|October|Oct|November|Nov|December|Dec)\\s\\d{4}",
                    "\\d{1,2}-\\d{1,2}-\\d{2,4}"]
                for pattern in str_pattern:
                    for match in re.finditer(pattern, text):
                        inv_dt.append(match.group())
                if len(inv_dt) == 0:
                    pass
                else:
                    return inv_dt[0]


async def find_money_or_cardinal(text):
    loop = asyncio.get_running_loop()
    doc = await loop.run_in_executor(None, nlp, text.strip())
    for ent in doc.ents:
        if ent.label_ == 'MONEY' or ent.label_ == 'CARDINAL':
            return ent.text


async def text_handle_func(text, label_id):
    if len(text.strip()) == 0:
        pass
    else:
        if label_id == 2:
            return text.strip()
        elif label_id == 0:
            return await find_date(text)
        elif label_id == 1:
            return await find_money_or_cardinal(text)
