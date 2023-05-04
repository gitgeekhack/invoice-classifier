"""
The TextHandler class provides methods to handle and extract information from given text.

Attributes:
nlp (object): An instance of spacy language model for natural language processing.

Methods:
find_date(text: str) -> str:
    Given a string of text, this method finds and returns the date in the text. If the date is not found,
    this method returns None.

find_money_or_cardinal(text: str) -> str:
    Given a string of text, this method finds and returns the amount mentioned in the text. If the amount is not found,
    this method returns None.

text_handle_func(text: str, label_id: int) -> str:
    Given a string of text and a label ID, this method returns the corresponding text for the given label ID:
    label_id 0: date
    label_id 1: amount
    label_id 2: other text
    If the label ID is not valid, this method returns None.
"""

import datefinder
import regex as re
import spacy
from app.constants import Regex, Labels


class TextHandler:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    async def find_date(self, text):
        cleaned_text = re.sub(Regex.CLEANED_TEXT, ' ', text.strip(), flags=re.IGNORECASE)
        if list(datefinder.find_dates(cleaned_text)):
            matches = list(datefinder.find_dates(cleaned_text))
            for match in matches:
                date_str = match.strftime(Regex.DATE_FORMAT)
                captures = re.findall(Regex.CAPTURES_PATTERN, text.strip())
                if int(re.findall(Regex.CAPTURES_PATTERN, date_str)[2]) == int(captures[0]):
                    return date_str
                else:
                    date_str = match.strftime(Regex.DATE_FORMAT)
                    return date_str
        else:
            doc = self.nlp(text.strip())
            for ent in doc.ents:
                if ent.label_ == Labels.DATE_LABEL:
                    return ent.text
                else:
                    inv_dt = []
                    str_pattern = Regex.PATTERN
                    for pattern in str_pattern:
                        for match in re.finditer(pattern, text):
                            inv_dt.append(match.group())
                    if len(inv_dt) == 0:
                        pass
                    else:
                        return inv_dt[0]

    async def find_money_or_cardinal(self, text):
        doc = self.nlp(text.strip())
        for ent in doc.ents:
            if ent.label_ in Labels.AMOUNT_LABEL:
                return ent.text

    async def text_handle_func(self, text, label_id):
        if len(text.strip()) == 0:
            pass
        else:
            if label_id == 2:
                return text.strip()
            elif label_id == 0:
                return await self.find_date(text)
            elif label_id == 1:
                return await self.find_money_or_cardinal(text)

