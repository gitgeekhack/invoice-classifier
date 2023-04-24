import spacy
import datefinder
import regex as re

nlp = spacy.load('en_core_web_sm')


def text_handle_func(text, label_id):
    captures = re.findall(r'\d+', text.strip())
    cleaned_text = re.sub(r'[a-z]:\s', ' ', text.strip(), flags=re.IGNORECASE)

    if len(text.strip()) == 0:
        pass
    else:
        if label_id == 2 :
            return text.strip()

        elif label_id == 0:
            if list(datefinder.find_dates(cleaned_text)):
                matches = list(datefinder.find_dates(cleaned_text))
                for match in matches:
                    date_str = match.strftime("%Y-%m-%d")
                    if int(re.findall(r'\d+', date_str)[2]) == int(captures[0]):
                        return date_str
                    else:
                        date_str = match.strftime("%Y-%d-%m")
                        return date_str
            else:
                # load data
                doc = nlp(text.strip())
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
                        if len(inv_dt) == 0:  # if yolo predicts a bounding box which contains only text (no digits)
                            pass
                        else:
                            return inv_dt[0]

        elif label_id == 1:
            doc = nlp(text.strip())
            for ent in doc.ents:
                if ent.label_ == 'MONEY' or ent.label_ == 'CARDINAL':
                    return ent.text
