from difflib import SequenceMatcher

def anti_spam(messages, threshold=0.8):
    to_remove = []
    for i in range(len(messages)):
        for j in range(i+1, len(messages)):
            if i != j:
                if SequenceMatcher(None, messages[i].content, messages[j].content).ratio() > threshold:
                    to_remove.append(j)
    to_remove = list(set(to_remove))
    messages = [messages[i] for i in range(len(messages)) if i not in to_remove]
    return messages, len(to_remove)

def standardize_punctuation(text):
    text = text.replace("’", "'")
    text = text.replace("`", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return text

def fix_trailing_quotes(text):
    num_quotes = text.count('"')
    if num_quotes % 2 == 0:
        return text
    else:
        return text + '"'

def cut_trailing_sentence(text):
    text = standardize_punctuation(text)
    last_punc = max(text.rfind("."), text.rfind("!"), text.rfind("?"), text.rfind(".\""), text.rfind("!\""), text.rfind("?\""), text.rfind(".\'"), text.rfind("!\'"), text.rfind("?\'"))
    if last_punc <= 0:
        last_punc = len(text) - 1
    et_token = text.find("<")
    if et_token > 0:
        last_punc = min(last_punc, et_token - 1)
    text = text[: last_punc + 1]
    text = fix_trailing_quotes(text)
    return text
