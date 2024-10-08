general = (
    "<|system|>You are an expert editor who corrects spelling, formatting and grammatical errors<|end|>\n"
    "<|user|> ** TASK**  \n"
    "1. Edit the following text for spelling and grammar mistakes:  '{text}'\n"
    "2. Remove any formatting errors such as extra spaces"
    "**IMPORTANT** use the following template to format your response **."
    "1. Edited Text: "
    "[ Your corrected text here ]"
    "2. Corrections: "
    "[ Make a numbered list of your corrections ]"
    "<|end|>\n"
    "<|assistant|>\n"
)
