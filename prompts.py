system_prompt = (
    "You are an expert editor which can fix spelling, formatting and grammatical errors."
    "You will be provided with text to edit. \n"
    "**IMPORTANT** \nUse the following template to format your response. \n"
    "1. Edited Text: "
    "[ Corrected text here ]\n"
    "2. Corrections: "
    "[ Make a numbered list of corrections ]\n"
)

general_prompt = (
    "**TASK**  \n"
    "1. Edit the following text for spelling and grammar mistakes:  '{text}'\n"
    "2. Remove any formatting errors such as extra spaces"
)

grammar_checker_prompt = (
    "**TASK**  \n"
    "Edit the following text for spelling and grammatical errors. "
    "If there are no corrections return the text as it is with "
    "the `Corrections` section empty. \n"
    "##################################################\n"
    "Text: '{text}'\n"
    "##################################################\n"
)
