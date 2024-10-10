from smartdoc_utils import (
    model_output,
    model_output_1,
    model_input,
    model_input_1,
    model_input_2,
    model_output_2,
)

karen_system_prompt = "You are an expert editor who corrects spelling, formatting and grammatical errors\n"

karen_prompt = (
    "** TASK **\n"
    "\n------------------------------------------------------------\n"
    "Example 1."
    "\n------------------------------------------------------------\n"
    f"\nText: {model_input}\n"
    f"{model_output}"
    "\n------------------------------------------------------------\n"
    "Example 2."
    "\n------------------------------------------------------------\n"
    f"\nText: {model_input_1}\n"
    f"{model_output_1}"
    "\n------------------------------------------------------------\n"
    "Example 3."
    "\n------------------------------------------------------------\n"
    f"\nText: {model_input_2}\n"
    f"{model_output_2}"
    "\n------------------------------------------------------------\n"
    "Only focus on the text after this. \n"
    "1. Edit the following text for spelling and grammar mistakes: "
    "'{text}'\n"
    "2. Remove any formatting errors such as extra spaces. \n"
    "\n**IMPORTANT**\nUse the following template to format your response."
    "1. Edited Text: \n"
    "{ Your corrected text here, blank if nothing is edited }\n"
    "2. Corrections: \n"
    "{ Make a numbered list of your corrections, blank if no corrections are there }\n\n"
    "Do not make up your own template. Use the one provided above."
)

llama_system_prompt = (
    "You are an expert editor who corrects spelling, formatting and grammatical errors. "
    "**RULES**\n"
    "* Analyze the text before editing it. \n"
    "* Follow British English spelling and grammar rules. Do not use American English. \n"
    "* Fix spelling mistakes, punctuation errors, and grammatical errors. \n"
    "* Do not correct or expand any abbrieviations or acronyms you do not know about. \n"
    "* Do not rename any names or proper nouns. Capitalise the initials if "
    "needed, but do not rename names like aircraft names, base names, locations etc. \n"
    "* Do not assume anything, if you're confused, leave it as it is. \n"
    "* Do not add new information. Only refer to the text provided. "
    "For example, do not add dates or months "
    "if they are not filled. \n"
    "* Remove any formatting errors such as extra spaces. \n"
    "\n**IMPORTANT**\nUse the following template to format your response."
    "1. Edited Text: \n"
    "{ Your corrected text here, blank if nothing is edited }\n"
    "2. Corrections: \n"
    "{ Make a numbered list of your corrections, blank if no corrections are there }\n\n"
    "Do not make up your own template. Use the one provided above."
)

llama_prompt = (
    "Use the following examples as reference.\n"
    "\n------------------------------------------------------------\n"
    "Example 1."
    "\n------------------------------------------------------------\n"
    f"\nText: {model_input}\n"
    f"{model_output}"
    "\n------------------------------------------------------------\n"
    "Example 2."
    "\n------------------------------------------------------------\n"
    f"\nText: {model_input_1}\n"
    f"{model_output_1}"
    "\n------------------------------------------------------------\n"
    "Example 3."
    "\n------------------------------------------------------------\n"
    f"\nText: {model_input_2}\n"
    f"{model_output_2}"
    "\n------------------------------------------------------------\n"
    "Following is the text you have to edit: "
    "\n------------------------------------------------------------\n"
    "\n"
    "Text: '{text}'\n"
    "\n"
    ""
)
