import re

model_input = """
'I saw a    sample prodcut.  These product are below expeccted quality. I start to underdstand what he said is quite right. The correct valu is 12.5%% and 3rd postion. The time is 10:30am. The length of the object is 845mm and costs $1million. The sunise happens at 0750 EST and visibilty is 35 km. The right way too sumize and favor some advisor's is still to be found.'"""
# The text generated by the KAREN model
model_output = """
1. Edited Text: 'I saw a sample product. These products are below expected quality. I started to understand what he said is quite right. The correct value is 12.5%% and 3rd position. The time is 10:30 am. The length of the object is 845mm and costs $1m. The sunrise happens at 0750 EST and visibility is 35 km. The right way to summarize and favor some advisors' is still to be found.'
2. Corrections:
   a) Removed extra spaces after 'I saw' and before 'These products are below expected quality.'
   b) Changed 'sample prodcut' to 'sample product.'
   c) Corrected the tense from 'start' to 'started'
   d) Corrected the spelling of 'expeccted quality.' to 'expected quality.'
   e) Fixed the spelling "correct valu" to 'correct value.'
   f) Fixed the time format to '10:30 am.'
   g) Abbreviated '$1million' to '$1m.'
   h) Corrected 'sunise' to 'sunrise.'
   i) Fixed the spelling of'visibilty' to 'visibility.'
   j) Corrected 'too sumize' to 'to summarize.'
   k) Fixed punctuation in 'advisor's' to 'advisors'.
"""

model_input_1 = """
'I read a article about new technology. These technologie are not very impresssive. I think the points he made are mostly valid. The value should be 15.3%% and in 2nd postion. The time is 9:45am. The object's width is 732cm and priced at $500,000. The moonrise occured at 0635 UTC and the visibility range is 40 miles. The correct strategy for optimizing results and consulting a specialist's insights remains unclear.'
"""
model_output_1 = """
1. Edited Text: 'I read an article about new technology. These technologies are not very impressive. I think the points he made are mostly valid. The value should be 15.3%% and in 2nd position. The time is 9:45 am. The object's width is 732cm and priced at $500,000. The moonrise occurred at 0635 UTC and the visibility range is 40 miles. The correct strategy for optimizing results and consulting a specialist's insights remains unclear.'

2. Corrections: 
   a) Changed 'I read a article' to 'I read an article.' 
   b) Corrected spelling of 'technologie' to 'technologies.' 
   c) Fixed spelling of 'impresssive' to 'impressive.' 
   d) Corrected spelling of 'postion' to 'position.' 
   e) Changed the time format to '9:45 am.'  
   f) Updated 'priced at $500,000' for clarity and consistency. 
   g) Corrected spelling of 'occured' to 'occurred.' 
   h) Replaced 'visibility is 40 miles' to 'visibility range is 40 miles.' 
   i) Clarified 'correct strategy for optimizing results and consulting a specialist's insights remains unclear.'
"""

model_input_2 = """
'I have a book called "The Sun Also Rises". The book is about a group of American and British expatriates who travel from Paris to Pamplona to watch the running of the bulls and the bullfights. The book was published in 1926 and was written by Ernest Hemingway. The main characters are Jake Barnes, Lady Brett Ashley, Robert Cohn, and Pedro Romero. The book is considered one of Hemingway's masterpieces and is a classic of American literature.'
"""

model_output_2 = """
1. Edited Text: 

2. Corrections:
"""


def document_postprocessing(model_output):
    """
    Process the output from a language model to extract edited text and corrections.

    This function takes the raw output from a language model that has performed
    proofreading and editing tasks. It extracts the edited version of the text
    and a list of specific corrections made.

    Parameters:
    model_output (str): The raw string output from the language model.
                        Expected to contain sections for "Edited Text" and
                        itemized corrections.

    Returns:
    tuple: A tuple containing two elements:
           - edited_text (str): The extracted edited version of the text.
                                If no edited text is found, returns "No edited text found."
           - corrections (list): A list of strings, each representing a specific
                                 correction made by the model.

    Prints:
    - The extracted edited text.
    - A numbered list of corrections.

    Note:
    The function assumes a specific format in the model_output:
    - Edited text is enclosed in single quotes after "1. Edited Text:"
    - Corrections are listed with lowercase letters followed by parentheses.

    If the expected format is not found, the function may not extract information correctly.
    """

    # Extract Edited Text
    edited_text_match = re.search(r"Edited Text: '([^']*)'", model_output)
    edited_text = (
        edited_text_match.group(1) if edited_text_match else "No edited text found."
    )

    # Extract Corrections
    corrections_match = re.findall(r"(\b[a-z]\))(.+)", model_output)
    corrections = [correction.strip() for _, correction in corrections_match]

    # Print the results
    print("Edited Text:")
    print(edited_text)
    print("\nCorrections:")
    for i, correction in enumerate(corrections, 1):
        print(f"{i}. {correction}")

    return edited_text, corrections


# def process_llm_output(llm_output):
#     """
#     Process the output from a custom Large Language Model and extract
#     the edited text and corrections.

#     Parameters:
#     llm_output (str): The raw output string from the LLM.

#     Returns:
#     tuple: A tuple containing two elements:
#            - edited_text (str): The extracted edited text.
#            - corrections (str): The extracted corrections, or "None needed." if no corrections.
#     """
#     # Extract Edited Text
#     edited_text_match = re.search(r"1\.\s*Edited Text:\s*'?(.+?)'?\s*(?=2\.|$)", llm_output, re.DOTALL)
#     edited_text = edited_text_match.group(1).strip() if edited_text_match else "No edited text found."

#     # Extract Corrections
#     corrections_match = re.search(r"2\.\s*Corrections:\s*(.+)$", llm_output, re.DOTALL)
#     if corrections_match:
#         corrections = corrections_match.group(1).strip()
#         # If corrections start with "[Your corrected text here]", remove it
#         corrections = re.sub(r"^\[Your corrected text here\]\s*-?\s*", "", corrections)
#         # Format each correction on a new line
#         corrections = re.sub(r"(?:^|\s+)-\s*", "\n- ", corrections).strip()
#     else:
#         corrections = "None needed."

#     return edited_text, corrections


def process_llm_output(response):
    """
    Process the output from a custom Large Language Model and extract
    the edited text and corrections.

    Parameters:
    llm_output (str): The raw output string from the LLM.

    Returns:
    tuple: A tuple containing two elements:
           - edited_text (str): The extracted edited text.
           - corrections (str): The extracted corrections, or "None needed." if no corrections.
    """
    # Extract Edited Text
    # Modify regex to capture edited text with or without quotes
    # edited_text_match = re.search(r"1\. Edited Text:\s*(.*)", response)
    edited_text_match = re.search(
        r"1\. Edited Text:\s*['\"]?(.*?)['\"]?\s*(?:\n|$)", response, re.DOTALL
    )
    edited_text = (
        edited_text_match.group(1).strip()
        if edited_text_match
        else "No edited text found."
    )

    # Extract the corrections
    corrections_match = re.search(r"2\. Corrections:\s*(.*)", response, re.DOTALL)
    corrections = (
        corrections_match.group(1).strip()
        if corrections_match
        else "No corrections found."
    )
    print("INSIDE process_llm_output")
    # Print the results
    print("Edited Text:")
    print(edited_text)
    print("\nCorrections:")
    print(corrections)
    print("DONE process_llm_output")
    return edited_text, corrections
