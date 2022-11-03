"""
Inference function for (finetuned) GPT-3 model.
"""

import openai


def run_gpt3(prompt, args):
    result = openai.Completion.create(model=args.model_name, 
                                    prompt=prompt, 
                                    max_tokens=args.max_tokens, 
                                    temperature=args.temperature, 
                                    top_p=args.top_p, 
                                    n=args.n, 
                                    stop=[args.stop])
    
    # Remove leading whitespace in generated question
    question = result['choices'][0]['text'].strip()
    question = question.strip("\"\'")

    # Add question mark at end if stop sequence was question mark
    if( args.stop == "?"):
        question = question + "?"

    return question