"""
Inference function for (finetuned) GPT-3 model.
"""
import time
import openai


def run_gpt3(prompts, args):
    # Slow down requests to avoid rate limit for codex
    if("code" in args.model_name):
        time.sleep(12)
    # TODO increase max_tokens if attribute has to be predicted as well
    response = openai.Completion.create(model=args.model_name, 
                                    prompt=prompts, 
                                    max_tokens=args.max_tokens, 
                                    temperature=args.temperature, 
                                    top_p=args.top_p, 
                                    n=args.n, 
                                    stop=[args.stop])
    
    # Match completions to prompts by index since completions are not returned in the same order as prompts
    questions = [None] * len(prompts)
    for choice in response.choices:
        # Remove leading whitespace in generated question
        questions[choice.index] = choice.text.strip()
        questions[choice.index] = questions[choice.index].strip("\"\'")
        # Add question mark at end if stop sequence was question mark
        if( args.stop == "?"):
            questions[choice.index] = questions[choice.index] + "?"

    return questions