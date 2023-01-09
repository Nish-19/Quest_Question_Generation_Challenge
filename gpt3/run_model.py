"""
Inference function for (finetuned) GPT-3 model.
"""
import time
import openai
from openai.error import RateLimitError, Timeout, APIError, ServiceUnavailableError, InvalidRequestError


delay_time = 5
decay_rate = 0.8
key_index = 0


def run_gpt3(prompts, args, api_keys):
    global delay_time
    global key_index
    # Slow down requests to avoid rate limit for codex
    if("code" in args.model_name):
        time.sleep(delay_time)

    # Alternate keys to avoid rate limit for codex
    openai.api_key = api_keys[key_index]
    #print(f"Using key {key_index} with delay {delay_time:.3f}")
    key_index = (key_index + 1) % len(api_keys)

    # Send request
    try:
        response = openai.Completion.create(model=args.model_name, 
                                        prompt=prompts, 
                                        max_tokens=args.max_tokens, 
                                        temperature=args.temperature, 
                                        top_p=args.top_p, 
                                        n=args.n, 
                                        best_of = args.best_of,
                                        stop=[args.stop])
        delay_time *= decay_rate
    except (RateLimitError, Timeout, APIError, ServiceUnavailableError) as exc:
        delay_time *= 2
        return run_gpt3(prompts, args, api_keys)
    except (InvalidRequestError) as exc:
        # Usually thrown when the prompt exceeds max tokens allowed
        print(f"InvalidRequestError = {exc}")
        questions = ["Error"] * len(prompts) * args.n
        questions = [questions[i:i+args.n] for i in range(0, len(questions), args.n)]
        return questions
    
    # Match completions to prompts by index since completions are not returned in the same order as prompts
    questions = [None] * len(prompts) * args.n
    for choice in response.choices:
        # Remove leading whitespace in generated question
        questions[choice.index] = choice.text.strip()
        questions[choice.index] = questions[choice.index].strip("\"\'")
        # Add question mark at end if stop sequence was question mark
        if( args.stop == "?"):
            questions[choice.index] = questions[choice.index] + "?"

    # Group n questions back together in a sublist
    questions = [questions[i:i+args.n] for i in range(0, len(questions), args.n)]

    return questions