"""
Generate function for (finetuned) GPT-2 model.
"""


def run_gpt2(prompt, args, model, tokenizer, device):
    #print(prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    #print(input_ids)
    if( args.decoding_type == "greedy" ):
        outputs = model.generate(input_ids, max_new_tokens=30, do_sample=False, num_return_sequences=args.num_return_sequences)
    elif( args.decoding_type == "beam_search" ):
        outputs = model.generate(input_ids, max_new_tokens=30, num_beams=5, early_stopping=True, num_return_sequences=args.num_return_sequences)
    elif( args.decoding_type == "top_k_sampling" ):
        outputs = model.generate(input_ids, max_new_tokens=30, do_sample=True, top_k=50, num_return_sequences=args.num_return_sequences)
    elif( args.decoding_type == "nucleus_sampling" ):
        outputs = model.generate(input_ids, max_new_tokens=30, do_sample=True, top_p=0.95, top_k=0, num_return_sequences=args.num_return_sequences)
    elif( args.decoding_type == "nucleus_sampling_with_top_k" ):
        outputs = model.generate(input_ids, max_new_tokens=30, do_sample=True, top_p=0.95, top_k=50, num_return_sequences=args.num_return_sequences)

    # Post-processing
    questions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Remove original prompt from generated text
    questions = [question.replace(prompt, "") for question in questions]
    # Remove surrounding whitespace in generated question
    questions = [question.strip() for question in questions]
    
    # Keep text only till first question mark
    questions = [question.split("?")[0] for question in questions]
    # Add question mark at end 
    questions = [question + "?" for question in questions]

    # Add question mark at end if stop sequence was question mark
    #if( args.stop == "?"):
    #    question = question + "?"
    
    # TODO: handle multiple questions when num_return_sequences > 1
    return questions[0]