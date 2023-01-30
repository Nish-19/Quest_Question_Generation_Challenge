def generate(prompt, args, model, tokenizer, device):
    # Tokenize input
    # TODO: max input length needs to be set here as well?
    input_encoding = tokenizer(prompt, padding="longest", max_length=args.max_source_length, truncation=True, return_tensors="pt")
    input_ids, attention_mask = input_encoding.input_ids.to(device), input_encoding.attention_mask.to(device)
    """
    print(input_ids)
    print(attention_mask)
    for ids in input_ids:
        print(tokenizer.decode(ids))
    """
    # Generate questions
    if( args.decoding_type == "greedy" ):
        questions = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64, do_sample=False, num_return_sequences=args.num_return_sequences)
    elif( args.decoding_type == "beam_search" ):
        questions = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64, num_beams=10, early_stopping=True, num_return_sequences=args.num_return_sequences)
    elif( args.decoding_type == "top_k_sampling" ):
        questions = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64, do_sample=True, top_k=50, num_return_sequences=args.num_return_sequences)
    elif( args.decoding_type == "nucleus_sampling" ):
        questions = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64, do_sample=True, top_p=0.95, top_k=0, num_return_sequences=args.num_return_sequences)
    elif( args.decoding_type == "nucleus_sampling_with_top_k" ):
        questions = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64, do_sample=True, top_p=0.95, top_k=50, num_return_sequences=args.num_return_sequences)
    elif( args.decoding_type == "contrastive_search" ):
        questions = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64, do_sample=True, penalty_alpha=0.6, top_k=4, num_return_sequences=args.num_return_sequences)

    # Post-processing
    questions = tokenizer.batch_decode(questions, skip_special_tokens=True)
    # Remove original prompt from generated text
    #questions = [question.replace(prompt, "") for question in questions]
    # Remove surrounding whitespace in generated question
    questions = [question.strip() for question in questions]
    
    # Keep text only till first question mark
    questions = [question.split("?")[0] for question in questions]
    # Add question mark at end 
    questions = [question + "?" for question in questions]

    return questions