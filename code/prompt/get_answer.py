'''
python get_answer.py -NK 6 -MN code-davinci-002 -D -MT 32 -T 0 -P 1 -FN 1
'''
import os
import argparse
import time
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import openai
from openai.error import RateLimitError, Timeout, APIError, ServiceUnavailableError


# set api keys
def set_api_keys():
    os.environ['OPENAI_API_KEY_1'] = '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'
    os.environ['OPENAI_API_KEY_2'] = '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'
    os.environ['OPENAI_API_KEY_3'] = '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'
    os.environ['OPENAI_API_KEY_4'] = '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'
    os.environ['OPENAI_API_KEY_5'] = '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'
    os.environ['OPENAI_API_KEY_6'] = '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'


# Collect api keys
def get_api_keys(num_keys):
    api_keys = []
    for i in range(num_keys):
        api_key = os.getenv("OPENAI_API_KEY_{:d}".format(i+1))
        if api_key:
            api_keys.append(api_key)
    return api_keys


# construct input prompt 
def construct_input_prompt(context, question):
    prompt = '<Example 8>\n'
    prompt += '<Context> {:s}\n'.format(context)
    prompt +=  '<Question> {:s}\n'.format(question)
    return prompt


###############################################
# NOTE: Response generation
delay_time = 5
decay_rate = 0.8
key_index = 0


def run_model(prompt, args, api_keys):
    global delay_time 
    global key_index 

    # sleep to avoid rate limit error
    time.sleep(delay_time)

    # alternate keys to avoid rate limit for codex
    print('Using Key Index {:d} and Key {:s}'.format(key_index, api_keys[key_index]))
    openai.api_key = api_keys[key_index]
    key_index = (key_index + 1) % len(api_keys)

    try:
        response = openai.Completion.create(
            model = args.model_name,
            prompt = prompt,
            max_tokens = args.max_tokens, 
            temperature = args.temperature,
            top_p = args.top_p,
            n = args.num_samples, 
#            best_of = 1, 
            stop = [args.stop]
        )

        delay_time *= decay_rate
    except (RateLimitError, Timeout, APIError, ServiceUnavailableError) as exc:
        print(exc)
        delay_time *= 2
        return run_model(prompt, args, api_keys)
    
    return response


def add_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-NK", "--num_keys", type=int, default=6, help="Number of API keys to use")
    parser.add_argument("-FN", "--fold_number", type=int, default=1, help="Fold Decoding")
    parser.add_argument("-MN", "--model_name", type=str, default="code-davinci-002", help="GPT-3 off-the-shelf or finetuned model name")
    parser.add_argument('-D', '--debug', action='store_true', help='Debug mode for augmenting on a small subset of 5 samples')
    # Codex generation parameters
    parser.add_argument("-MT", "--max_tokens", type=int, default=32, help="Maximum number of tokens to generate")
    parser.add_argument("-T", "--temperature", type=float, default=0, help="Temperature for sampling")
    parser.add_argument("-P", "--top_p", type=float, default=1, help="Top-p sampling")
    parser.add_argument("-N", "--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("-BO", "--best_of", type=int, default=1, help="Number of samples to take the best n, best_of must be >= n")
    parser.add_argument('-S', "--stop", type=str, default='<END>', help="Stop sequence")

    params = parser.parse_args()
    return params


def main():
    args = add_params()
    # set api keys
    set_api_keys()
    # get api keys
    api_keys = get_api_keys(args.num_keys)
    if args.fold_number % 2 == 0:
        api_keys = api_keys[:len(api_keys)//2]
    else:
        api_keys = api_keys[len(api_keys)//2:]
    # NOTE: Read QG prompt
    with open('prompt_dir/prompt.txt', 'r') as infile:
        main_prompt = infile.read()
    # NOTE: read test data
    print('Loading Fold {:d}'.format(args.fold_number))
    test_file_path = 'clean_aug/augment_fold_{:d}.csv'.format(args.fold_number)
    if args.debug:
        test_df = pd.read_csv(test_file_path)[:5]
    else:
        test_df = pd.read_csv(test_file_path)

    # get input prompts
    ip_prompts_org, ip_prompts_r1, ip_prompts_r2 = [], [], []
    for i in range(len(test_df)):
        prompt_org_ques = construct_input_prompt(test_df.loc[i, 'content'], test_df.loc[i, 'question'])
        prompt_res_1 = construct_input_prompt(test_df.loc[i, 'content'], test_df.loc[i, 'Response_1'])
        prompt_res_2 = construct_input_prompt(test_df.loc[i, 'content'], test_df.loc[i, 'Response_2'])
        ip_prompts_org.append(main_prompt + prompt_org_ques)
        ip_prompts_r1.append(main_prompt + prompt_res_1)
        ip_prompts_r2.append(main_prompt + prompt_res_2)
    
    # generate response for each prompt 
    all_org, all_r1, all_r2 = [], [], []
    for i, prompt in enumerate(tqdm(ip_prompts_org)):
        print('Prompt {:d} Original Question'.format(i))
        ans_org = run_model(ip_prompts_org[i], args, api_keys)
        print('Prompt {:d} Response 1'.format(i))
        ans_r1 = run_model(ip_prompts_r1[i], args, api_keys)
        print('Prompt {:d} Response 2'.format(i))
        ans_r2 = run_model(ip_prompts_r2[i], args, api_keys)
        all_org.append(ans_org)
        all_r1.append(ans_r1)
        all_r2.append(ans_r2)

    # NOTE: Collect responses
    org_ans, r1_ans, r2_ans = [], [], []
    for i in range(len(all_org)):
        org_ans.append(all_org[i]['choices'][0]['text'].strip())
        r1_ans.append(all_r1[i]['choices'][0]['text'].strip())
        r2_ans.append(all_r2[i]['choices'][0]['text'].strip())
        # print(response['choices']['text'])
    
    # Dump responses 
    test_df['Org Answer'] = org_ans
    test_df['R1 Answer'] = r1_ans
    test_df['R2 Answer'] = r2_ans
    
    output_dir = 'answer'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    save_path = os.path.join(output_dir, 'augment_fold_{:d}.csv'.format(args.fold_number))
    test_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    main()
