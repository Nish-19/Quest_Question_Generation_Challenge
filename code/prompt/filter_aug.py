import os 
import pandas as pd 

def filter_aug(aug_df):
    dist_match = {'action': ['what'], 'causal relationship': ['why'], 
                  'character':['who'], 'feeling':['how'], 
                  'outcome resolution': ['what'], 'prediction':['what will', 'how will'],
                  'setting': ['where']}
    story, content, ans, ques, ls, attr, ei = [], [], [], [], [], [], []
    for i, row in aug_df.iterrows():
        # NOTE: Ques-1
        if row['r1_ans_score'] >= 0.5 or row['r1_org_score'] >= 0.5:
            if row['attribute'] == 'prediction':
                prefix = ' '.join(row['Response_1'].split(' ')[:2])
            else:
                prefix = row['Response_1'].split(' ')[0]
            if prefix in dist_match[row['attribute']]:
                story.append(row['story_name'])
                content.append(row['content'])
                ans.append(row['R1 Answer'])
                ques.append(row['Response_1'])
                ls.append(row['local_or_sum'])
                attr.append(row['attribute'])
                ei.append(row['ex_or_im'])
        # NOTE: Ques-2
        if row['r2_ans_score'] >= 0.5 or row['r2_org_score'] >= 0.5:
            if row['attribute'] == 'prediction':
                prefix = ' '.join(row['Response_2'].split(' ')[:2])
            else:
                prefix = row['Response_2'].split(' ')[0]
            if prefix in dist_match[row['attribute']]:
                story.append(row['story_name'])
                content.append(row['content'])
                ans.append(row['R2 Answer'])
                ques.append(row['Response_2'])
                ls.append(row['local_or_sum'])
                attr.append(row['attribute'])
                ei.append(row['ex_or_im'])
    return story, content, ans, ques, ls, attr, ei

def main():
    rouge_dir = 'rouge'
    all_story, all_content, all_ans, all_ques, all_ls, all_attr, all_ei = [], [], [], [], [], [], []
    for i in range(5):
        aug_filename = 'augment_fold_{:d}.csv'.format(i+1)
        print(aug_filename)
        aug_file_path = os.path.join(rouge_dir, aug_filename) 
        aug_df = pd.read_csv(aug_file_path)
        story, content, ans, ques, ls, attr, ei = filter_aug(aug_df)
        all_story.extend(story)
        all_content.extend(content)
        all_ans.extend(ans)
        all_ques.extend(ques)
        all_ls.extend(ls)
        all_attr.extend(attr)
        all_ei.extend(ei)
    df = pd.DataFrame()
    df['story_name'] = all_story
    df['content'] = all_content
    df['answer'] = all_ans
    df['question'] = all_ques
    df['local_or_sum'] = all_ls 
    df['attribute'] = all_attr 
    df['ex_or_im'] = all_ei
    # save df
    output_dir = 'filter'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    save_name = 'filter_aug.csv'
    save_path = os.path.join(output_dir, save_name)
    df.to_csv(save_path, index=False)
        
if __name__ == '__main__':
    main()