import os

from code.utils.create_dataset_split import load_df


RAW_DIR = "./data/"


def explore_train_set(filepath):
    print("\nExplore train set\n")
    df_train = load_df("train.csv", filepath)
    # Number of samples
    print("Number of samples: {}".format(len(df_train)))
    # Number of unique stories
    print("Number of unique stories: {}".format(len(df_train['source_title'].unique())))    
    # Stats for length of answer
    print("Stats for length of answer (words):\n{}".format(df_train['answer'].apply(lambda x: len(x.split(" "))).describe(percentiles=[.25, .5, .75, .9, .95, .99])))
    # Stats for length of question
    print("Stats for length of question (words):\n{}".format(df_train['question'].apply(lambda x: len(x.split(" "))).describe(percentiles=[.25, .5, .75, .9, .95, .99])))
    # Percentage of questions local (vs summary)
    print("Percentage of questions local (vs summary): {}".format(len(df_train[df_train['local_or_sum'] == 'local']) / len(df_train)))
    # Percentage of questions explicit (vs implicit)
    print("Percentage of questions explicit (vs implicit): {}".format(len(df_train[df_train['ex_or_im'] == 'explicit']) / len(df_train)))
    # Histogram over question attribute tags
    print("Histogram over question attribute tags:\n{}".format(df_train['attribute1'].value_counts(normalize=True)))


def explore_test_set(filepath):
    print("\nExplore test set\n")
    df_test = load_df("test.csv", filepath)
    # Number of samples
    print("Number of samples: {}".format(len(df_test)))
    # Number of unique stories
    print("Number of unique stories: {}".format(len(df_test['source_title'].unique())))
    # Stats for length of answer
    print("Stats for length of answer (words):\n{}".format(df_test['answer'].apply(lambda x: len(x.split(" "))).describe(percentiles=[.25, .5, .75, .9, .95, .99])))


def explore_story_source_texts(filepath):
    print("\nExplore story source texts\n")
    df_story = load_df("source_texts.csv", filepath)
    # Number of unique stories
    print("Number of unique stories: {}".format(len(df_story['source_title'].unique())))
    # Stats for number of sections per story
    print("Stats for number of sections per story:\n{}".format(df_story['cor_section'].describe(percentiles=[.25, .5, .75, .9, .95, .99])))
    # Stats for length of section
    print("Stats for length of section (words):\n{}".format(df_story['text'].apply(lambda x: len(x.split(" "))).describe(percentiles=[.25, .5, .75, .9, .95, .99])))
    # Stats for length of story, one story contains many sections
    df_story['num_words_sec'] = df_story['text'].apply(lambda x: len(x.split(" ")))
    df_groupby_story = df_story.groupby('source_title')
    print("Stats for length of story (words):\n{}".format(df_groupby_story.apply(lambda x: x["num_words_sec"].sum()).describe(percentiles=[.25, .5, .75, .9, .95, .99])))
    # TODO: why is mean story len of 1945 words != mean section len of 139 words X mean num of sections of 10.4?


def explore_data_aug(filepath):
    print("\nExplore for data augmentation\n")
    df_train = load_df("train.csv", filepath)
    # Group by source title and cor_section
    df_aug = df_train.groupby(['source_title', 'cor_section']).size().reset_index(name='count').sort_values(by=['count'], ascending=False).head(30)
    print(df_aug)


def main():
    filepath = os.path.join(RAW_DIR, "original")
    explore_data_aug(filepath)
    #explore_train_set(filepath)
    #explore_test_set(filepath)
    #explore_story_source_texts(filepath)


if __name__ == '__main__':
    main()