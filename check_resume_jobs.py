import pandas as pd
from src.utils import wordDistance, selectSQLPandas, compareTexts, insertSQLPandas
from src.config import Config
config = Config()

resumes_df = selectSQLPandas("select * from resumes")
jobs_df = selectSQLPandas("select * from jobs")

insert_dict = {
    'job_postid':[]
    ,'resume_postid':[]
    ,'text_similarity':[]
}

for index,row in resumes_df.iterrows():
    rtext = row['body']
    rpostid = row['postid']
    for index2,row2 in jobs_df.iterrows():
        insert_dict = {
            'job_postid':[]
            ,'resume_postid':[]
            ,'text_similarity':[]
        }
        jtext = row2['body']
        jpostid = row2['postid']
        text_similarity = compareTexts(rtext,jtext)
        insert_dict['job_postid'].append(jpostid)
        insert_dict['resume_postid'].append(rpostid)
        insert_dict['text_similarity'].append(text_similarity)
        df = pd.DataFrame.from_dict(insert_dict)
        insertSQLPandas(df,'text_similarity')