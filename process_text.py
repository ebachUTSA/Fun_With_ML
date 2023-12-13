from src.utils import generateTextFeatures, selectSQLPandas, insertSQLPandas
from src.config import Config
config = Config()

df = selectSQLPandas('select postid, body from craigslist where body is not null')

df = generateTextFeatures(df,'body')

df.drop('body',axis=1,inplace=True)

insertSQLPandas(df,'post_body_text_features')