import time
import boto3
import pandas as pd
import io
import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

BUCKET = 'dm-mimic-iii'
PATH = "query-results"
AWS_REGION = "us-east-1"

DATABASE = 'mimiciii'

class QueryAthena:

    def __init__(self):
        self.s3_output =  's3://' + BUCKET + '/' + PATH


    def _load_conf(self, q):
        try:
            self.client = boto3.client('athena', 
                              region_name = AWS_REGION, 
                              aws_access_key_id = AWS_ACCESS_KEY,
                              aws_secret_access_key= AWS_SECRET_ACCESS_KEY)
            response = self.client.start_query_execution(
                QueryString = q,
                    QueryExecutionContext={
                    'Database': DATABASE
                    },
                    ResultConfiguration={
                    'OutputLocation': self.s3_output,
                    }
            )
            self.filename = response['QueryExecutionId']
            print('Execution ID: ' + response['QueryExecutionId'])

        except Exception as e:
            print(e)
        return response                     

    def _obtain_data(self):
        try:
            self.resource = boto3.resource('s3', 
                                  region_name = AWS_REGION, 
                                  aws_access_key_id = AWS_ACCESS_KEY,
                                  aws_secret_access_key= AWS_SECRET_ACCESS_KEY)

            response = self.resource \
            .Bucket(BUCKET) \
            .Object(key= PATH + '/' + self.filename + '.csv') \
            .get()

            return pd.read_csv(io.BytesIO(response['Body'].read()), encoding='utf8')   
        except Exception as e:
            print(e)

    def run_query(self, query):
        queries = [query]
        for q in queries:
            res = self._load_conf(q)
        try:              
            query_status = None
            while query_status == 'QUEUED' or query_status == 'RUNNING' or query_status is None:
                query_status = self.client.get_query_execution(QueryExecutionId=res["QueryExecutionId"])['QueryExecution']['Status']['State']
                print(query_status)
                if query_status == 'FAILED' or query_status == 'CANCELLED':
                    raise Exception('Athena query with the string "{}" failed or was cancelled'.format(query))
                time.sleep(10)
            print('Query "{}" finished.'.format(query))

            df = self._obtain_data()
            return df

        except Exception as e:
            print(e)