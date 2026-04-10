from src.components.data_ingestion import DataIngestion

data = DataIngestion()
df = data.data_ingest()
print(df.head())