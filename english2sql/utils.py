def get_cleaned_model_id(model_id: str) -> str:
    return model_id.replace('/', '__').replace('.', '_')
