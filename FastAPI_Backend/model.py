import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def scaling(dataframe):
    scaler = StandardScaler()
    numeric_cols = [
        'Calories',
        'FatContent',
        'SaturatedFatContent',
        'CholesterolContent',
        'SodiumContent',
        'CarbohydrateContent',
        'FiberContent',
        'SugarContent',
        'ProteinContent',
    ]
    numeric_df = dataframe[numeric_cols].apply(pd.to_numeric, errors='coerce')
    if numeric_df.isnull().any().any():
        raise ValueError('Numeric columns contain non-convertible values')
    prep_data = scaler.fit_transform(numeric_df.to_numpy())
    return prep_data, scaler


def nn_predictor(prep_data):
    neigh = NearestNeighbors(metric='cosine', algorithm='brute')
    neigh.fit(prep_data)
    return neigh


def build_pipeline(neigh, scaler, params):
    transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
    pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])
    return pipeline


def extract_ingredient_filtered_data(dataframe, ingredients):
    """Filter recipes to those containing all specified ingredients.

    Uses the pre-processed '_ingredients_parsed' frozenset column (built once at
    startup in main.py) for fast substring matching without regex overhead. Each
    query term is checked against every ingredient name via a simple 'in' test,
    so "chicken" matches "chicken breast", "chicken thigh", etc.
    """
    if not ingredients:
        return dataframe
    queries = [i.lower() for i in ingredients]

    def matches(ingredient_set):
        return all(any(q in ing for ing in ingredient_set) for q in queries)

    return dataframe[dataframe['_ingredients_parsed'].apply(matches)]


def apply_pipeline(pipeline, _input, extracted_data):
    _input = np.array(_input).reshape(1, -1)
    return extracted_data.iloc[pipeline.transform(_input)[0]]


def recommend(dataframe, _input, ingredients=[], params={'n_neighbors': 5, 'return_distance': False}):
    extracted_data = extract_ingredient_filtered_data(dataframe, ingredients)
    if extracted_data.shape[0] >= params['n_neighbors']:
        prep_data, scaler = scaling(extracted_data)
        neigh = nn_predictor(prep_data)
        pipeline = build_pipeline(neigh, scaler, params)
        return apply_pipeline(pipeline, _input, extracted_data)
    return None


def extract_quoted_strings(s):
    return re.findall(r'"([^"]*)"', s)


def output_recommended_recipes(dataframe):
    if dataframe is None:
        return None
    # Drop the pre-processed helper column — it's a frozenset, not JSON-serializable.
    output = dataframe.drop(columns=['_ingredients_parsed'], errors='ignore').to_dict("records")
    for recipe in output:
        recipe['RecipeIngredientParts'] = extract_quoted_strings(recipe.get('RecipeIngredientParts', ''))
        recipe['RecipeInstructions'] = extract_quoted_strings(recipe.get('RecipeInstructions', ''))
        # Dataset stores time fields as integers; coerce to str for Pydantic v2 compatibility.
        # Some recipes may not have the time fields (older/filtered data). Handle safely.
        for time_field in ('CookTime', 'PrepTime', 'TotalTime'):
            if time_field in recipe and recipe[time_field] is not None:
                recipe[time_field] = str(recipe[time_field])
            else:
                recipe[time_field] = ''
    return output
