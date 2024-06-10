import pandas as pd
import numpy as np
import re
import math
from collections import Counter
import requests

WORD = re.compile(r'\w+')

# Applying cosine similarity for finding similarities between user interests and places
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

# Remove spaces from the category column of the dataset
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Calculating weighted rating of places
metadata = pd.read_csv('pak_content.csv', low_memory=False)

print("Select your preferred category:\n1.wildlife \n2.heritage \n3.pilgrimage\n4.park\n5.museum")
text1 = input("Enter User Interests: ")  # User preference
vector1 = text_to_vector(text1)
C = metadata['p_rating'].mean()
m = metadata['count'].quantile(0.75)

def weighted_rating(x, m=m, C=C):
    v = x['count']
    R = x['p_rating']
    # Calculation based on the Bayesian Rating Formula
    return (v / (v + m) * R) + (m / (m + v) * C)

metadata['category'] = metadata['category'].apply(clean_data)
metadata['score'] = metadata.apply(weighted_rating, axis=1)

# Cosine similarity calculation
cos = []
for i in list(metadata['category']):
    text2 = i
    vector2 = text_to_vector(text2)
    cosine = get_cosine(vector1, vector2)
    cos.append(cosine)
metadata['cosine'] = cos

x = metadata['cosine'] > 0.0
rec = pd.DataFrame(metadata[x])
rec = rec.sort_values('score', ascending=False)
src = input("Enter your location: ")
dest = list(rec['title'])

# Function to calculate distance matrix using distancematrix.ai API
# Function to calculate distance matrix using distancematrix.ai API
def distance_matrix(origins, destinations, api_key):
    base_url = "https://api.distancematrix.ai/maps/api/distancematrix/json"
    params = {
        "origins": "|".join(origins),
        "destinations": "|".join(destinations),
        "key": api_key
    }

    response = requests.get(base_url, params=params)
    result = response.json()

    # Check if the response contains the expected keys
    if 'rows' in result and result['rows']:
        elements = result['rows'][0]['elements']
        if elements and 'distance' in elements[0] and 'duration' in elements[0]:
            return elements[0]

    # If the response structure is not as expected, return None
    return None

api_key = 'XVX4x2VShmxwPM7FtTd1VvqSDXjnyqAscbniA9HuBjuIpBgdbPlkP4xl0Vg1J0Ni'
dist = []
dur = []
for d in dest:
    output = distance_matrix([src], [d], api_key)

    if output:
        a1 = output['distance']['text']
        a2 = output['duration']['text']
    else:
        # If the response is not as expected, set default values
        a1 = 'N/A'
        a2 = 'N/A'

    dist.append(a1)
    dur.append(a2)

rec['distance'] = dist
rec['duration'] = dur

final = pd.DataFrame(rec, index=None, columns=['title', 'category', 'score', 'distance', 'duration'])
print(final)
