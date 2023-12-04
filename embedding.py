import gensim.downloader as api

# Load the model (this might take some time)
model = api.load('word2vec-google-news-300')

# Get the vector for 'hot'
hot_vector = model['hot']
hotter_vector = model['hotter']
cold_vector = model['cold']
print(hot_vector)
print(hotter_vector)
print(hotter_vector - hot_vector)
print(cold_vector)
print(hot_vector - cold_vector)

