from Script_Matcher import ScriptMatcher

# Create an instance of ScriptMatcher
script_matcher = ScriptMatcher(data_path="your_dataset.csv")

# Preprocess the dataset
script_matcher.preprocess_dataset()

# Create keyword dataset
script_matcher.create_keyword_dataset()

# Calculate similarity matrix
script_matcher.calculate_similarity_matrix()

# Define new synopsis and genres keywords
new_synopsis = "Your new synopsis here."
genres_keywords = ["genre1", "genre2", "genre3"]

# Find similar series
similar_series = script_matcher.find_similar_series(new_synopsis, genres_keywords)

# Output the similar series
for series in similar_series:
    print("Series:", series["Series"])
    print("Genre:", series["Genre"])
    print("Score:", series["Score"])
    print()
