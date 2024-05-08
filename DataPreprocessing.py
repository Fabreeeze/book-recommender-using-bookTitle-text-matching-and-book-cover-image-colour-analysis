# Load the dataset
# url = '/content/drive/MyDrive/books dataset/New folder/sales/kindle_data-v2.csv'
# dataset = pd.read_csv(url)

# Display the first few rows of the dataset
# print(dataset.head())

# Import necessary libraries


# Display the first few rows of the dataset
# print("First few rows of the dataset:")
# print(dataset.head())

# Check for missing values
# print("\nMissing values:")
# print(dataset.isnull().sum())

# Check data types of columns
# print("\nData types:")
# print(dataset.dtypes)

# dataset.shape

# Summary statistics
# print("\nSummary statistics:")
# print(dataset.describe())

# # Data Visualization
# # Distribution of prices
# plt.figure(figsize=(10, 6))
# sns.histplot(dataset['price'], bins=20, kde=True)
# plt.title('Distribution of Prices')
# plt.xlabel('Price')
# plt.ylabel('Frequency')
# plt.show()

# # Distribution of ratings
# plt.figure(figsize=(10, 6))
# sns.histplot(dataset['stars'], bins=20, kde=True)
# plt.title('Distribution of Ratings')
# plt.xlabel('Stars')
# plt.ylabel('Frequency')
# plt.show()

# # Relationship between ratings and price
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='price', y='stars', data=dataset)
# plt.title('Relationship between Ratings and Prices')
# plt.xlabel('Price')
# plt.ylabel('Stars')
# plt.show()

# dataset['reviews'].fillna(dataset['reviews'].median(), inplace=True)  # Fill missing reviews with median
# dataset['soldBy'].fillna('Unknown', inplace=True)  # Fill missing soldBy with 'Unknown'

# Data Cleaning


# # Convert categorical columns to numerical using one-hot encoding
# dataset = pd.get_dummies(dataset, columns=['author', 'soldBy', 'category_name'])

# Drop irrelevant columns for correlation analysis
# columns_to_drop = ['asin', 'imgUrl', 'productURL', 'category_id', 'publishedDate','author', 'soldBy', 'category_name','title']
# dataset_for_correlation = dataset.drop(columns_to_drop, axis=1)



# Check the cleaned dataset
# print("\nCleaned dataset:")
# print(dataset.head())

# dataset_for_correlation.head()

# # Correlation heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(dataset_for_correlation.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap')
# plt.show()

# # Import necessary libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score


# # Selecting relevant columns for the ML model
# X = dataset[['price', 'stars', 'isBestSeller', 'isEditorsPick', 'isGoodReadsChoice']]
# y = dataset['asin']  # Assuming 'asin' is the unique identifier for each book

# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the Random Forest Classifier
# clf = RandomForestClassifier(n_estimators=10, random_state=42)
# clf.fit(X_train, y_train)

# # Predicting on the test set
# y_pred = clf.predict(X_test)

# # Calculating accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

"""#above code blocks fail to execute due to large dataset size(over 133000 entries taking up all ram and crashing system)

#so we now try to take a subset of data

"""

# subset = pd.DataFrame()

# for category in dataset['category_name'].unique():
#     category_data = dataset[dataset['category_name'] == category]
#     if len(category_data) >= 500:
#         category_subset = category_data.sample(n=500, random_state=42, replace=False)
#     else:
#         category_subset = category_data.sample(n=500, random_state=42, replace=True)
#     subset = pd.concat([subset, category_subset])


# print(subset)

# columns_to_drop = ['asin', 'imgUrl', 'productURL', 'category_id', 'publishedDate','author', 'soldBy', 'category_name','title']
# subset_for_correlation = subset.drop(columns_to_drop, axis=1)

# subset['reviews'].fillna(subset['reviews'].median(), inplace=True)
# subset['soldBy'].fillna('Unknown', inplace=True)

# # subset = pd.get_dummies(subset, columns=['author', 'soldBy', 'category_name'])

# plt.figure(figsize=(10, 6))
# sns.heatmap(subset_for_correlation.corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap for Subset')
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.histplot(subset['price'], bins=20, kde=True)
# plt.title('Distribution of Prices for Subset')
# plt.xlabel('Price')
# plt.ylabel('Frequency')
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.histplot(subset['stars'], bins=20, kde=True)
# plt.title('Distribution of Ratings for Subset')
# plt.xlabel('Stars')
# plt.ylabel('Frequency')
# plt.show()

# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='price', y='stars', data=subset)
# plt.title('Relationship between Ratings and Prices for Subset')
# plt.xlabel('Price')
# plt.ylabel('Stars')
# plt.show()














    # import cv2
    # import numpy as np
    # from sklearn.cluster import KMeans
    # from collections import Counter
    # import os

    # import requests
    # import cv2
    # import numpy as np
    # from sklearn.cluster import KMeans
    # from collections import Counter
    # import os

    # # Function to download an image from a URL
    # def download_image(url, save_path):
    #     response = requests.get(url)
    #     if response.status_code == 200:
    #         with open(save_path, 'wb') as f:
    #             f.write(response.content)
    #         return True
    #     else:
    #         print("Failed to download image from:", url)
    #         return False

    # # Function to extract dominant color from downloaded images by class
    # def extract_dominant_color_by_class(dataset, class_name, num_colors=3):
    #     # Filter dataset by class
    #     class_subset = dataset[dataset['category_name'] == class_name]

    #     # Create directory to store downloaded images
    #     class_dir = os.path.join('downloaded_images', class_name)
    #     os.makedirs(class_dir, exist_ok=True)

    #     # Load and preprocess book cover images
    #     dominant_colors = []
    #     for i, image_url in enumerate(class_subset['imgUrl']):
    #         # Download image
    #         image_path = os.path.join(class_dir, f'image_{i}.jpg')
    #         if download_image(image_url, image_path):
    #             # Load downloaded image
    #             image = cv2.imread(image_path)
    #             if image is not None:
    #                 # Convert image from BGR to RGB format
    #                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #                 # Flatten image into list of pixels
    #                 pixels = image.reshape(-1, 3)

    #                 # Use KMeans clustering to find dominant colors
    #                 kmeans = KMeans(n_clusters=num_colors)
    #                 kmeans.fit(pixels)

    #                 # Get the dominant colors
    #                 dominant_color = kmeans.cluster_centers_[kmeans.labels_]

    #                 # Count the frequency of each dominant color
    #                 color_counts = Counter(tuple(color) for color in dominant_color)

    #                 # Get the most common dominant color
    #                 most_common_color = color_counts.most_common(1)[0][0]

    #                 # Add the most common dominant color to the list
    #                 dominant_colors.append(most_common_color)
    #             else:
    #                 print("Failed to load image:", image_path)
    #         else:
    #             print("Failed to download image:", image_url)

    #     # Return the dominant colors for the class
    #     return dominant_colors

    # # # Example usage:
    # # class_name = 'Medical'
    # # dominant_colors = extract_dominant_color_by_class(dataset, class_name)
    # # print("Dominant Colors for", class_name, ":", dominant_colors)

    # # dominant_colors

    # # # Function to extract dominant colors for all genres and store in a dictionary
    # # def extract_dominant_colors_for_all_genres(dataset):
    # #     dominant_colors_dict = {}
    # #     for genre in dataset['category_name'].unique():
    # #         print("Extracting dominant colors for", genre)
    # #         dominant_colors = extract_dominant_color_by_class(dataset, genre)
    # #         dominant_colors_dict[genre] = dominant_colors
    # #     return dominant_colors_dict

    # # # Example usage:
    # # dominant_colors_dict = extract_dominant_colors_for_all_genres(dataset)
    # # print("Dominant Colors for all genres:", dominant_colors_dict)

    # """the below is different from code a couple blocks above in sense it doesnt extract class dominant colour

    # """

    # # Function to extract dominant color from downloaded image
    # def extract_dominant_color(image_path, num_colors=3):
    #     # Load image
    #     image = cv2.imread(image_path)
    #     if image is not None:
    #         # Convert image from BGR to RGB format
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #         # Flatten image into list of pixels
    #         pixels = image.reshape(-1, 3)

    #         # Use KMeans clustering to find dominant colors
    #         kmeans = KMeans(n_clusters=num_colors)
    #         kmeans.fit(pixels)

    #         # Get the dominant colors
    #         dominant_color = kmeans.cluster_centers_[kmeans.labels_]

    #         # Count the frequency of each dominant color
    #         color_counts = Counter(tuple(color) for color in dominant_color)

    #         # Get the most common dominant color
    #         most_common_color = color_counts.most_common(1)[0][0]

    #         return most_common_color
    #     else:
    #         print("Failed to load image:", image_path)
    #         return None

    # # Function to extract and store dominant color for each book cover in the dataset
    # def extract_and_store_dominant_colors(dataset_subset, start_index=0):
    #     os.makedirs('downloaded_images', exist_ok=True)  # Create directory if not exists
    #     for index in range(start_index, len(dataset_subset)):
    #         row = dataset_subset.iloc[index]  # Get the current row from the subset
    #         print("Processing book", row['asin'])
    #         image_url = row['imgUrl']
    #         image_path = f'downloaded_images/image_{row["asin"]}.jpg'  # Use ASIN as filename
    #         if download_image(image_url, image_path):
    #             dominant_color = extract_dominant_color(image_path)
    #             if dominant_color is not None:
    #                 dataset_subset.at[row.name, 'DominantColor'] = dominant_color
    #             else:
    #                 # Set a default color if dominant_color is None
    #                 dataset_subset.at[row.name, 'DominantColor'] = (255, 255, 255)  # Default color: white
    #         else:
    #             print("Failed to download image for book", row['asin'])

    #         # Periodically save the progress
    #         if index % 100 == 0:
    #             save_checkpoint(dataset_subset, index)

    # # Function to save a checkpoint of the progress
    # def save_checkpoint(subset, index):
    #     # Save the dataset with the progress
    #     path = '/content/drive/MyDrive/books dataset/New folder/sales/checkpoint_subset.csv'
    #     subset.to_csv(path, index=False)
    #     # Save the index of the last processed image

    #     path2 = '/content/drive/MyDrive/books dataset/New folder/sales/checkpoint_index.txt'
    #     with open(path2, 'w') as f:
    #         f.write(str(index))

    # len(subset)

    # # subset

    # subset.columns

    # dataset.columns

    # subset = subset.reset_index( drop = True)
    # # subset



    # # Load the checkpoint if it exists
    # if os.path.exists('/content/drive/MyDrive/books dataset/New folder/sales/checkpoint_dataset.csv') and os.path.exists('/content/drive/MyDrive/books dataset/New folder/sales/checkpoint_index.txt'):
    #     subset = pd.read_csv('/content/drive/MyDrive/books dataset/New folder/sales/checkpoint_dataset.csv')
    #     with open('/content/drive/MyDrive/books dataset/New folder/sales/checkpoint_index', 'r') as f:
    #         start_index = int(f.read())
    #         print( start_index )
    # else:
    #     start_index = 1

    # start_index

    # start_index = 5600

    # # Create a new column to store dominant color
    # subset['DominantColor'] = None

    # # Call the function to extract and store dominant colors
    # extract_and_store_dominant_colors(subset,start_index)

    # excel_file_path = '/content/drive/MyDrive/books dataset/New folder/sales/updated_dataset.xlsx'
    # subset.to_excel(excel_file_path, index=False)

    # subset1 = pd.read_excel('/content/drive/MyDrive/books dataset/New folder/sales/5600 to end updated_dataset.xlsx') #contains dominant Colour info from index 5600 to 15500
    # subset2 = pd.read_excel('/content/drive/MyDrive/books dataset/New folder/sales/2100 to 5600 checkpoint_subset.xlsx') #contains 2100 to 5600
    # subset3 = pd.read_excel('/content/drive/MyDrive/books dataset/New folder/sales/1 to 2100 checkpoint_subset.xlsx') #contains 1 to 2100

    # dominant_colour_subset = pd.concat( [subset3['DominantColor'],subset2['DominantColor'], subset1['DominantColor'] ] , ignore_index =True)
    # other_cols_subset = subset1.drop(columns= 'DominantColor')

    # print(len(dominant_colour_subset))
    # print( len( other_cols_subset))

    # len(dominant_colour_subset) - dominant_colour_subset.isna().sum()

    # print(subset3['DominantColor'].iloc[2100] , subset3['DominantColor'].iloc[2101])
    # print(subset2['DominantColor'].iloc[2099] , subset2['DominantColor'].iloc[2100],subset2['DominantColor'].iloc[2101])
    # print(subset2['DominantColor'].iloc[5599] , subset2['DominantColor'].iloc[5600],subset2['DominantColor'].iloc[5601])

    # print(subset1['DominantColor'].iloc[5599] , subset1['DominantColor'].iloc[5600],subset1['DominantColor'].iloc[5601])

    # resultant_subset = subset3
    # print( len(resultant_subset))
    # print( resultant_subset['DominantColor'].isna().sum())
    # print(len(resultant_subset) - resultant_subset['DominantColor'].isna().sum())

    # for i in range(2101,5601):
    #   print("original = ",resultant_subset['DominantColor'].iloc[i])
    #   resultant_subset['DominantColor'].iloc[i] = subset2['DominantColor'].iloc[i]
    #   print(resultant_subset['DominantColor'].iloc[i])
    #   print(subset2['DominantColor'].iloc[i])

    # print( len(resultant_subset))
    # print( resultant_subset['DominantColor'].isna().sum())
    # print(len(resultant_subset) - resultant_subset['DominantColor'].isna().sum())

    # print( len(subset2))
    # print( subset2['DominantColor'].isna().sum())
    # print(len(subset2) - subset2['DominantColor'].isna().sum())
    # print( 15500 - 5600)

    # for i in range(5601,len(resultant_subset) ):
    #   print("at index = ",i," ,original = ",resultant_subset['DominantColor'].iloc[i])
    #   resultant_subset['DominantColor'].iloc[i] = subset1['DominantColor'].iloc[i]
    #   print(resultant_subset['DominantColor'].iloc[i])
    #   print(subset1['DominantColor'].iloc[i])

    # print( len(resultant_subset))
    # print( resultant_subset['DominantColor'].isna().sum())
    # print(len(resultant_subset) - resultant_subset['DominantColor'].isna().sum())

    # resultant_subset

    # path  = '/content/drive/MyDrive/books dataset/New folder/sales/final_updated_subset_dataset.xlsx'
    # resultant_subset.to_excel(path,index = False)

    # subset = pd.read_excel('/content/drive/MyDrive/books dataset/New folder/sales/final_updated_subset_dataset.xlsx')






# import numpy as np
    # from sklearn.metrics.pairwise import cosine_similarity

    # def nearest_matched_colour( dominant_color, color_labels):
    # similarities = []
    # for color_label in color_labels.values():
    #     similarity = cosine_similarity([dominant_color],[color_label])

    #     # print(similarity)
    #     similarity = similarity[0][0]
    #     similarities.append(similarity)
    # maxInd = np.argmax(similarities)
    # return list(color_labels.keys())[maxInd]

    # nearest_matched_colour((236.2931827910946, 239.16545376711557, 236.19654323629567) , color_labels)

    # # the rgb tuples in our subset dataset containt string values sintead of float, so needed to execute this block, after searching from net on how to , in order
    # # to convert them to flaoat tuples


    # import ast

    # # Function to convert string representation of tuple to actual tuple
    # def parse_tuple_string(tuple_string):
    #     return ast.literal_eval(tuple_string)

    # # Convert string representations of tuples to actual tuples in the 'DominantColor' column
    # subset['DominantColor'] = subset['DominantColor'].apply(parse_tuple_string)

    # subset['MatchedColor'] = subset['DominantColor'].apply(lambda x: nearest_matched_colour(x, color_labels))

    # subset[['asin','DominantColor','MatchedColor']]