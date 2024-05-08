# path = 'final_updated_with_nearestMatchedColors_subset_dataset.xlsx'

# subset.to_excel(path,index = False)

# subset = pd.read_excel(path)
# print("dataset read")

# subset

# # Group the subset DataFrame by 'category_name'
# grouped_subset = subset.groupby('category_name')

# # Initialize dictionaries to store results
# most_common_colors = {}
# color_counts = {}
# color_percentages = {}

# # Iterate over each category group
# for category, group in grouped_subset:
#     # Find the mode (most common) dominant color for the category
#     most_common_color = group['MatchedColor'].mode()[0]
#     most_common_colors[category] = most_common_color

#     # Count the occurrences of each dominant color within the category
#     color_counts[category] = group['MatchedColor'].value_counts()

#     # Calculate the total count of books in the category
#     total_books = len(group)

#     # Calculate the percentage of the most common color within the category
#     most_common_color_count = color_counts[category].get(most_common_color, 0)
#     color_percentages[category] = most_common_color_count / total_books * 100

# # Display the results
# for category in most_common_colors:
#     print(f"Category: {category}")
#     print(f"Most Common Color: {most_common_colors[category]}")
#     print(f"Percentage of Most Common Color: {color_percentages[category]}%")
#     print()

# categoryList = []
# freqColor = []
# percentages = []
# for category in most_common_colors:
# categoryList.append(category)
# freqColor.append(most_common_colors[category])
# percentages.append(color_percentages[category])
# print( freqColor)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
# plt.bar(np.arange(len(categoryList)), percentages, color=freqColor, edgecolor='black')
# plt.xlabel('Category')
# plt.ylabel('Percentage of Dominant Color')
# plt.title('Most Common Colors for Each Category')
# plt.xticks(np.arange(len(categoryList)), categoryList, rotation=45, ha='right')
# plt.tight_layout()
# plt.show()

# """#So it turns out that white is most dominant, this could be due to many factors like large book headings or other words written in white on multicoloured backgrounds.
# # so we focus on the second domiant colour if first one is white
# """

# # Group the subset DataFrame by 'category_name'
# grouped_subset = subset.groupby('category_name')

# # Initialize dictionaries to store results
# most_common_colors2 = {}
# color_counts2 = {}
# color_percentages2 = {}
# common_color_dict = {}
# # Iterate over each category group
# for category, group in grouped_subset:
#     # Find the mode (most common) dominant color for the category
#     common_color_dict[category] = group['MatchedColor'].value_counts()
#     most_common_color = 'White'

#     if( len(common_color_dict[category]) > 0 and common_color_dict[category].head(1).index.tolist()[0] != 'White' ):
#     most_common_color = common_color_dict[category].head(1).index.tolist()[0]
#     elif( len(common_color_dict[category]) > 1 ):
#     # print("here")
#     most_common_color = common_color_dict[category].head(2).index.tolist()[1]
#     most_common_colors2[category] = most_common_color

#     # Count the occurrences of each dominant color within the category
#     color_counts2[category] = group['MatchedColor'].value_counts()

#     # Calculate the total count of books in the category
#     total_books = len(group)

#     # Calculate the percentage of the most common color within the category
#     most_common_color_count = color_counts2[category].get(most_common_color, 0)
#     color_percentages2[category] = most_common_color_count / total_books * 100


# # Display the results
# for category in most_common_colors2:
#     print(f"Category: {category}")
#     print(f"Most Common Color: {most_common_colors2[category]}")
#     print(f"Percentage of Most Common Color: {color_percentages2[category]}%")
#     print()

# categoryList2 = []
# freqColor2 = []
# percentages2 = []
# for category in most_common_colors2:
# categoryList2.append(category)
# freqColor2.append(most_common_colors2[category])
# percentages2.append(color_percentages2[category])
# print( freqColor2)


# plt.figure(figsize=(10, 6))
# plt.bar(np.arange(len(categoryList2)), percentages2, color=freqColor2, edgecolor='black')
# plt.xlabel('Category')
# plt.ylabel('Percentage of Dominant Color')
# plt.title('Second Most Common Colors for Each Category')
# plt.xticks(np.arange(len(categoryList2)), categoryList2, rotation=45, ha='right')
# plt.tight_layout()
# plt.show()



# """#Now we try to find and plot the top 5 dominant catgroy colours"""

# # Group the subset DataFrame by 'category_name'
# grouped_subset = subset.groupby('category_name')

# for i in range(10):

# # Initialize dictionaries to store results
# most_common_colors_i = {}
# color_counts_i = {}
# color_percentages_i = {}
# common_color_dict_i = {}
# # Iterate over each category group
# for category, group in grouped_subset:
#     # Find the mode (most common) dominant color for the category
#     common_color_dict_i[category] = group['MatchedColor'].value_counts()
#     most_common_color = None

#     if( len(common_color_dict_i[category]) >= i+1):
#         # print("here")
#         most_common_color = common_color_dict_i[category].head(i+1).index.tolist()[i]
#         if( most_common_color == 'Red-Orange'):
#         most_common_color = (255/255,69/255,0)

#     most_common_colors_i[category] = most_common_color

#     # Count the occurrences of each dominant color within the category
#     color_counts_i[category] = group['MatchedColor'].value_counts()

#     # Calculate the total count of books in the category
#     total_books = len(group)

#     # Calculate the percentage of the most common color within the category
#     most_common_color_count = color_counts_i[category].get(most_common_color, 0)
#     color_percentages_i[category] = most_common_color_count / total_books * 100


# # # Display the results
# # for category in most_common_colors_i:
# #     print(f"Category: {category}")
# #     print(f"Most Common Color: {most_common_colors_i[category]}")
# #     print(f"Percentage of Most Common Color: {color_percentages_i[category]}%")
# #     print()


# categoryList_i = []
# freqColor_i = []
# percentages_i = []
# for category in most_common_colors_i:
#     categoryList_i.append(category)
#     if( most_common_colors_i[category] != None):
#     freqColor_i.append(most_common_colors_i[category])
#     percentages_i.append(color_percentages_i[category])
#     else:
#     freqColor_i.append('gray')
#     percentages_i.append(0)
# print( freqColor_i)


# plt.figure(figsize=(10, 6))
# plt.bar(np.arange(len(categoryList_i)), percentages_i, color=freqColor_i, edgecolor='black')
# plt.xlabel('Category')
# plt.ylabel('Percentage of Dominant Color')
# plt.title(f'Number {i+1} Most Common Color for Each Category')
# plt.xticks(np.arange(len(categoryList_i)), categoryList_i, rotation=45, ha='right')
# plt.tight_layout()
# plt.show()