# %%
import pandas as pd
import os

# Specify the directory containing the CSV files
directory = os.getcwd()

# List to hold DataFrames
dataframes = []
resultList = []


# Loop through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Construct full file path
        file_path = os.path.join(directory, filename)
        # Read the CSV file and append the DataFrame to the list
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Drop all empty columns
combined_df = combined_df.dropna(axis=1, how='all')

# Keep only specified columns
combined_df = combined_df[['trial', 'currentSchedule', 'choice', 'participant']]

# Filter rows based on column: 'currentSchedule'
combined_df = combined_df[combined_df['currentSchedule'] == "stable"]

# Sort by column: 'trial' (ascending)
combined_df = combined_df.sort_values(['trial'])

# Keep only one trial value per participant
combined_df = combined_df.drop_duplicates(subset=['participant', 'trial'])

#Export the cleaned DataFrame to a new CSV file
combined_df.to_csv('cleaned_chinese_data.csv', index=False)

# Calculate the percentage of times a participant switched choices for each participant and print the results
for participant in combined_df['participant'].unique():
    number_of_times_participant_switched = 0
    participant_df = combined_df[combined_df['participant'] == participant]
    previous_value = participant_df['choice'].iloc[0]
    total_attempts = len(participant_df)

    for current_value in participant_df['choice'].iloc[1:]:
        if previous_value != current_value:
            number_of_times_participant_switched += 1
        previous_value = current_value

    percentage = (number_of_times_participant_switched / (total_attempts)) * 100

    resultList.append((participant, number_of_times_participant_switched, total_attempts, percentage))

    # Print the results
    print(f"Participant {participant} switched choices {number_of_times_participant_switched} times out of {total_attempts} attempts or {percentage:.2f}% of the time.")

# Create a DataFrame from the resultList
resultDataFrame = pd.DataFrame(resultList, columns=['participant', 'number_of_times_participant_switched', 'total_attempts', 'percentage'])

#Sort the resultDataFrame by participant in ascending order
resultDataFrame = resultDataFrame.sort_values(by='participant')

#Export the resultDataFrame to a new CSV file
resultDataFrame.to_csv('chinese_results.csv', index=False)


