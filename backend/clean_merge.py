import pandas as pd
import json
import os

# 1. Load Data
food_df = pd.read_csv('daily_food_nutrition_dataset.csv', on_bad_lines='skip')
workout_df = pd.read_csv('fitness_and_workout_dataset.csv', on_bad_lines='skip')
members_df = pd.read_csv('gym_members_exercise_tracking.csv', on_bad_lines='skip')

print(f"Original lengths -> Food: {len(food_df)}, Workout: {len(workout_df)}, Members: {len(members_df)}")

# 2. Clean Data (Drop missing values and duplicates)
food_df = food_df.dropna().drop_duplicates()
workout_df = workout_df.dropna().drop_duplicates()
members_df = members_df.dropna().drop_duplicates()

print(f"Cleaned lengths -> Food: {len(food_df)}, Workout: {len(workout_df)}, Members: {len(members_df)}")

docs = []

# 3. Transform Food Data
for idx, row in food_df.iterrows():
    text = (
        f"Nutritional Information for {row['Food_Item']} (Category: {row['Category']}). "
        f"Meal Type: {row['Meal_Type']}. Calories: {row['Calories (kcal)']} kcal, "
        f"Protein: {row['Protein (g)']}g, Carbohydrates: {row['Carbohydrates (g)']}g, "
        f"Fat: {row['Fat (g)']}g, Fiber: {row['Fiber (g)']}g, Sugars: {row['Sugars (g)']}g, "
        f"Sodium: {row['Sodium (mg)']}mg, Cholesterol: {row['Cholesterol (mg)']}mg. "
        f"Water Intake: {row['Water_Intake (ml)']} ml."
    )
    docs.append({
        'source': 'daily_food_nutrition',
        'content': text
    })

# 4. Transform Workout Data
for idx, row in workout_df.iterrows():
    desc = str(row['description']).replace('\n', ' ')
    text = (
        f"Workout Program: {row['title']}. Difficulty Level: {row['level']}. "
        f"Main Goal: {row['goal']}. Equipment Required: {row['equipment']}. "
        f"Program Length: {row['program_length']}. Time per Workout: {row['time_per_workout']}. "
        f"Total Exercises: {row['total_exercises']}. "
        f"Description: {desc}"
    )
    docs.append({
        'source': 'fitness_and_workout',
        'content': text
    })

# 5. Transform Gym Members Data
for idx, row in members_df.iterrows():
    text = (
        f"Gym Member Profile Demo - Age: {row['Age']}, Gender: {row['Gender']}, "
        f"Height: {row['Height (m)']}m, Weight: {row['Weight (kg)']}kg, "
        f"BMI: {row['BMI']}, Fat Percentage: {row['Fat_Percentage']}%. "
        f"This member performs '{row['Workout_Type']}' workouts {row['Workout_Frequency (days/week)']} days a week "
        f"for {row['Session_Duration (hours)']} hours per session. "
        f"Experience Level: {row['Experience_Level']}. "
        f"Performance Stats: {row['Calories_Burned']} calories burned per session, "
        f"Resting BPM: {row['Resting_BPM']}, Avg BPM: {row['Avg_BPM']}, Max BPM: {row['Max_BPM']}. "
        f"Water Intake: {row['Water_Intake (liters)']} liters."
    )
    docs.append({
        'source': 'gym_members_tracking',
        'content': text
    })

# 6. Save Merged Dataset
merged_df = pd.DataFrame(docs)
output_path = 'merged_rag_dataset.csv'
merged_df.to_csv(output_path, index=False)
print(f"Data successfully merged into '{output_path}'. Total documents: {len(merged_df)}")
