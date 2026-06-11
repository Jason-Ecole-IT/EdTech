"""Vérifier si les prédictions correspondent à la réalité."""
import pandas as pd

df = pd.read_csv('data/raw/student dropout.csv')

# Filtrer les étudiants avec des notes basses (similaire aux prédictions affichées)
low_grades = df[(df['Grade_1'] <= 10) & (df['Grade_2'] <= 10) & (df['Final_Grade'] <= 10)]

print(f'Étudiants avec notes basses (grade <= 10): {len(low_grades)}')
print(f'Taux de dropout réel: {low_grades["Dropped_Out"].mean():.2%}')
print()

print('Échantillon d\'étudiants à risque (notes basses):')
sample = low_grades[['Grade_1', 'Grade_2', 'Final_Grade', 'Number_of_Absences', 'Dropped_Out']].head(20)
print(sample.to_string())
print()

# Vérifier les étudiants avec final_grade = 0 (très à risque)
zero_grade = df[df['Final_Grade'] == 0]
print(f'Étudiants avec final_grade = 0: {len(zero_grade)}')
print(f'Taux de dropout réel: {zero_grade["Dropped_Out"].mean():.2%}')
print()

# Vérifier les étudiants avec beaucoup d'absences (> 20)
high_absences = df[df['Number_of_Absences'] > 20]
print(f'Étudiants avec absences > 20: {len(high_absences)}')
print(f'Taux de dropout réel: {high_absences["Dropped_Out"].mean():.2%}')
