import pandas as pd
#Pandas makes it easy to read, process and analyze data contained in files such as EXCEL
#Pandas упрощает чтение, обработку и анализ данных содержащихся в файлах к примеру ЭКСЕЛЬ
from sklearn.linear_model import Ridge
#Ridge is an implementation of linear regression with L2 regularization in the
# scikit-learn library. L2 regularization helps prevent model overfitting.
#Ridge - это реализация линейной регрессии с L2-регуляризацией в библиотеке scikit-learn.
# L2-регуляризация помогает предотвратить переобучение модели.
from sklearn.model_selection import train_test_split
#train_test_split is a function for splitting data into training and test sets. This is useful
# for evaluating a model's performance on data it did not see during training.
#train_test_split - это функция для разделения данных на обучающие и тестовые наборы. Это полезно для оценки
# производительности модели на данных, которые она не видела во время обучения.
import numpy as np
#NumPy provides support for working with large, multidimensional arrays and matrices.
# It is often used in scientific computing and machine learning.
#NumPy предоставляет поддержку для работы с большими, многомерными массивами и матрицами.
# Он часто используется в научных вычислениях и машинном обучении.
import warnings
#The module provides warning management.
# To temporarily disable warnings that may appear while executing code.
#Модуль предоставляет управление предупреждениями.
# Для временного отключения предупреждений, которые могут появиться при выполнении кода.
import matplotlib.pyplot as plt
#Matplotlib is a library for creating static, interactive and animated plots in Python.
# pyplot is a module that provides an interface for creating different kinds of plots.
#Matplotlib - это библиотека для создания статических, интерактивных и анимационных графиков в Python.
# pyplot - это модуль , предоставляющий интерфейс для создания различных видов графиков.
import seaborn as sns
#Seaborn is a data visualization library based on Matplotlib.
#Seaborn - это библиотека для визуализации данных, основанная на Matplotlib.

# Set seaborn style for better aesthetics
# оапределение стиля
sns.set(style="whitegrid")

# Assuming your data is in an Excel file named Data_Germany_lost.xlsx
# Replace 'Data_Germany_lost.xlsx' with your actual file name
# указание на название файда в котором содержаться необходимые данные
file_path = 'Data_Germany_lost.xlsx'

# Read the Excel file into a DataFrame
# чтение файла с расширением Excel
df = pd.read_excel(file_path)

# Select rows where 'commodity' is 'Wheat' and 'year' is between 1991 and 2001
# выборка строк с указанием интересующего нас продукта и временных периодов
wheat_data_training = df[(df['commodity'] == 'Wheat') & (df['year'] >= 1991) & (df['year'] <= 2015)]

# Prepare the data for model training
# подготовка данных для обучения модели
X_train = wheat_data_training[['year']]
y_train = wheat_data_training['loss_quantity']

# Suppress the UserWarning
# игорировать предупреждение пользователя
warnings.filterwarnings("ignore", category=UserWarning)

# Split the data into training and testing sets
#test_size=0.2: This parameter determines the proportion of data that will be used for the test set. In this case,
# test_size=0.2 means that 20% of the data will be allocated for testing, and 80% will be used for training models.
#random_state ensures reproducibility of data partitioning results.
# разделение данных на обучающие и тестовые
#test_size=0,2: Этот параметр определяет долю данных, которые будут использоваться для тестового набора.
# В данном случае test_size=0.2 означает, что 20% данных будут выделены для тестирования, а 80% будут использоваться
# для обучения моделей.
#random_state обеспечивает воспроизводимость результатов разделения данных.

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create and train the linear regression model with Ridge regularization
#Создаем  и обучаем модель линейной регрессии с помощью регуляризации Риджа.
alpha = 1.0  # you can adjust this value, but it is not recommended as a larger value will be obtained from your
# data since the learning model will be more childish.
#можно настроить это значение, но не рекомендуется, так как большее значение будет получено от ваших данных,
# так как модель обучения будет более десткой.
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)

# Make predictions for the years 2016 to 2018 указание на даты прогноза.
future_years = np.arange(2016, 2019).reshape(-1, 1)
predicted_losses_kg = model.predict(future_years)

# Convert predicted losses from kilograms to tons
predicted_losses_ton = predicted_losses_kg / 1000

# Define real losses
real_losses = [799, 797, 670]

# Create a DataFrame with the predictions and real losses
predictions_df = pd.DataFrame({
    'year': future_years.flatten(),
    'predicted_loss_quantity_ton': predicted_losses_ton,
    'real_loss_quantity': real_losses
})

# Display the predictions with real losses
print(predictions_df)

# Plot the predicted and real losses with different colors for each year. отображение данных реальныых потерь и прогнозируемых
plt.figure(figsize=(10, 6))
plt.plot(predictions_df['year'], predictions_df['predicted_loss_quantity_ton'], marker='o', linestyle='-', color='blue', label='Predicted Loss')
plt.plot(predictions_df['year'], predictions_df['real_loss_quantity'], linestyle='--', marker='o', color='blue', label='Real Loss (2016-2018)')

# Customize the plot
plt.title('Predicted vs Real Losses for Wheat (2016-2018)')
plt.xlabel('Year')
plt.ylabel('Loss Quantity (in tons)')
plt.xticks(predictions_df['year'])
plt.yticks(np.arange(500, 1001, 100))
plt.legend()

# Show the plot
plt.show()

# Revert the warning filter to default behavior Возврат фильтра предупреждений в значение по умолчанию.
warnings.filterwarnings("default", category=UserWarning)



