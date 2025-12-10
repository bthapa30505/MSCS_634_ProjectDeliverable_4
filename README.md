# MSCS_634_ProjectDeliverable_4

## Introduction

For the analysis, I have chosen the dataset related to student performance and various factors that are related to student performance. As a student myself, I am curious to know about the factors that play a major role in academic success. Another reason I chose this dataset is the number of columns it has. It has 20 columns, which are mostly numeric, and will help to perform various calculations to understand the impact of various factors. Additionally, the number of students participating in this study exceeded 5,000, which meets the minimum number of rows required for the dataset to be used for this project.

Also, this dateset contains a diverse set of academic, personal, and socioeconomic factors that influence student performance. With variables like study hours, attendance, parental involvement, motivation level, and family income, it enables the exploration of complex relationships and patterns affecting exam scores. With this dataset, I can apply a variety of data mining techniques such as correlation analysis, classification, regression, and clustering to identify key predictors of academic success. Moreover, the dataset’s mix of numerical and categorical attributes provides opportunities to practice data preprocessing, feature selection, and model evaluation which makes it an ideal, real-world dataset for learning and applying data mining concepts effectively.

Dataset link: https://www.kaggle.com/datasets/anassarfraz13/student-success-factors-and-insights

# Data cleaning and Visulization
For the data cleaning steps, I initially ensured that the missing information were filled by using imputation technique. I filled those data points with mean or median based on datatype. Secondly, I also ensured there were no duplicate rows present. And lastly,
I calculated IQR ranges to help understand the outliers in the dataset. However I could not find critical outliers in the dataset apart from the student score itself. After that I used the cleaned data to create various plots to visualize and understand the dataset in depth.

# Dataset Summary:

Shape: (6607, 20)

1. TOP PREDICTORS OF EXAM SCORE (by correlation):
   2. Attendance: 0.581
   3. Hours_Studied: 0.445
   4. Previous_Scores: 0.175
   5. Tutoring_Sessions: 0.157

2. IMPACT OF CATEGORICAL FEATURES ON EXAM SCORE:
   Parental Involvement:
     - Low: 66.36
     - Medium: 67.10
     - High: 68.09
   Motivation Level:
     - Low: 66.75
     - Medium: 67.33
     - High: 67.70
   Learning Disabilities:
     - No: 67.35
     - Yes: 66.27

3. OUTLIER SUMMARY:
   Features with most outliers:
     - Tutoring_Sessions: 430 outliers (6.5%)
     - Exam_Score: 104 outliers (1.6%)
     - Hours_Studied: 43 outliers (0.7%)
     - Attendance: 0 outliers (0.0%)
     - Sleep_Hours: 0 outliers (0.0%)

# Challenges Faced

Throughout this project,I faced several technical challenges that required careful problem-solving and iterative refinement. One significant obstacle was handling missing values across multiple categorical columns (Teacher_Quality, Parental_Education_Level, and Distance_from_Home), which required implementing appropriate imputation strategies using mode values for categorical data. Another critical challenge emerged during outlier detection and removal, where I faced an IndexingError due to pandas Series index misalignment when creating boolean masks for filtering outlier rows. I was able to resolve this by explicitly setting the Series index to match the DataFrame index using `pd.Series(False, index=df.index)` rather than relying on default integer indexing.


# EDA Findings:

The exploratory data analysis (EDA) of the student performance dataset provided meaningful insights into the factors influencing academic outcomes. The dataset was found to be well-structured with minimal missing or inconsistent data, ensuring reliable analysis. Key predictors such as Previous Scores, Hours Studied, and Attendance showed strong correlations with Exam Score, highlighting their significance in predicting performance. Additionally, the interactions between categorical and numerical features revealed complex dependencies that could further enhance model accuracy. Overall, these insights serve as a solid foundation for developing robust predictive models aimed at forecasting student exam performance with high precision for the future project deliverables.


## Summary of the dataset, modeling process, and evaluation results.


The project used a Kaggle dataset with 6,607 student records to figure out what factors influence exam scores. This data included 20 different features for each student, such as their study hours, attendance, and parental involvement. Before building any models, the data was cleaned by filling in a small amount of missing information and checking for any duplicate rows.

The main goal was to predict a student's Exam_Score. To do this, different types of regression models were built and compared, including simple, multiple, and polynomial models. A key step was "feature engineering," which involved converting all 20 data columns into 24 usable features for the models to analyze.

The results showed that the number of features was the most critical factor. A model using only one feature performed poorly, but models using all 24 features saw a massive 121% improvement. The most complex model (Polynomial Degree 3) failed completely, performing just as badly as the simplest one. The best models like Lasso, Ridge, and standard Multiple Linear all performed equally well, explaining about 77% of the score variance and predicting scores with an average error of just ±1.8 points.


## Meaningful insights and key observations about model performance.

Based on the regression models, there were multiple factors that influenced what the final exam score if the student will be. There was no single factor that would best predict the score of the student. This is why the multiple linear regression performed extremely well compared to single factor linear regression.While several models ended up at the top, their performance was nearly identical, meaning the special "regularization" techniques didn't offer a real advantage over the standard multiple linear model. The key was finding a balance; the simplest model was too basic, and the most complicated ones performed poorly. The best model seems to be Lasso as it is reliable and can explain about 77% of the variation in exam scores, and its predictions are typically accurate within ±1.8 points.

## Challenges encountered and describe how they were addressed.

One of the main challenges I faced was running polynomial regression with higher-degree values, such as 4 and 5. These higher degrees required a significantly larger amount of computation due to the increased number of features, which made the process slow and resource-intensive. To address this, I decided to omit degrees 4 and 5 and instead focused on completing the calculations for degree 3, which only took a few minutes. By analyzing the performance of degrees 1, 2, and 3, I was able to gain a clear understanding of the trend and expected performance for higher-degree models without overloading the system.

Another challenge I faced was understanding why Ridge and Lasso regularization seemed to have little effect. The cross-validated R² scores were almost identical across different alpha values, making it hard to tell if regularization was helping. To address this, I expanded the alpha range to include very small values (0.0001) and very large values (up to 1000), increased precision to six decimal places to detect subtle differences, and added analysis to separate the small alpha range (where models performed best) from values that were too extreme. I also implemented automatic detection and explanations for cases where regularization had minimal impact. From this, I realized that the model was not heavily overfitting, so regularization offered little additional benefit.


## Summarize key insights from your classification, clustering, and pattern mining efforts.

Classification Insights: Predictive Intervention
The analysis revealed that the Tuned Naïve Bayes model achieved the highest performance among classification models, with an accuracy of 0.9281 and an F1 score of 0.9536. This makes it a robust tool for predicting whether a student is likely to achieve a high or low exam score. By applying this model to historical or current student data, such as previous scores, attendance, and demographics, educational institutions can identify students at high risk of low performance long before exams occur. This predictive capability enables targeted interventions, allowing schools to prioritize resources such as remedial classes, counseling, tutoring, or parent engagement. By shifting from reactive to proactive support, institutions can maximize the effectiveness of their academic assistance programs.

Clustering Insights: Differentiated Support Groups
K-Means clustering revealed two distinct student groups primarily differentiated by previous scores and family income, indicating a strong link between socioeconomic factors and academic performance.

Cluster 0 (Lower Previous Scores, Low Family Income): This group contains the lowest percentage of high performers (20%). Interventions for these students could focus on addressing foundational skill gaps and providing additional material or financial resources.

Cluster 1 (Higher Previous Scores, Medium Family Income): While performing better overall, 70.7% of students in this cluster still score low. Interventions here may target advanced subject enrichment, motivation, or strategies to maintain academic momentum.

These clusters provide a clear, data-driven basis for resource allocation, enabling the school to distribute scholarships, tutoring slots, or counseling time where it is needed most, ensuring equitable support.

Association Rule Insights: Causal Factors and Policy
Association rule mining highlighted strong conditional relationships between various factors and student performance outcomes.

Risk Mitigation: High-confidence rules, such as Attendance_Low ⇒ Exam_Score_Low (98.16% confidence, 1.17 lift), indicate nearly certain predictors of poor performance. This underscores the importance of attendance as a non-negotiable factor in academic success. Schools should implement or strengthen policies to monitor attendance closely, such as mandatory check-ins or automated alerts for parents and counselors.

Resource Management: Rules like Access_to_Resources_Low ⇒ Exam_Score_Low (91.16% confidence) and Access_to_Resources_Medium ⇒ Exam_Score_Low (85.18% confidence) reveal that inadequate access to resources is a major barrier. The institution should audit and enhance resource accessibility, including library hours, computer labs, and textbook loan programs, particularly for students with reported low access. Interestingly, the rule Access_to_Resources_Low ⇒ Internet_Access_Yes (93.22% confidence) suggests that “low resources” may not mean a lack of internet but a deficiency in other critical tools, allowing the school to refine its understanding of resource gaps.

## Discuss the practical relevance of your findings and their real-world applications.

The study shows that data models can help schools improve student success. The Tuned Naïve Bayes model can predict which students may perform poorly, so schools can provide help early, like mentors, tutoring, or extra classes. Clustering and association rules show which students need more support based on past scores and family income. Schools can use this to give resources fairly, such as free tutoring or technology for those who need it most. Attendance is very important, and low attendance strongly predicts low exam scores, so strict monitoring and quick intervention can help. The findings also suggest that resources and student effort matter more than teacher quality alone. Overall, these tools help schools make smart, practical decisions to support students and improve learning outcomes.

## Identify challenges encountered and describe how they were addressed.

During this project, I faced several challenges, each requiring a tailored solution. One major challenge was selecting the most suitable models for the dataset I had chosen in Project Deliverable 1. For the classification task, the best model was identified based on accuracy and F1 score. I tested three different models and selected the one that performed the best. In contrast, for clustering, it was less clear which model would yield the most meaningful results for my dataset. To address this, I applied a similar strategy, experimenting with two different clustering models to explore the diverse insights each could provide. Another issue I faced was while using Jupyter Notebook was that the kernel kept crashing when processing heavy copmutation. I resolved this by restarting the kernel and ensuring that data was loaded in smaller, manageable chunks to prevent memory overload.
