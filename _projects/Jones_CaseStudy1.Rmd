library(tm)
library(class)
library(caret)
library(e1071)
library(dplyr)
library(ggplot2)
library(GGally)
library(tidyr)
library(plyr)
library(jsonlite)
library(tidyverse)
library(pROC)

#Load new
attrition_data <- CaseStudy1.data

#observations
dim(attrition_data)
str(attrition_data)
summary(attrition_data)

#clean out variables that are qualitatively not predictors or consistent throughout
attrition_clean <- attrition_data %>%
  select(-ID, -EmployeeCount, -EmployeeNumber, -Over18, -StandardHours)

summary(attrition_clean)

#categorical Varaibles to Factors via specified data dictionary labels
attrition_cleanNB <- attrition_clean

text_vars <- c("Attrition", "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime")

attrition_cleanNB[text_vars] <- lapply(attrition_cleanNB[text_vars], as.factor)

attrition_cleanNB$Education <- factor(
  attrition_cleanNB$Education,
  levels = 1:5,
  labels = c("Below_College", "College", "Bachelor", "Master", "Doctor"),
  ordered = TRUE
)

attrition_cleanNB$EnvironmentSatisfaction <- factor(
  attrition_cleanNB$EnvironmentSatisfaction,
  levels = 1:4,
  labels = c("Low", "Medium", "High", "Very_High"),
  ordered = TRUE
)

attrition_cleanNB$JobInvolvement <- factor(
  attrition_cleanNB$JobInvolvement,
  levels = 1:4,
  labels = c("Low", "Medium", "High", "Very_High"),
  ordered = TRUE
)

attrition_cleanNB$JobLevel <- factor(
  attrition_cleanNB$JobLevel,
  levels = 1:5,
  labels = c("Entry", "Junior", "Mid", "Senior", "Executive"),
  ordered = TRUE
)

attrition_cleanNB$JobSatisfaction <- factor(
  attrition_cleanNB$JobSatisfaction,
  levels = 1:4,
  labels = c("Low", "Medium", "High", "Very_High"),
  ordered = TRUE
)

attrition_cleanNB$PreformanceRating <- factor(
  attrition_cleanNB$PerformanceRating,
  levels = 1:4,
  labels = c("Low", "Good", "Excellent", "Outstanding"),
  ordered = TRUE
)

attrition_cleanNB$RealtionshipSatisfaction <- factor(
  attrition_cleanNB$RelationshipSatisfaction,
  levels = 1:4,
  labels = c("Low", "Medium", "High", "Very_High"),
  ordered = TRUE
)

attrition_cleanNB$StockOptionLevel <- factor(
  attrition_cleanNB$StockOptionLevel,
  levels = 0:3,
  labels = c("None", "Low", "Medium", "High"),
  ordered = TRUE
)

attrition_cleanNB$WorkLifeBalance <- factor(
  attrition_cleanNB$WorkLifeBalance,
  levels = 1:4,
  labels = c("Bad", "Good", "Better", "Best"),
  ordered = TRUE
)

#Take out performance rating only two outcomes - subjective and only two outcomes
attrition_cleanNB1 <- attrition_data %>%
  select(-PerformanceRating)

attrition_cleanNB1 <- attrition_cleanNB %>%
  select(-PreformanceRating)

attrition_cleanNB2 <- attrition_cleanNB1 %>%
  select(-PerformanceRating)

#Train/Split
set.seed(1)

trainIndices = sample(seq(1:length(attrition_cleanNB1$Age)),round(.7*length(attrition_cleanNB1$Age)))
                      
trainAttrition = attrition_cleanNB1[trainIndices,]
testAttrition = attrition_cleanNB1[-trainIndices,]

model = naiveBayes(trainAttrition[,c("Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction", 
"Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate", 
"NumCompaniesWorked", "OverTime", "PercentSalaryHike", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", 
"WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "RelationshipSatisfaction")],
factor(trainAttrition$Attrition, labels = c("No", "Yes")))
table(factor(testAttrition$Attrition, labels = c("No", "Yes")),
      predict(model,testAttrition[,c("Age", "BusinessTravel", "DailyRate", "Department", "DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction", 
"Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate", 
"NumCompaniesWorked", "OverTime", "PercentSalaryHike", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear", 
"WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "RelationshipSatisfaction")]))

#positive=no
confusionMatrix(table(factor(testAttrition$Attrition, labels = c("No", "Yes")),predict(model,testAttrition[,c("Age", "BusinessTravel", "DailyRate", "Department", 
"DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", 
"MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", 
"TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "RelationshipSatisfaction")])))


#CM1NB
CM1NB = confusionMatrix(table(factor(testAttrition$Attrition, levels = c("No", "Yes"), labels = c("Attrition_No", "Attrition_Yes")),
 factor(  # ← Starting factor call
predict(model, testAttrition[,c("Age","BusinessTravel", "DailyRate", "Department", 
"DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction", "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", 
"MaritalStatus", "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears", 
"TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager", "RelationshipSatisfaction")]), # ← End predict call HERE
levels = c("No", "Yes"), labels = c("Attrition_No", "Attrition_Yes")  # ← Factor arguments go HERE
                        )), positive = "Attrition_Yes")

CM1NB <- confusionMatrix(pred, ref, positive = "Attrition_Yes")
CM1NB

#2 NB Load Data set
attrition_cleanNB3 <- attrition_cleanNB2%>%
  select(-NumCompaniesWorked, -MonthlyRate, -HourlyRate, -DailyRate, -PercentSalaryHike, -YearsSinceLastPromotion,
         -TrainingTimesLastYear, -EducationField, -RelationshipSatisfaction, -Gender, -Education)

set.seed(1)

trainIndices = sample(seq(1:length(attrition_cleanNB3$Age)),round(.7*length(attrition_cleanNB3$Age)))

trainAttrition = attrition_cleanNB3[trainIndices,]
testAttrition = attrition_cleanNB3[-trainIndices,]

model = naiveBayes(trainAttrition[,c("Age", "BusinessTravel", "Department", "DistanceFromHome", 
"EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", 
"OverTime", "StockOptionLevel", "TotalWorkingYears", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager")],
                   factor(trainAttrition$Attrition, labels = c("No", "Yes")))
table(factor(testAttrition$Attrition, labels = c("No", "Yes")),
      predict(model,testAttrition[,c("Age", "BusinessTravel", "Department", "DistanceFromHome", 
"EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", 
"OverTime", "StockOptionLevel", "TotalWorkingYears", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager")]))

confusionMatrix(table(factor(testAttrition$Attrition, labels = c("No", "Yes")),predict(model,testAttrition[,c("Age", "BusinessTravel", 
"Department", "DistanceFromHome", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", 
"OverTime", "StockOptionLevel", "TotalWorkingYears", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager")])))



#CM2NB
CM2NB = confusionMatrix(table(factor(testAttrition$Attrition, levels = c("No", "Yes"), labels = c("Attrition_No", "Attrition_Yes")),
                              predict(model,testAttrition[,c("Age", "BusinessTravel", "Department", "DistanceFromHome", 
"EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", "MonthlyIncome", 
"OverTime", "StockOptionLevel", "TotalWorkingYears", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager")])))

CM2NB = confusionMatrix(table(factor(testAttrition$Survived, labels = c("No", "Yes")),predict(model,testTitanic[,c("Age", "BusinessTravel", 
"Department", "DistanceFromHome", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction", "MaritalStatus", 
"MonthlyIncome", "OverTime", "StockOptionLevel", "TotalWorkingYears", "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole", "YearsWithCurrManager")])))

CM2NB

#Load Variables
CaseStudy1 <- CaseStudy1.data %>%
  select(-ID, -EmployeeCount, -EmployeeNumber, -Over18, -StandardHours)

Variables <- CaseStudy1

#factor

text_vars <- c("Attrition", "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime")

Variables[text_vars] <- lapply(Variables[text_vars], as.factor)

Variables$Education <- factor(
  Variables$Education,
  levels = 1:5,
  labels = c("Below_College", "College", "Bachelor", "Master", "Doctor"),
  ordered = TRUE
)

Variables$EnvironmentSatisfaction <- factor(
  Variables$EnvironmentSatisfaction,
  levels = 1:4,
  labels = c("Low", "Medium", "High", "Very_High"),
  ordered = TRUE
)

Variables$JobInvolvement <- factor(
  Variables$JobInvolvement,
  levels = 1:4,
  labels = c("Low", "Medium", "High", "Very_High"),
  ordered = TRUE
)

Variables$JobLevel <- factor(
  Variables$JobLevel,
  levels = 1:5,
  labels = c("Entry", "Junior", "Mid", "Senior", "Executive"),
  ordered = TRUE
)

Variables$JobSatisfaction <- factor(
  Variables$JobSatisfaction,
  levels = 1:4,
  labels = c("Low", "Medium", "High", "Very_High"),
  ordered = TRUE
)

Variables$PreformanceRating <- factor(
  Variables$PerformanceRating,
  levels = 1:4,
  labels = c("Low", "Good", "Excellent", "Outstanding"),
  ordered = TRUE
)

Variables$RealtionshipSatisfaction <- factor(
  Variables$RelationshipSatisfaction,
  levels = 1:4,
  labels = c("Low", "Medium", "High", "Very_High"),
  ordered = TRUE
)

variables$StockOptionLevel <- factor(
  Variables$StockOptionLevel,
  levels = 0:3,
  labels = c("None", "Low", "Medium", "High"),
  ordered = TRUE
)

variables$WorkLifeBalance <- factor(
  Variables$WorkLifeBalance,
  levels = 1:4,
  labels = c("Bad", "Good", "Better", "Best"),
  ordered = TRUE
)



#Categorical Variables Test

Catgorical_vars <- c("BusinessTravel", "Department", "Education", "EducationField", 
                     "Gender", "JobRole", "MaritalStatus", "OverTime", "JobInvolvement",
                     "JobLevel", "JobSatisfaction", "EnvironmentSatisfaction", 
                     "RelationshipSatisfaction", "PerformanceRating", "StockOptionLevel",
                     "TrainingTimesLastYear", "WorkLifeBalance")

chi_results <- data.frame(
  variable = character(),
  Chi_Square = numeric(),
  P_Value = numeric(),
  Significant = character(),
  stringsAsFactors = FALSE
)

for (var in Catgorical_vars) {
  contingency_table <- table(variables[[var]], variables$Attrition)
  chisq.test(contingency_table)
  
  chi_test <- chisq.test(contingency_table)
  
  chi_results <- rbind(chi_results, data.frame(
    variable = var,
    Chi_Square = chi_test$statistic,
    P_Value = chi_test$p.value,
    Significant = ifelse(chi_test$p.value < .05, "YES", "NO")
  ))}

chi_results <- chi_results[order(chi_results$P_Value),]

print(chi_results)

#for numerical variables

numeric_vars <- c("Age", "DailyRate", "DistanceFromHome", "HourlyRate", 
                  "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", 
                  "PercentSalaryHike", "TotalWorkingYears", "YearsAtCompany", 
                  "YearsInCurrentRole", "YearsSinceLastPromotion", 
                  "YearsWithCurrManager")

t_results <- data.frame(
  Variable = character(),
  Mean_Attrition_Yes = numeric(),
  Mean_Attrition_No = numeric(),
  T_Statistic = numeric(),
  P_Value = numeric(),
  Significant = character(),
  stringsAsFactors = FALSE
)

for (var in numeric_vars) {
  group_yes <- variables[variables$Attrition == "Yes", var]
  group_no <- variables[variables$Attrition == "No", var]
  t_test <- t.test(group_yes, group_no)
  
  t_results <- rbind(t_results, data.frame(
    Variable = var,
    Mean_Attrition_Yes = mean(group_yes, na.rm = TRUE),
    Mean_Attrition_No = mean(group_no, na.rm = TRUE),
    T_Statistic = t_test$statistic,
    P_Value = t_test$p.value,
    Significant = ifelse(t_test$p.value < .05, "YES", "NO")
  ))
}

t_results <- t_results[(order(t_results$P_Value)), ]

print(t_results)

#Visualization
chi_data <- data.frame(
  Variables = c("OverTime", "StockOptionLevel", "JobRole", "JobInvolvement", "JobLevel",
                "MaritalStatus", "WorkLifeBalance", "Department", "EnvironmentSatisfaction",
                "JobSatisfaction", "BusinessTravel", "TrainingTimesLastYear", "EducationField",
                "RelationshipSatisfaction", "Gender", "Education", "PerformanceRating"),
  P_Value = c(2.332981e-15, 3.724464e-12, 3.646836e-10, 5.211041e-09, 2.084703e-08,
              3.378946e-08, 2.495090e-03, 9.423773e-03, 1.054090e-02,
              1.115122e-02, 4.992536e-02, 1.192044e-01, 2.682198e-01,
              3.727117e-01, 5.151297e-01, 6.242838e-01, 7.461706e-01),
  Type = "Categorical", 
  InModel = c("YES", "YES", "YES", "YES", "YES", "NO", "NO",
              "NO", "NO", "NO", "NO", "NO", "NO", "NO", "NO", "NO", "NO" )
)

#Numerical
t_data <- data.frame(
  Variables = c("MonthlyIncome", "TotalWorkingYears", "YearsInCurrentRole", 
                "YearsWithCurrManager", "Age", "YearsAtCompany", "DistanceFromHome",
                "NumCompaniesWorked", "MonthlyRate", "HourlyRate", 
                "DailyRate", "PercentSalaryHike", "YearsSinceLastPromotion"), 
  P_Value = c(2.412488e-07, 6.595682e-07, 1.522152e-06, 5.084229e-06, 5.049764e-05,
              2.563021e-04, 1.640519e-02, 9.788235e-02, 1.980950e-01,
              2.744798e-01, 3.188749e-01, 6.692297e-01, 8.983165e-01),
  Type = "Numerical",
  InModel = "NO"
)

all_vars <- rbind(chi_data, t_data)
all_vars$NegLog10_P <- -log10(all_vars$P_Value)
all_vars <- all_vars[order(-all_vars$NegLog10_P), ]
top20_vars <- all_vars[1:20, ]

ggplot(top20_vars, aes(x = reorder(Variables, NegLog10_P), y = NegLog10_P, fill = Type)) +
  geom_bar(stat = "identity", width = 0.7, alpha = 0.9) +
  geom_bar(data = top20_vars[top20_vars$InModel == "YES", ], 
           aes(x = reorder(Variables, NegLog10_P), y = NegLog10_P),
           stat = "identity", width = 0.7, fill = NA, color = "black", linewidth = 1.5) +
  coord_flip() +
  geom_hline(yintercept = yintercept = 1.30, linetype = "dashed", color = "red", linewidth = 1.2) +
  geom_text(aes(label = sprintf("%.1e", P_Value)), 
            hjust = -0.1, size = 3.2, fontface = "bold") +
  scale_fill_manual(values = c("Categorical" = "blue", "Numerical" = "green"),
                    name = "Variable Type") +
  scale_y_continuous(breaks = seq(0, 16, 2), limits = c(0, 16)) +
  labs(title = "Variable Significance Analysis",
       x = "Different Variables",
       y = "-log10(P-Value) [Higher = More Statistically Significant]",
       caption = "Red dashed line = Significance Threshold (p = 0.05)") +
  theme_minimal()

#3 (Best Naive Bayes Model)
attrition_cleanTop5 <- attrition_cleanNB3%>%
  select(-EnvironmentSatisfaction, -JobSatisfaction, -MaritalStatus, -MonthlyIncome, -TotalWorkingYears, -WorkLifeBalance, -YearsAtCompany, -YearsInCurrentRole, -YearsWithCurrManager)

set.seed(1)

trainIndices = sample(seq(1:length(attrition_cleanTop5$Age)),round(.7*length(attrition_cleanTop5$Age)))

trainAttrition = attrition_cleanTop5[trainIndices,]
testAttrition = attrition_cleanTop5[-trainIndices,]

model = naiveBayes(trainAttrition[,c("OverTime", "StockOptionLevel", "JobRole", "JobInvolvement", "JobLevel")],
                   factor(trainAttrition$Attrition, labels = c("No", "Yes")))
table(factor(testAttrition$Attrition, labels = c("No", "Yes")),
      predict(model,testAttrition[,c("OverTime", "StockOptionLevel", "JobRole", "JobInvolvement", "JobLevel")]))

confusionMatrix(table(factor(testAttrition$Attrition, labels = c("No", "Yes")),predict(model,testAttrition[,c("OverTime", "StockOptionLevel", "JobRole", "JobInvolvement", "JobLevel")])))


#4
attrition_cleanNB4 <- attrition_cleanNB3%>%
  select(-EnvironmentSatisfaction, -JobSatisfaction, -WorkLifeBalance, -YearsAtCompany)

set.seed(1)

trainIndices = sample(seq(1:length(attrition_cleanNB4$Age)),round(.7*length(attrition_cleanNB4$Age)))

trainAttrition = attrition_cleanNB4[trainIndices,]
testAttrition = attrition_cleanNB4[-trainIndices,]

model = naiveBayes(trainAttrition[,c("OverTime", "StockOptionLevel", "JobRole", "JobInvolvement", "JobLevel", 
                                     "MaritalStatus", "MonthlyIncome", "TotalWorkingYears", "YearsInCurrentRole", "YearsWithCurrManager", "Age")],
                   factor(trainAttrition$Attrition, labels = c("No", "Yes")))
table(factor(testAttrition$Attrition, labels = c("No", "Yes")),
      predict(model,testAttrition[,c("OverTime", "StockOptionLevel", "JobRole", "JobInvolvement", "JobLevel", 
                                     "MaritalStatus", "MonthlyIncome", "TotalWorkingYears", "YearsInCurrentRole", "YearsWithCurrManager", "Age")]))

confusionMatrix(table(factor(testAttrition$Attrition, labels = c("No", "Yes")),
                      predict(model,testAttrition[,c("OverTime", "StockOptionLevel", "JobRole", "JobInvolvement", "JobLevel",
                                                     "MaritalStatus", "MonthlyIncome", "TotalWorkingYears", "YearsInCurrentRole", "YearsWithCurrManager", "Age")])))


#(Using best model- top 5 Predictors) Modify Threshold for 60% Spec.(Run Model #3 First)
library(pROC)

nb_probs <- predict(model, testAttrition[, c("OverTime", "StockOptionLevel", "JobRole", "JobInvolvement", "JobLevel")],
  type = "raw")

p_pos_nbC <- nb_probs[, "Yes"]

roc_nb_fac <- roc(response = testAttrition$Attrition, 
                  predictor = p_pos_nbC, 
                  levels = c("No", "Yes"), direction = "<")
auc_nb_fac <- auc(roc_nb_fac)


roc_nb_fac
auc_nb_fac

plot(roc_nb_fac, main = sprintf("Navie Bayes ROC (factor) AUC = %.3f", as.numeric(auc_nb_fac)))
abline(a = 0, b= 1, lty = 2)

best_thr_nb_fac <- as.numeric(
  coords(roc_nb_fac, x = "best", best.method = "youden", ret = "threshold")
)
best_thr_nb_fac

nb_fac_opt <- factor(ifelse(p_pos_nbC >= best_thr_nb_fac, 
                                 "Yes", "No"),
                          levels = c("Yes", "No"))

confusionMatrix(nb_fac_opt, factor(testAttrition$Attrition, levels = c("Yes", "No")), 
positive = "No")

CM_FNB = confusionMatrix(nb_fac_opt, factor(testAttrition$Attrition, levels = c("Yes", "No")), 
                  positive = "No")

#AUC chart w/ optimal point
plot(roc_nb_fac, main = sprintf("Navie Bayes ROC (factor) - AUC = %.3f", as.numeric(auc_nb_fac)),
     col = "blue", lwd = 2, cex.main = 1.5)
     abline(a = 0, b= 1, lty = 2, col = "black")
     
optimal_coords <- coords(roc_nb_fac, x= "best" , best.method = "youden")
points(1 - optimal_coords$specificity, optimal_coords$sensitivity,
       col = "red", pch = 19, cex = 2)
text(1 - optimal_coords$specificity + .1, optimal_coords$sensitivity,
     paste0("Optimal/nThresh =", round(best_thr_nb_fac, 3),
            "\nSens =", round(optimal_coords$sensitivity*100, 1), "%",
            "\nSpec =", round(optimal_coords$specificity*100, 1), "%"),
     cex = .9)


#KNN
library(class)
library(caret)

set.seed(1)

attrition_KNN <- CaseStudy1.data

attrition_KNN$OverTime_num <- as.numeric(as.factor(attrition_KNN$OverTime))
attrition_KNN$StockOptionLevel_num <- as.numeric(as.factor(attrition_KNN$StockOptionLevel))
attrition_KNN$JobRole_num <- as.numeric(as.factor(attrition_KNN$JobRole))


train_indices = sample(seq(1, nrow(attrition_KNN), 1), round(.7 * nrow(attrition_KNN), ))
train_data = attrition_KNN[train_indices,]
test_data = attrition_KNN[-train_indices,]

#KNN Classifier

classifications = knn(train = train_data[,c("OverTime_num", "StockOptionLevel_num","JobRole_num")], 
                      test = test_data[,c("OverTime_num", "StockOptionLevel_num","JobRole_num")], 
                      cl = train_data$Attrition, prob = TRUE, k = 10)

table(classifications,test_data$Attrition)
confusionMatrix(table(classifications,test_data$Attrition))



#KNN AUC
library(pROC)

knn_probs <- attr(classifications, "prob")

p_pos_knnC <- ifelse(classifications == "Yes", knn_probs, 1 - knn_probs)

roc_knn_num <- roc(response = test_data$Attrition, 
                  predictor = p_pos_knnC, 
                  levels = c("No", "Yes"), direction = "<")

auc_knn_num <- auc(roc_knn_num)


roc_knn_num
auc_knn_num

plot(roc_knn_num, main = sprintf("KNN ROC (factor) AUC = %.3f", as.numeric(auc_knn_num)))
abline(a = 0, b = 1, lty = 2)

best_thr_knn_num <- as.numeric(
  coords(roc_knn_num, x = "best", best.method = "youden", ret = "threshold")
)

best_thr_knn_num

knn_num_opt <- factor(ifelse(p_pos_knnC >= best_thr_knn_num, 
                            "Yes", "No"),
                     levels = c("Yes", "No"))

#levels flipped for youtube presentation to highlight yes... 
confusionMatrix(knn_num_opt, factor(test_data$Attrition, levels = c("No", "Yes")), 
                positive = "Yes")

CM_KNN = confusionMatrix(knn_num_opt, factor(test_data$Attrition, levels = c("No", "Yes")), 
                         positive = "Yes")

#AUC chart w/ optimal point
plot(roc_knn_num, main = sprintf("KNN ROC (Numerical, k=10) - AUC = %.3f", as.numeric(auc_knn_num)),
     col = "blue", lwd = 2, cex.main = 1.5)
abline(a = 0, b= 1, lty = 2, col = "black")

optimal_coords <- coords(roc_knn_num, x= "best" , best.method = "youden")
points(1 - optimal_coords$specificity, optimal_coords$sensitivity,
       col = "red", pch = 19, cex = 2)
text(1 - optimal_coords$specificity + .1, optimal_coords$sensitivity,
     paste0("Optimal/nThresh =", round(best_thr_knn_num, 3),
            "\nSens =", round(optimal_coords$sensitivity*100, 1), "%",
            "\nSpec =", round(optimal_coords$specificity*100, 1), "%"),
     cex = .9)


#Submission.R
comp_data <- CaseStudy1CompSet.No.Attrition

comp_data$StockOptionLevel <- as.factor(comp_data$StockOptionLevel)
comp_data$JobInvolvement <- as.factor(comp_data$JobInvolvement)
comp_data$JobLevel <- as.factor(comp_data$JobLevel)

comp_data$OverTime <- as.factor(comp_data$OverTime)
comp_data$JobRole <- as.factor(comp_data$JobRole)

nb_probs_comp <- predict(model, 
                         comp_data[, c("OverTime", "StockOptionLevel", "JobRole", "JobInvolvement", "JobLevel")],
                         type = "raw")

p_pos_comp <- nb_probs_comp[, "Yes"]

best_threshold <- best_thr_nb_fac

nb_final_predictions <- factor(ifelse(p_pos_comp >= best_threshold,
                                      "Yes", "No"),
                               levels = c("No", "Yes"))

submission <- data.frame(
  ID = comp_data$ID,
  Attrition = nb_final_predictions
)

write.csv(submission, "Case1PredictionsJones_Attrition.csv", row.names = FALSE)
getwd()

