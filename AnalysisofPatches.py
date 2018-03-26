

# Final result will be presented by the following functions

def Output_go_Task1_S1():
    print('\n\n\nTASK 1: Check if Dota2 Gameplay varied by Patch update')
    print('        (Scenario 1: 7.06 - 7.07 - 7.08 - 7.09 - 7.10)')
    print('\n        3 Groups of data: All features, Match results, Match details')
    Output_go_Task1_S1_all()
    Output_go_Task1_S1_res()
    Output_go_Task1_S1_fac()
    

def Output_go_Task1_S2():
    print('\n\n\nTASK 1: Check if Dota2 Gameplay varied by Patch update')
    print('        (Scenario 2: Before and After 7.07 updated)')
    print('\n        3 Groups of data: All features, Match results, Match details')
    Output_go_Task1_S2_all()
    Output_go_Task1_S2_res()
    Output_go_Task1_S2_fac()
    
    
def Output_go_Task2():
    print('\n\n\nTASK 2: Check if Kills/Deaths/Assists values vitally result in the match result')
    print('        (Scenario: Win or Lose of the match)')
    print('\n        3 Groups of data: Match results, Match details, Gold per minute')
    Output_go_Task2_all()
    Output_go_Task2_res()
    Output_go_Task2_fac()
    
    
    
##############################################################   
#    
# Task 1: Check if Dota2 Gameplay varied by Patch update
# Scenario 1: 7.06 - 7.07 - 7.08 - 7.09 - 7.10）
# Models to be applied:
# 
# 1. KMeans cluster                  (perform_conKMClu)
# 2. KMeans cluster with PCA         (perform_conKMClu_PCA)
# 3. Linear Regression               (perform_LnR)
# 4. Linear Regression with PCA      (perform_LnR_PCA)
# 
# Checking: Presentation of T1 S1    (Output_go_Task1_S1)
#
#############################################################
#
# Task 1: Check if Dota2 Gameplay varied by Patch update
# Scenario 2: Before and After 7.07 updated
# Models to be applied:
#
# 1. Logistic Regression          (perform_LoR)
# 2. Logistic Regression with PCA (perform_LoR_PCA)
# 3. Random Forest                (perform_RaF)
# 4. Decision Tree                (perform_DeT)
# 5. KMeans cluster               (perform_KMClu)
# 6. KMeans cluster with PCA      (perform_KMClu_PCA)
# 
# Checking: Presentation of T1 S2    (Output_go_Task1_S2)
#
#############################################################
#
# Task 2: Check if Kills/Deaths/Assists values vitally 
#         result in the game result
# Scenario: Win or Lose of the match
# Models to be applied:
#
# 1. Logistic Regression          (perform_LoR)
# 2. Logistic Regression with PCA (perform_LoR_PCA)
# 3. Random Forest                (perform_RaF)
# 4. Decision Tree                (perform_DeT)
# 5. KMeans cluster               (perform_KMClu)
# 6. KMeans cluster with PCA      (perform_KMClu_PCA)
# 
# Checking: Presentation of T2    (Output_go_Task2)
#

############################ Data Preparation #################################


import pandas as pd

dataset = r'/Users/dafeining/Desktop/Dataset/dotaset2.xlsx'
df = pd.read_excel(dataset)

############### Task 1:
# 3 groups: (1) all (2) results (3) factors of the results
X1_all_df = df[['duration','kills','deaths','assists','stuns',
                'denies','stacks','gpm','herodamage','herohealing',
                'runes','buybacks','wardplaced','warddestroyed']]
X1_res_df = df[['duration','kills','deaths','assists']]
X1_fac_df = df[['stuns','denies','stacks','herodamage','herohealing',
                'runes','buybacks','wardplaced','warddestroyed']]
X1_all = X1_all_df.as_matrix()
X1_res = X1_res_df.as_matrix()
X1_fac = X1_fac_df.as_matrix()

y1 = df[['patch']].as_matrix()
y1_2 = df['patch_1'].as_matrix()

############### Task 2:

X2_res_df = df[['kills','deaths','assists']]
X2_fac_df = df[['stuns','denies','stacks','gpm','herodamage','herohealing',
                'runes','buybacks','wardplaced','warddestroyed']]
X2_res = X2_res_df.as_matrix()
X2_fac = X2_fac_df.as_matrix()
X2_gpm = df[['gpm']].as_matrix()

y2 = df['win'].as_matrix()
###############################################################################


# Functions of applying data in models  
    
    
def perform_LnR (X_input, Y_input):
    from sklearn.model_selection import train_test_split    
    from sklearn.linear_model import LinearRegression    
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    
    mse_sum = 0
    rsq_sum = 0
    rangen = 100
    
    for i in range(rangen):
        X_train, X_test, y_train, y_test = train_test_split(X_input, Y_input)
        lr = LinearRegression()
        lr.fit(X_train,y_train)
        y_pred = lr.predict(X_test)
        
        rsquare = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mse_sum += mse
        rsq_sum += rsquare
    
    MSE = mse_sum/rangen
    Rsquare = rsq_sum/rangen
    
    return [Rsquare, MSE]


def perform_LnR_PCA (X_input, Y_input, PCA_component):
    from sklearn.model_selection import train_test_split    
    from sklearn.linear_model import LinearRegression    
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import r2_score
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    
    mse_sum = 0
    rsq_sum = 0
    rangen = 100
    
    for i in range(rangen):
        X_train, X_test, y_train, y_test = train_test_split(X_input, Y_input)
        scaler = preprocessing.StandardScaler().fit(X_train)
        N_train = scaler.transform(X_train)
        N_test = scaler.transform(X_test)
        pca = PCA(n_components= PCA_component)
        pca.fit(N_train)
        N_train_pca = pca.transform(N_train)
        N_test_pca = pca.transform(N_test)
        
        lr = LinearRegression()
        lr.fit(N_train_pca,y_train)
        y_pred = lr.predict(N_test_pca)
        
        rsquare = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mse_sum += mse
        rsq_sum += rsquare
    
    MSE = mse_sum/rangen
    Rsquare = rsq_sum/rangen
    
    return [Rsquare, MSE]


def perform_conKMCluster (X_input, Y_input):
    from sklearn.cluster import KMeans
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn import preprocessing
    
    fnr_sum = 0
    accu_sum = 0
    rangen = 50
    
    for i in range(rangen):
        k_means = KMeans(init='k-means++', n_clusters=5)
        k_means.fit(X_input)
        labels = k_means.labels_

        labelsc = preprocessing.scale(labels)
        Ysc = preprocessing.scale(Y_input)
        
        fnr_i = mean_squared_error(Ysc, labelsc)
        fnr_sum += fnr_i
        accu_i = r2_score(Ysc, labelsc)
        accu_sum += accu_i
    
    MSE = fnr_sum/rangen
    RSquare = accu_sum/rangen
    
    return [RSquare, MSE]


def perform_conKMCluster_PCA (X_input, Y_input, PCA_c):
    from sklearn.cluster import KMeans
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    
    fnr_sum = 0
    accu_sum = 0
    rangen = 50
    
    for i in range(rangen):
        scaler = preprocessing.StandardScaler().fit(X_input)
        X_N = scaler.transform(X_input)
        pca = PCA(n_components=PCA_c) 
        pca.fit(X_N)
        X_pca = pca.transform(X_N)
        
        k_means = KMeans(init='k-means++', n_clusters=5)
        k_means.fit(X_pca)
        labels = k_means.labels_

        labelsc = preprocessing.scale(labels)
        Ysc = preprocessing.scale(Y_input)
        
        fnr_i = mean_squared_error(Ysc, labelsc)
        fnr_sum += fnr_i
        accu_i = r2_score(Ysc, labelsc)
        accu_sum += accu_i
    
    MSE = fnr_sum/rangen
    RSquare = accu_sum/rangen
    
    return [RSquare, MSE]


def perform_LoR (X_input, Y_input):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    
    lr_fnr_sum = 0
    lr_accu_sum = 0
    rangen = 100
    
    for i in range(rangen):
        X_train, X_test, y_train, y_test = train_test_split(X_input, Y_input)
        lr = LogisticRegression()
        lr.fit(X_train,y_train)
        y_pred_lr = lr.predict(X_test)
        
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        lr_fnr_i = cm_lr[1][0]/(cm_lr[1][1]+cm_lr[1][0])
        lr_fnr_sum += lr_fnr_i
        lr_accu_i = accuracy_score(y_test, y_pred_lr)
        lr_accu_sum += lr_accu_i
    
    FNR = lr_fnr_sum/rangen
    Accuracy = lr_accu_sum/rangen
    
    return [Accuracy, FNR]


def perform_LoR_PCA (X_input, Y_input, PCA_component):
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    
    lr_fnr_sum_pca = 0
    lr_accu_sum_pca = 0
    rangen = 100
    
    for i in range(rangen):        
        X_train, X_test, y_train, y_test = train_test_split(X_input, Y_input)
        scaler = preprocessing.StandardScaler().fit(X_train)
        N_train = scaler.transform(X_train)
        N_test = scaler.transform(X_test)
        pca = PCA(n_components= PCA_component)
        pca.fit(N_train)
        N_train_pca = pca.transform(N_train)
        N_test_pca = pca.transform(N_test)
     
        lr = LogisticRegression()
        lr.fit(N_train_pca,y_train)
        y_pred_lr = lr.predict(N_test_pca)
        
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        lr_fnr_i = cm_lr[1][0]/(cm_lr[1][1]+cm_lr[1][0])
        lr_fnr_sum_pca += lr_fnr_i
        lr_accu_i = accuracy_score(y_test, y_pred_lr)
        lr_accu_sum_pca += lr_accu_i
    
    FNR = lr_fnr_sum_pca/rangen
    Accuracy = lr_accu_sum_pca/rangen
        
    return [Accuracy, FNR]


def perform_RaF (X_input, Y_input):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    
    rf_fnr_sum = 0
    rf_accu_sum = 0
    rangen = 150
    
    for i in range(rangen):
        X_train, X_test, y_train, y_test = train_test_split(X_input, Y_input)
        rf = RandomForestClassifier()
        rf.fit(X_train,y_train)
        y_pred_rf = rf.predict(X_test)
        
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        rf_fnr_i = cm_rf[1][0]/(cm_rf[1][1]+cm_rf[1][0])
        rf_fnr_sum += rf_fnr_i
        rf_accu_i = accuracy_score(y_test, y_pred_rf)
        rf_accu_sum += rf_accu_i
    
    FNR = rf_fnr_sum/rangen
    Accuracy = rf_accu_sum/rangen
    
    return [Accuracy, FNR]


def perform_DeT (X_input,y):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    
    fnr_sum = 0
    accu_sum = 0
    rangen = 150
    
    for i in range(rangen):
        X_train, X_test, y_train, y_test = train_test_split(X_input, y)
        dt = DecisionTreeClassifier()
        dt.fit(X_train,y_train)
        y_pred_dt = dt.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred_dt)
        fnr_i = cm[1][0]/(cm[1][1]+cm[1][0])
        fnr_sum += fnr_i
        accu_i = accuracy_score(y_test, y_pred_dt)
        accu_sum += accu_i
    
    FNR = fnr_sum/rangen
    Accuracy = accu_sum/rangen
    
    return [Accuracy, FNR]


def perform_KMCluster (X_input, y):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.cluster import KMeans
    
    fnr_sum = 0
    accu_sum = 0
    rangen = 60
    
    for i in range(rangen):
        k_means = KMeans(init='k-means++', n_clusters=2)
        k_means.fit(X_input)
        labels = k_means.labels_
        cm = confusion_matrix(y, labels)
        
        fnr_i = cm[1][0]/(cm[1][1]+cm[1][0])
        fnr_sum += fnr_i
        accu_i = accuracy_score(y, labels)
        accu_sum += accu_i
    
    FNR = fnr_sum/rangen
    Accuracy = accu_sum/rangen
    
    return [Accuracy, FNR]
        
        
def perform_KMCluster_PCA (X_input, y, PCA_component):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn.cluster import KMeans
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    
    fnr_sum = 0
    accu_sum = 0
    rangen = 60
    
    for i in range(rangen):
        scaler = preprocessing.StandardScaler().fit(X_input)
        X_N = scaler.transform(X_input)
        pca = PCA(n_components=PCA_component) 
        pca.fit(X_N)
        X_pca = pca.transform(X_N)
        k_means = KMeans(init='k-means++', n_clusters=2)
        k_means.fit(X_pca)
        labels = k_means.labels_
        cm = confusion_matrix(y, labels)
        
        fnr_i = cm[1][0]/(cm[1][1]+cm[1][0])
        fnr_sum += fnr_i
        accu_i = accuracy_score(y, labels)
        accu_sum += accu_i
    
    FNR = fnr_sum/rangen
    Accuracy = accu_sum/rangen
    
    return [Accuracy, FNR]



# For Presentation of T1 S1

def Output_go_Task1_S1_all():
    feature = X1_all
    outcome = y1
    cpn = 10
    print('\n1.1.1. Testing-->All features    PCA#:',cpn)
    print('Model                          RSquare        MeanSquareError')
    print('-------------------------------------------------------------')
    print('Linear Regression             ','%.4f'%perform_LnR(feature, outcome)[0],'       ','%.4f'%perform_LnR(feature, outcome)[1])
    print('Linear Regression(with PCA)   ','%.4f'%perform_LnR_PCA(feature, outcome,cpn)[0],'       ','%.4f'%perform_LnR_PCA(feature, outcome,cpn)[1])
    print('KMeans cluster                ','%.4f'%perform_conKMCluster(feature,outcome)[0],'      ','%.4f'%perform_conKMCluster(feature,outcome)[1])
    print('KMeans cluster(with PCA)      ','%.4f'%perform_conKMCluster_PCA(feature,outcome,cpn)[0],'      ','%.4f'%perform_conKMCluster_PCA(feature, outcome, cpn)[1])
    print('=============================================================')

def Output_go_Task1_S1_res():
    feature = X1_res
    outcome = y1
    cpn = 4
    print('\n1.1.2. Testing-->Match results data    PCA#:',cpn)
    print('Model                          RSquare        MeanSquareError')
    print('-------------------------------------------------------------')
    print('Linear Regression             ','%.4f'%perform_LnR(feature, outcome)[0],'       ','%.4f'%perform_LnR(feature, outcome)[1])
    print('Linear Regression(with PCA)   ','%.4f'%perform_LnR_PCA(feature, outcome,cpn)[0],'       ','%.4f'%perform_LnR_PCA(feature, outcome,cpn)[1])
    print('KMeans cluster                ','%.4f'%perform_conKMCluster(feature,outcome)[0],'      ','%.4f'%perform_conKMCluster(feature,outcome)[1])
    print('KMeans cluster(with PCA)      ','%.4f'%perform_conKMCluster_PCA(feature,outcome,cpn)[0],'      ','%.4f'%perform_conKMCluster_PCA(feature, outcome, cpn)[1])
    print('=============================================================')
    
def Output_go_Task1_S1_fac():
    feature = X1_fac
    outcome = y1
    cpn = 9
    print('\n1.1.3. Testing-->Match details data    PCA#:',cpn)
    print('Model                          RSquare        MeanSquareError')
    print('-------------------------------------------------------------')
    print('Linear Regression             ','%.4f'%perform_LnR(feature, outcome)[0],'       ','%.4f'%perform_LnR(feature, outcome)[1])
    print('Linear Regression(with PCA)   ','%.4f'%perform_LnR_PCA(feature, outcome,cpn)[0],'       ','%.4f'%perform_LnR_PCA(feature, outcome,cpn)[1])
    print('KMeans cluster                ','%.4f'%perform_conKMCluster(feature,outcome)[0],'      ','%.4f'%perform_conKMCluster(feature,outcome)[1])
    print('KMeans cluster(with PCA)      ','%.4f'%perform_conKMCluster_PCA(feature,outcome,cpn)[0],'      ','%.4f'%perform_conKMCluster_PCA(feature, outcome, cpn)[1])
    print('=============================================================')
    

# For Presentation of T1 S2
    
def Output_go_Task1_S2_all():
    feature = X1_all
    outcome = y1_2
    cpn = 10
    print('\n1.2.1. Testing-->All features    PCA#:',cpn)
    print('Model                          Accuracy       FalseNegRate')
    print('----------------------------------------------------------')
    print('Logistic Regression           ','%.4f'%perform_LoR(feature, outcome)[0],'       ','%.4f'%perform_LoR(feature, outcome)[1])
    print('Logistic Regression(with PCA) ','%.4f'%perform_LoR_PCA(feature, outcome, cpn)[0],'       ','%.4f'%perform_LoR_PCA(feature, outcome, cpn)[1])
    print('Random Forest                 ','%.4f'%perform_RaF(feature, outcome)[0],'       ','%.4f'%perform_RaF(feature, outcome)[1])
    print('Decision Tree                 ','%.4f'%perform_DeT(feature, outcome)[0],'       ','%.4f'%perform_DeT(feature, outcome)[1])
    print('KMeans cluster                ','%.4f'%perform_KMCluster(feature, outcome)[0],'       ','%.4f'%perform_KMCluster(feature, outcome)[1])
    print('KMeans cluster(with PCA)      ','%.4f'%perform_KMCluster_PCA(feature, outcome, cpn)[0],'       ','%.4f'%perform_KMCluster_PCA(feature, outcome, cpn)[1])  
    print('==========================================================')

def Output_go_Task1_S2_res():
    feature = X1_res
    outcome = y1_2
    cpn = 4
    print('\n1.2.2. Testing-->Match results data   PCA#:',cpn)
    print('Model                          Accuracy       FalseNegRate')
    print('----------------------------------------------------------')
    print('Logistic Regression           ','%.4f'%perform_LoR(feature, outcome)[0],'       ','%.4f'%perform_LoR(feature, outcome)[1])
    print('Logistic Regression(with PCA) ','%.4f'%perform_LoR_PCA(feature, outcome, cpn)[0],'       ','%.4f'%perform_LoR_PCA(feature, outcome, cpn)[1])
    print('Random Forest                 ','%.4f'%perform_RaF(feature, outcome)[0],'       ','%.4f'%perform_RaF(feature, outcome)[1])
    print('Decision Tree                 ','%.4f'%perform_DeT(feature, outcome)[0],'       ','%.4f'%perform_DeT(feature, outcome)[1])
    print('KMeans cluster                ','%.4f'%perform_KMCluster(feature, outcome)[0],'       ','%.4f'%perform_KMCluster(feature, outcome)[1])
    print('KMeans cluster(with PCA)      ','%.4f'%perform_KMCluster_PCA(feature, outcome, cpn)[0],'       ','%.4f'%perform_KMCluster_PCA(feature, outcome, cpn)[1]) 
    print('==========================================================')

def Output_go_Task1_S2_fac():
    feature = X1_fac
    outcome = y1_2
    cpn = 9
    print('\n1.2.3. Testing-->Match details data    PCA#:',cpn)
    print('Model                          Accuracy       FalseNegRate')
    print('----------------------------------------------------------')
    print('Logistic Regression           ','%.4f'%perform_LoR(feature, outcome)[0],'       ','%.4f'%perform_LoR(feature, outcome)[1])
    print('Logistic Regression(with PCA) ','%.4f'%perform_LoR_PCA(feature, outcome, cpn)[0],'       ','%.4f'%perform_LoR_PCA(feature, outcome, cpn)[1])
    print('Random Forest                 ','%.4f'%perform_RaF(feature, outcome)[0],'       ','%.4f'%perform_RaF(feature, outcome)[1])
    print('Decision Tree                 ','%.4f'%perform_DeT(feature, outcome)[0],'       ','%.4f'%perform_DeT(feature, outcome)[1])
    print('KMeans cluster                ','%.4f'%perform_KMCluster(feature, outcome)[0],'       ','%.4f'%perform_KMCluster(feature, outcome)[1])
    print('KMeans cluster(with PCA)      ','%.4f'%perform_KMCluster_PCA(feature, outcome, cpn)[0],'       ','%.4f'%perform_KMCluster_PCA(feature, outcome, cpn)[1]) 
    print('==========================================================')
    
    
# For Presentation of T2
    
def Output_go_Task2_all():
    feature = X2_res
    outcome = y2
    cpn = 3
    print('\n1.2.1. Testing-->Match results    PCA#:',cpn)
    print('Model                          Accuracy       FalseNegRate')
    print('----------------------------------------------------------')
    print('Logistic Regression           ','%.4f'%perform_LoR(feature, outcome)[0],'       ','%.4f'%perform_LoR(feature, outcome)[1])
    print('Logistic Regression(with PCA) ','%.4f'%perform_LoR_PCA(feature, outcome, cpn)[0],'       ','%.4f'%perform_LoR_PCA(feature, outcome, cpn)[1])
    print('Random Forest                 ','%.4f'%perform_RaF(feature, outcome)[0],'       ','%.4f'%perform_RaF(feature, outcome)[1])
    print('Decision Tree                 ','%.4f'%perform_DeT(feature, outcome)[0],'       ','%.4f'%perform_DeT(feature, outcome)[1])
    print('KMeans cluster                ','%.4f'%perform_KMCluster(feature, outcome)[0],'       ','%.4f'%perform_KMCluster(feature, outcome)[1])
    print('KMeans cluster(with PCA)      ','%.4f'%perform_KMCluster_PCA(feature, outcome, cpn)[0],'       ','%.4f'%perform_KMCluster_PCA(feature, outcome, cpn)[1])  
    print('==========================================================')

def Output_go_Task2_res():
    feature = X2_fac
    outcome = y2
    cpn = 10
    print('\n1.2.2. Testing-->Match details  PCA#:',cpn)
    print('Model                          Accuracy       FalseNegRate')
    print('----------------------------------------------------------')
    print('Logistic Regression           ','%.4f'%perform_LoR(feature, outcome)[0],'       ','%.4f'%perform_LoR(feature, outcome)[1])
    print('Logistic Regression(with PCA) ','%.4f'%perform_LoR_PCA(feature, outcome, cpn)[0],'       ','%.4f'%perform_LoR_PCA(feature, outcome, cpn)[1])
    print('Random Forest                 ','%.4f'%perform_RaF(feature, outcome)[0],'       ','%.4f'%perform_RaF(feature, outcome)[1])
    print('Decision Tree                 ','%.4f'%perform_DeT(feature, outcome)[0],'       ','%.4f'%perform_DeT(feature, outcome)[1])
    print('KMeans cluster                ','%.4f'%perform_KMCluster(feature, outcome)[0],'       ','%.4f'%perform_KMCluster(feature, outcome)[1])
    print('KMeans cluster(with PCA)      ','%.4f'%perform_KMCluster_PCA(feature, outcome, cpn)[0],'       ','%.4f'%perform_KMCluster_PCA(feature, outcome, cpn)[1]) 
    print('==========================================================')

def Output_go_Task2_fac():
    feature = X2_gpm
    outcome = y2
    print('\n1.2.3. Testing-->Gold per minute   ')
    print('Model                          Accuracy       FalseNegRate')
    print('----------------------------------------------------------')
    print('Logistic Regression           ','%.4f'%perform_LoR(feature, outcome)[0],'       ','%.4f'%perform_LoR(feature, outcome)[1])
    print('Random Forest                 ','%.4f'%perform_RaF(feature, outcome)[0],'       ','%.4f'%perform_RaF(feature, outcome)[1])
    print('Decision Tree                 ','%.4f'%perform_DeT(feature, outcome)[0],'       ','%.4f'%perform_DeT(feature, outcome)[1])
    print('KMeans cluster                ','%.4f'%perform_KMCluster(feature, outcome)[0],'       ','%.4f'%perform_KMCluster(feature, outcome)[1]) 
    print('==========================================================')
