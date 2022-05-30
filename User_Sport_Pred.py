# import modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
import json
import os

# import variables
mRegression_empty = pd.read_csv('./PredVariables/mRegression_empty.csv')
fRegression_empty = pd.read_csv('./PredVariables/fRegression_empty.csv')

# import logistic regression models
fArchery_clf = pickle.load(open('./H5/fArchery.pkl', 'rb'))
fAthletics_clf = pickle.load(open('./H5/fAthletics.pkl', 'rb'))
fBadminton_clf = pickle.load(open('./H5/fBadminton.pkl', 'rb'))
fBasketball_clf = pickle.load(open('./H5/fBasketball.pkl', 'rb'))
fBeach_Volleyball_clf = pickle.load(open('./H5/fBeach_Volleyball.pkl', 'rb'))
fBoxing_clf = pickle.load(open('./H5/fBoxing.pkl', 'rb'))
fCanoeing_clf = pickle.load(open('./H5/fCycling.pkl', 'rb'))
fCycling_clf = pickle.load(open('./H5/fCanoeing.pkl', 'rb'))
fDiving_clf = pickle.load(open('./H5/fDiving.pkl', 'rb'))
fEquestrianism_clf = pickle.load(open('./H5/fEquestrianism.pkl', 'rb'))
fFencing_clf = pickle.load(open('./H5/fFencing.pkl', 'rb'))
fFootball_clf = pickle.load(open('./H5/fFootball.pkl', 'rb'))
fGolf_clf = pickle.load(open('./H5/fGolf.pkl', 'rb'))
fGymnastics_clf = pickle.load(open('./H5/fGymnastics.pkl', 'rb'))
fHandball_clf = pickle.load(open('./H5/fHandball.pkl', 'rb'))
fHockey_clf = pickle.load(open('./H5/fArchery.pkl', 'rb'))
fJudo_clf = pickle.load(open('./H5/fJudo.pkl', 'rb'))
fModern_Pentathlon_clf = pickle.load(open('./H5/fModern_Pentathlon.pkl', 'rb'))
fRhythmic_Gymnastics_clf = pickle.load(open('./H5/fRhythmic_Gymnastics.pkl', 'rb'))
fRowing_clf = pickle.load(open('./H5/fRowing.pkl', 'rb'))
fRugby_Sevens_clf = pickle.load(open('./H5/fRugby_Sevens.pkl', 'rb'))
fShooting_clf = pickle.load(open('./H5/fShooting.pkl', 'rb'))
fSoftball_clf = pickle.load(open('./H5/fSoftball.pkl', 'rb'))
fSwimming_clf = pickle.load(open('./H5/fRowing.pkl', 'rb'))
fSynchronized_Swimming_clf = pickle.load(open('./H5/fSynchronized_Swimming.pkl', 'rb'))
fTable_Tennis_clf = pickle.load(open('./H5/fTable_Tennis.pkl', 'rb'))
fTaekwondo_clf = pickle.load(open('./H5/fTaekwondo.pkl', 'rb'))
fTennis_clf = pickle.load(open('./H5/fTennis.pkl', 'rb'))
fTrampolining_clf = pickle.load(open('./H5/fTrampolining.pkl', 'rb'))
fTriathlon_clf = pickle.load(open('./H5/fTriathlon.pkl', 'rb'))
fVolleyball_clf = pickle.load(open('./H5/fVolleyball.pkl', 'rb'))
fWater_Polo_clf = pickle.load(open('./H5/fWater_Polo.pkl', 'rb'))
fWeightlifting_clf = pickle.load(open('./H5/fWeightlifting.pkl', 'rb'))
fWrestling_clf = pickle.load(open('./H5/fWrestling.pkl', 'rb'))

mArchery_clf = pickle.load(open('./H5/mArchery.pkl', 'rb'))
mAthletics_clf = pickle.load(open('./H5/mAthletics.pkl', 'rb'))
mBadminton_clf = pickle.load(open('./H5/mBadminton.pkl', 'rb'))
mBaseball_clf = pickle.load(open('./H5/mBaseball.pkl', 'rb'))
mBasketball_clf = pickle.load(open('./H5/mBasketball.pkl', 'rb'))
mBeach_Volleyball_clf = pickle.load(open('./H5/mBeach_Volleyball.pkl', 'rb'))
mBoxing_clf = pickle.load(open('./H5/mBoxing.pkl', 'rb'))
mCycling_clf = pickle.load(open('./H5/mCycling.pkl', 'rb'))
mCanoeing_clf = pickle.load(open('./H5/mCanoeing.pkl', 'rb'))
mDiving_clf = pickle.load(open('./H5/mDiving.pkl', 'rb'))
mEquestrianism_clf = pickle.load(open('./H5/mEquestrianism.pkl', 'rb'))
mFencing_clf = pickle.load(open('./H5/mFencing.pkl', 'rb'))
mFootball_clf = pickle.load(open('./H5/mFootball.pkl', 'rb'))
mGolf_clf = pickle.load(open('./H5/mGolf.pkl', 'rb'))
mGymnastics_clf = pickle.load(open('./H5/mGymnastics.pkl', 'rb'))
mHandball_clf = pickle.load(open('./H5/mHandball.pkl', 'rb'))
mHockey_clf = pickle.load(open('./H5/mArchery.pkl', 'rb'))
mJudo_clf = pickle.load(open('./H5/mJudo.pkl', 'rb'))
mModern_Pentathlon_clf = pickle.load(open('./H5/mModern_Pentathlon.pkl', 'rb'))
mRowing_clf = pickle.load(open('./H5/mRowing.pkl', 'rb'))
mRugby_Sevens_clf = pickle.load(open('./H5/mRugby_Sevens.pkl', 'rb'))
mShooting_clf = pickle.load(open('./H5/mShooting.pkl', 'rb'))
mSwimming_clf = pickle.load(open('./H5/mRowing.pkl', 'rb'))
mTable_Tennis_clf = pickle.load(open('./H5/mTable_Tennis.pkl', 'rb'))
mTaekwondo_clf = pickle.load(open('./H5/mTaekwondo.pkl', 'rb'))
mTennis_clf = pickle.load(open('./H5/mTennis.pkl', 'rb'))
mTrampolining_clf = pickle.load(open('./H5/mTrampolining.pkl', 'rb'))
mTriathlon_clf = pickle.load(open('./H5/mTriathlon.pkl', 'rb'))
mVolleyball_clf = pickle.load(open('./H5/mVolleyball.pkl', 'rb'))
mWater_Polo_clf = pickle.load(open('./H5/mWater_Polo.pkl', 'rb'))
mWeightlifting_clf = pickle.load(open('./H5/mWeightlifting.pkl', 'rb'))
mWrestling_clf = pickle.load(open('./H5/mWrestling.pkl', 'rb'))

# Recreate bins
bin_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

mAge_bins = np.array([12, 18.55555556, 25.11111111, 31.66666667, 38.22222222, 44.77777778, 51.33333333, 57.88888889, 64.44444444, 71])
mHeight_bins = np.array([127, 138, 149, 160, 171, 182, 193, 204, 215, 226])
mWeight_bins = np.array([ 37, 56.66666667, 76.33333333, 96, 115.66666667, 135.33333333, 155, 174.66666667, 194.33333333, 214])

fAge_bins = np.array([11, 17.44444444, 23.88888889, 30.33333333, 36.77777778, 43.22222222, 49.66666667, 56.11111111, 62.55555556, 69])
fHeight_bins = np.array([127, 136.55555556, 146.11111111, 155.66666667, 165.22222222, 174.77777778, 184.33333333, 193.88888889, 203.44444444, 213])
fWeight_bins = np.array([25, 40.77777778, 56.55555556, 72.33333333, 88.11111111, 103.88888889, 119.66666667, 135.44444444, 151.22222222, 167])

# user_Sex = 'M'
# user_Age = 30
# user_Height = 185
# user_Weight = 77
# user_NOC = 'USA'


def user_predictor(Sex, Age, Height, Weight, NOC):
    
    user_df = pd.DataFrame(
        {
         "Sex": [Sex],
         "Age": [Age],
         "Height": [Height],
         "Weight": [Weight],
         "NOC": [NOC]
        })


    if Sex == 'M':


        user_df['Age'] = user_df['Age'].astype(float, errors = 'raise')
        user_df['Height'] = user_df['Height'].astype(float, errors = 'raise')
        user_df['Weight'] = user_df['Weight'].astype(float, errors = 'raise')

        user_df["Age"] = pd.cut(user_df["Age"], mAge_bins, labels=bin_names)
        user_df["Height"] = pd.cut(user_df["Height"], mHeight_bins, labels=bin_names)
        user_df["Weight"] = pd.cut(user_df["Weight"], mWeight_bins, labels=bin_names)
        user_df["NOC"] = 'NOC_' + user_df["NOC"].astype(str)

        user_df["Age"] = 'Age_' + user_df["Age"].astype(str)
        user_df["Height"] = 'Height_' + user_df["Height"].astype(str)
        user_df["Weight"] = 'Weight_' + user_df["Weight"].astype(str)

        dummy_col = list(mRegression_empty.columns.values.tolist())

        user_dummies_data = []

        for col in dummy_col:
            if col == user_df.at[0, 'Age']:
                user_dummies_data.append(1)
            elif col == user_df.at[0, 'Height']:
                user_dummies_data.append(1)
            elif col == user_df.at[0, 'Weight']:
                user_dummies_data.append(1)
            elif col == user_df.at[0, 'NOC']:
                user_dummies_data.append(1)
            else: user_dummies_data.append(0)

        user_df_reg = mRegression_empty.append(pd.Series(user_dummies_data, index=dummy_col), ignore_index=True)
        user_df_reg = user_df_reg.drop(['Unnamed: 0'], axis=1)

        msports = []
        user_pred_list = []

        mBasketball_user_pred = mBasketball_clf.predict_proba(user_df_reg)
        mBasketball_user_pred = mBasketball_user_pred[:,1]
        msports.append('Basketball')
        user_pred_list.append(mBasketball_user_pred)

        mJudo_user_pred = mJudo_clf.predict_proba(user_df_reg)
        mJudo_user_pred = mJudo_user_pred[:,1]
        msports.append('Judo')
        user_pred_list.append(mJudo_user_pred)

        mBadminton_user_pred = mBadminton_clf.predict_proba(user_df_reg)
        mBadminton_user_pred = mBadminton_user_pred[:,1]
        msports.append('Badminton')
        user_pred_list.append(mBadminton_user_pred)

        mAthletics_user_pred = mAthletics_clf.predict_proba(user_df_reg)
        mAthletics_user_pred = mAthletics_user_pred[:,1]
        msports.append('Athletics')
        user_pred_list.append(mAthletics_user_pred)

        mWeightlifting_user_pred = mWeightlifting_clf.predict_proba(user_df_reg)
        mWeightlifting_user_pred = mWeightlifting_user_pred[:,1]
        msports.append('Weightlifting')
        user_pred_list.append(mWeightlifting_user_pred)

        mWrestling_user_pred = mWrestling_clf.predict_proba(user_df_reg)
        mWrestling_user_pred = mWrestling_user_pred[:,1]
        msports.append('Wrestling')
        user_pred_list.append(mWrestling_user_pred)

        mRowing_user_pred = mRowing_clf.predict_proba(user_df_reg)
        mRowing_user_pred = mRowing_user_pred[:,1]
        msports.append('Rowing')
        user_pred_list.append(mRowing_user_pred)

        mSwimming_user_pred = mSwimming_clf.predict_proba(user_df_reg)
        mSwimming_user_pred = mSwimming_user_pred[:,1]
        msports.append('Swimming')
        user_pred_list.append(mSwimming_user_pred)

        mFootball_user_pred = mFootball_clf.predict_proba(user_df_reg)
        mFootball_user_pred = mFootball_user_pred[:,1]
        msports.append('Football')
        user_pred_list.append(mFootball_user_pred)

        mEquestrianism_user_pred = mEquestrianism_clf.predict_proba(user_df_reg)
        mEquestrianism_user_pred = mEquestrianism_user_pred[:,1]
        msports.append('Equestrianism')
        user_pred_list.append(mEquestrianism_user_pred)

        mShooting_user_pred = mShooting_clf.predict_proba(user_df_reg)
        mShooting_user_pred = mShooting_user_pred[:,1]
        msports.append('Shooting')
        user_pred_list.append(mShooting_user_pred)

        mGymnastics_user_pred = mGymnastics_clf.predict_proba(user_df_reg)
        mGymnastics_user_pred = mGymnastics_user_pred[:,1]
        msports.append('Gymnastics')
        user_pred_list.append(mGymnastics_user_pred)

        mTaekwondo_user_pred = mTaekwondo_clf.predict_proba(user_df_reg)
        mTaekwondo_user_pred = mTaekwondo_user_pred[:,1]
        msports.append('Taekwondo')
        user_pred_list.append(mTaekwondo_user_pred)

        mBoxing_user_pred = mBoxing_clf.predict_proba(user_df_reg)
        mBoxing_user_pred = mBoxing_user_pred[:,1]
        msports.append('Boxing')
        user_pred_list.append(mBoxing_user_pred)

        mFencing_user_pred = mFencing_clf.predict_proba(user_df_reg)
        mFencing_user_pred = mFencing_user_pred[:,1]
        msports.append('Fencing')
        user_pred_list.append(mFencing_user_pred)

        mDiving_user_pred = mDiving_clf.predict_proba(user_df_reg)
        mDiving_user_pred = mDiving_user_pred[:,1]
        msports.append('Diving')
        user_pred_list.append(mDiving_user_pred)

        mCanoeing_user_pred = mCanoeing_clf.predict_proba(user_df_reg)
        mCanoeing_user_pred = mCanoeing_user_pred[:,1]
        msports.append('Canoeing')
        user_pred_list.append(mCanoeing_user_pred)

        mHandball_user_pred = mHandball_clf.predict_proba(user_df_reg)
        mHandball_user_pred = mHandball_user_pred[:,1]
        msports.append('Handball')
        user_pred_list.append(mHandball_user_pred)

        mWater_Polo_user_pred = mWater_Polo_clf.predict_proba(user_df_reg)
        mWater_Polo_user_pred = mWater_Polo_user_pred[:,1]
        msports.append('Water Polo')
        user_pred_list.append(mWater_Polo_user_pred)

        mTennis_user_pred = mTennis_clf.predict_proba(user_df_reg)
        mTennis_user_pred = mTennis_user_pred[:,1]
        msports.append('Tennis')
        user_pred_list.append(mTennis_user_pred)

        mCycling_user_pred = mCycling_clf.predict_proba(user_df_reg)
        mCycling_user_pred = mCycling_user_pred[:,1]
        msports.append('Cycling')
        user_pred_list.append(mCycling_user_pred)

        mHockey_user_pred = mHockey_clf.predict_proba(user_df_reg)
        mHockey_user_pred = mHockey_user_pred[:,1]
        msports.append('Hockey')
        user_pred_list.append(mHockey_user_pred)

        mArchery_user_pred = mArchery_clf.predict_proba(user_df_reg)
        mArchery_user_pred = mArchery_user_pred[:,1]
        msports.append('Archery')
        user_pred_list.append(mArchery_user_pred)

        mVolleyball_user_pred = mVolleyball_clf.predict_proba(user_df_reg)
        mVolleyball_user_pred = mVolleyball_user_pred[:,1]
        msports.append('Volleyball')
        user_pred_list.append(mVolleyball_user_pred)

        mModern_Pentathlon_user_pred = mModern_Pentathlon_clf.predict_proba(user_df_reg)
        mModern_Pentathlon_user_pred = mModern_Pentathlon_user_pred[:,1]
        msports.append('Modern Pentathlon')
        user_pred_list.append(mModern_Pentathlon_user_pred)

        mTable_Tennis_user_pred = mTable_Tennis_clf.predict_proba(user_df_reg)
        mTable_Tennis_user_pred = mTable_Tennis_user_pred[:,1]
        msports.append('Table Tennis')
        user_pred_list.append(mTable_Tennis_user_pred)

        mBaseball_user_pred = mBaseball_clf.predict_proba(user_df_reg)
        mBaseball_user_pred = mBaseball_user_pred[:,1]
        msports.append('Baseball')
        user_pred_list.append(mBaseball_user_pred)

        mRugby_Sevens_user_pred = mRugby_Sevens_clf.predict_proba(user_df_reg)
        mRugby_Sevens_user_pred = mRugby_Sevens_user_pred[:,1]
        msports.append('Rugby Sevens')
        user_pred_list.append(mRugby_Sevens_user_pred)

        mTrampolining_user_pred = mTrampolining_clf.predict_proba(user_df_reg)
        mTrampolining_user_pred = mTrampolining_user_pred[:,1]
        msports.append('Trampolining')
        user_pred_list.append(mTrampolining_user_pred)

        mBeach_Volleyball_user_pred = mBeach_Volleyball_clf.predict_proba(user_df_reg)
        mBeach_Volleyball_user_pred = mBeach_Volleyball_user_pred[:,1]
        msports.append('Beach Volleyball')
        user_pred_list.append(mBeach_Volleyball_user_pred)

        mTriathlon_user_pred = mTriathlon_clf.predict_proba(user_df_reg)
        mTriathlon_user_pred = mTriathlon_user_pred[:,1]
        msports.append('Triathlon')
        user_pred_list.append(mTriathlon_user_pred)

        mGolf_user_pred = mGolf_clf.predict_proba(user_df_reg)
        mGolf_user_pred = mGolf_user_pred[:,1]
        msports.append('Golf')
        user_pred_list.append(mGolf_user_pred)

        # Find max predicted sports

        user_max = max(user_pred_list)
        user_max_pos = user_pred_list.index(user_max)
        user_max_sport = msports[user_max_pos]
        
        two_user_pred_x = user_pred_list.remove(user_max)
        two_msports_x = msports.remove(user_max_sport)

        two_max = max(two_user_pred_x)
        two_max_pos = two_user_pred_x.index(two_max)
        two_max_sport = two_msports_x[two_max_pos]

        three_user_pred_x = two_user_pred_x.remove(two_max)
        three_msports_x = two_msports_x.remove(two_max_sport)

        three_max = max(three_user_pred_x)
        three_max_pos = three_user_pred_x.index(three_max)
        three_max_sport = three_msports_x[three_max_pos]       

        # Find min predicted sports

        user_min = min(user_pred_list)
        user_min_pos = user_pred_list.index(user_min)
        user_min_sport = msports[user_min_pos]
        
        two_user_pred_n = user_pred_list.remove(user_min)
        two_msports_n = msports.remove(user_min_sport)

        two_min = min(two_user_pred_n)
        two_min_pos = two_user_pred_n.index(two_min)
        two_min_sport = two_msports_n[two_min_pos]

        three_user_pred_n = two_user_pred_n.remove(two_min)
        three_msports_n = two_msports_n.remove(two_min_sport)

        three_min = min(three_user_pred_n)
        three_min_pos = three_user_pred_n.index(three_min)
        three_min_sport = three_msports_n[three_min_pos]

        final_max = [user_max_sport, two_max_sport, three_max_sport]
        final_min = [user_min_sport, two_min_sport, three_min_sport]

 
    else:


        user_df['Age'] = user_df['Age'].astype(float, errors = 'raise')
        user_df['Height'] = user_df['Height'].astype(float, errors = 'raise')
        user_df['Weight'] = user_df['Weight'].astype(float, errors = 'raise')

        user_df["Age"] = pd.cut(user_df["Age"], fAge_bins, labels=bin_names)
        user_df["Height"] = pd.cut(user_df["Height"], fHeight_bins, labels=bin_names)
        user_df["Weight"] = pd.cut(user_df["Weight"], fWeight_bins, labels=bin_names)
        user_df["NOC"] = 'NOC_' + user_df["NOC"].astype(str)

        user_df["Age"] = 'Age_' + user_df["Age"].astype(str)
        user_df["Height"] = 'Height_' + user_df["Height"].astype(str)
        user_df["Weight"] = 'Weight_' + user_df["Weight"].astype(str)

        dummy_col = list(fRegression_empty.columns.values.tolist())

        user_dummies_data = []

        for col in dummy_col:
            if col == user_df.at[0, 'Age']:
                user_dummies_data.append(1)
            elif col == user_df.at[0, 'Height']:
                user_dummies_data.append(1)
            elif col == user_df.at[0, 'Weight']:
                user_dummies_data.append(1)
            elif col == user_df.at[0, 'NOC']:
                user_dummies_data.append(1)
            else: user_dummies_data.append(0)

        user_df_reg = fRegression_empty.append(pd.Series(user_dummies_data, index=dummy_col), ignore_index=True)
        user_df_reg = user_df_reg.drop(['Unnamed: 0'], axis=1)

        fsports = []
        user_pred_list = []

        fBasketball_user_pred = fBasketball_clf.predict_proba(user_df_reg)
        fBasketball_user_pred = fBasketball_user_pred[:,1]
        fsports.append('Basketball')
        user_pred_list.append(fBasketball_user_pred)

        fJudo_user_pred = fJudo_clf.predict_proba(user_df_reg)
        fJudo_user_pred = fJudo_user_pred[:,1]
        fsports.append('Judo')
        user_pred_list.append(fJudo_user_pred)

        fBadminton_user_pred = fBadminton_clf.predict_proba(user_df_reg)
        fBadminton_user_pred = fBadminton_user_pred[:,1]
        fsports.append('Badminton')
        user_pred_list.append(fBadminton_user_pred)

        fAthletics_user_pred = fAthletics_clf.predict_proba(user_df_reg)
        fAthletics_user_pred = fAthletics_user_pred[:,1]
        fsports.append('Athletics')
        user_pred_list.append(fAthletics_user_pred)

        fWeightlifting_user_pred = fWeightlifting_clf.predict_proba(user_df_reg)
        fWeightlifting_user_pred = fWeightlifting_user_pred[:,1]
        fsports.append('Weightlifting')
        user_pred_list.append(fWeightlifting_user_pred)

        fWrestling_user_pred = fWrestling_clf.predict_proba(user_df_reg)
        fWrestling_user_pred = fWrestling_user_pred[:,1]
        fsports.append('Wrestling')
        user_pred_list.append(fWrestling_user_pred)

        fRowing_user_pred = fRowing_clf.predict_proba(user_df_reg)
        fRowing_user_pred = fRowing_user_pred[:,1]
        msports.append('Rowing')
        user_pred_list.append(mRowing_user_pred)

        fSwimming_user_pred = fSwimming_clf.predict_proba(user_df_reg)
        fSwimming_user_pred = fSwimming_user_pred[:,1]
        fsports.append('Swimming')
        user_pred_list.append(fSwimming_user_pred)

        fFootball_user_pred = fFootball_clf.predict_proba(user_df_reg)
        fFootball_user_pred = fFootball_user_pred[:,1]
        fsports.append('Football')
        user_pred_list.append(fFootball_user_pred)

        fEquestrianism_user_pred = fEquestrianism_clf.predict_proba(user_df_reg)
        fEquestrianism_user_pred = fEquestrianism_user_pred[:,1]
        fsports.append('Equestrianism')
        user_pred_list.append(fEquestrianism_user_pred)

        fShooting_user_pred = fShooting_clf.predict_proba(user_df_reg)
        fShooting_user_pred = fShooting_user_pred[:,1]
        fsports.append('Shooting')
        user_pred_list.append(fShooting_user_pred)

        fGymnastics_user_pred = fGymnastics_clf.predict_proba(user_df_reg)
        fGymnastics_user_pred = fGymnastics_user_pred[:,1]
        fsports.append('Gymnastics')
        user_pred_list.append(fGymnastics_user_pred)

        fTaekwondo_user_pred = fTaekwondo_clf.predict_proba(user_df_reg)
        fTaekwondo_user_pred = fTaekwondo_user_pred[:,1]
        fsports.append('Taekwondo')
        user_pred_list.append(fTaekwondo_user_pred)

        fBoxing_user_pred = fBoxing_clf.predict_proba(user_df_reg)
        fBoxing_user_pred = fBoxing_user_pred[:,1]
        fsports.append('Boxing')
        user_pred_list.append(fBoxing_user_pred)

        fFencing_user_pred = fFencing_clf.predict_proba(user_df_reg)
        fFencing_user_pred = fFencing_user_pred[:,1]
        fsports.append('Fencing')
        user_pred_list.append(fFencing_user_pred)

        fDiving_user_pred = fDiving_clf.predict_proba(user_df_reg)
        fDiving_user_pred = fDiving_user_pred[:,1]
        fsports.append('Diving')
        user_pred_list.append(fDiving_user_pred)

        fCanoeing_user_pred = fCanoeing_clf.predict_proba(user_df_reg)
        fCanoeing_user_pred = fCanoeing_user_pred[:,1]
        fsports.append('Canoeing')
        user_pred_list.append(fCanoeing_user_pred)

        fHandball_user_pred = fHandball_clf.predict_proba(user_df_reg)
        fHandball_user_pred = fHandball_user_pred[:,1]
        fsports.append('Handball')
        user_pred_list.append(fHandball_user_pred)

        fWater_Polo_user_pred = fWater_Polo_clf.predict_proba(user_df_reg)
        fWater_Polo_user_pred = fWater_Polo_user_pred[:,1]
        fsports.append('Water Polo')
        user_pred_list.append(fWater_Polo_user_pred)

        fTennis_user_pred = fTennis_clf.predict_proba(user_df_reg)
        fTennis_user_pred = fTennis_user_pred[:,1]
        fsports.append('Tennis')
        user_pred_list.append(fTennis_user_pred)

        fCycling_user_pred = fCycling_clf.predict_proba(user_df_reg)
        fCycling_user_pred = fCycling_user_pred[:,1]
        fsports.append('Cycling')
        user_pred_list.append(fCycling_user_pred)

        fHockey_user_pred = fHockey_clf.predict_proba(user_df_reg)
        fHockey_user_pred = fHockey_user_pred[:,1]
        fsports.append('Hockey')
        user_pred_list.append(fHockey_user_pred)

        fArchery_user_pred = fArchery_clf.predict_proba(user_df_reg)
        fArchery_user_pred = fArchery_user_pred[:,1]
        fsports.append('Archery')
        user_pred_list.append(fArchery_user_pred)

        fVolleyball_user_pred = fVolleyball_clf.predict_proba(user_df_reg)
        fVolleyball_user_pred = fVolleyball_user_pred[:,1]
        fsports.append('Volleyball')
        user_pred_list.append(fVolleyball_user_pred)

        fModern_Pentathlon_user_pred = fModern_Pentathlon_clf.predict_proba(user_df_reg)
        fModern_Pentathlon_user_pred = fModern_Pentathlon_user_pred[:,1]
        fsports.append('Modern Pentathlon')
        user_pred_list.append(fModern_Pentathlon_user_pred)

        fTable_Tennis_user_pred = fTable_Tennis_clf.predict_proba(user_df_reg)
        fTable_Tennis_user_pred = fTable_Tennis_user_pred[:,1]
        fsports.append('Table Tennis')
        user_pred_list.append(fTable_Tennis_user_pred)

        fRugby_Sevens_user_pred = fRugby_Sevens_clf.predict_proba(user_df_reg)
        fRugby_Sevens_user_pred = fRugby_Sevens_user_pred[:,1]
        fsports.append('Rugby Sevens')
        user_pred_list.append(fRugby_Sevens_user_pred)

        fTrampolining_user_pred = fTrampolining_clf.predict_proba(user_df_reg)
        fTrampolining_user_pred = fTrampolining_user_pred[:,1]
        fsports.append('Trampolining')
        user_pred_list.append(fTrampolining_user_pred)

        fBeach_Volleyball_user_pred = fBeach_Volleyball_clf.predict_proba(user_df_reg)
        fBeach_Volleyball_user_pred = fBeach_Volleyball_user_pred[:,1]
        msports.append('Beach Volleyball')
        user_pred_list.append(mBeach_Volleyball_user_pred)

        fTriathlon_user_pred = fTriathlon_clf.predict_proba(user_df_reg)
        fTriathlon_user_pred = fTriathlon_user_pred[:,1]
        msports.append('Triathlon')
        user_pred_list.append(mTriathlon_user_pred)

        fGolf_user_pred = fGolf_clf.predict_proba(user_df_reg)
        fGolf_user_pred = fGolf_user_pred[:,1]
        fsports.append('Golf')
        user_pred_list.append(fGolf_user_pred)

        fRhythmic_Gymnastics_user_pred = fRhythmic_Gymnastics_clf.predict_proba(user_df_reg)
        fRhythmic_Gymnastics_user_pred = fRhythmic_Gymnastics_user_pred[:,1]
        fsports.append('Rhythmic_Gymnastics')
        user_pred_list.append(fRhythmic_Gymnastics_user_pred)

        fSoftball_user_pred = fSoftball_clf.predict_proba(user_df_reg)
        fSoftball_user_pred = fSoftball_user_pred[:,1]
        fsports.append('Softball')
        user_pred_list.append(fSoftball_user_pred)

        fSynchronized_Swimming_user_pred = fSynchronized_Swimming_clf.predict_proba(user_df_reg)
        fSynchronized_Swimming_user_pred = fSynchronized_Swimming_user_pred[:,1]
        fsports.append('Synchronized_Swimming')
        user_pred_list.append(fSynchronized_Swimming_user_pred)

        # Find max predicted sports

        user_max = max(user_pred_list)
        user_max_pos = user_pred_list.index(user_max)
        user_max_sport = msports[user_max_pos]
        
        two_user_pred_x = user_pred_list.remove(user_max)
        two_msports_x = msports.remove(user_max_sport)

        two_max = max(two_user_pred_x)
        two_max_pos = two_user_pred_x.index(two_max)
        two_max_sport = two_msports_x[two_max_pos]

        three_user_pred_x = two_user_pred_x.remove(two_max)
        three_msports_x = two_msports_x.remove(two_max_sport)

        three_max = max(three_user_pred_x)
        three_max_pos = three_user_pred_x.index(three_max)
        three_max_sport = three_msports_x[three_max_pos]       

        # Find min predicted sports

        user_min = min(user_pred_list)
        user_min_pos = user_pred_list.index(user_min)
        user_min_sport = fsports[user_min_pos]
        
        two_user_pred_n = user_pred_list.remove(user_min)
        two_msports_n = fsports.remove(user_min_sport)

        two_min = min(two_user_pred_n)
        two_min_pos = two_user_pred_n.index(two_min)
        two_min_sport = two_msports_n[two_min_pos]

        three_user_pred_n = two_user_pred_n.remove(two_min)
        three_msports_n = two_msports_n.remove(two_min_sport)

        three_min = min(three_user_pred_n)
        three_min_pos = three_user_pred_n.index(three_min)
        three_min_sport = three_msports_n[three_min_pos]

        final_max = [user_max_sport, two_max_sport, three_max_sport]
        final_min = [user_min_sport, two_min_sport, three_min_sport]   

        return (final_max, final_min)
    


    

    

