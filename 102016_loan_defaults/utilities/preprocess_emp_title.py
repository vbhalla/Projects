import pandas as pd
import numpy as np
from datetime import datetime
from time import time
import string
import re
    
def preprocess(dfc):

    dfc.emp_title = dfc.emp_title.astype('str')
    
    dfc.emp_title.fillna(value=0,inplace=True)
    dfc.emp_title.fillna(value=0.0,inplace=True)
    
    
    def normalize(s):
        for p in string.punctuation:
            s = s.replace(p, '')
        return s.lower().strip()
    
    dfc['f_norm_emp_title']=dfc.emp_title.map(lambda s: normalize(s))

    #regex
    dfc['f_norm_emp_title'] = dfc.f_norm_emp_title.map(lambda x: re.sub(r'[^\x00-\x7F]+','', x))
    
    #standardize some acronymns
    dfc.ix[(dfc.f_norm_emp_title == "rn"), 'f_norm_emp_title']="nurse"
    dfc.ix[(dfc.f_norm_emp_title == "registered nurse"), 'f_norm_emp_title']="nurse"
    dfc.ix[(dfc.f_norm_emp_title == "lpn"), 'f_norm_emp_title']="nurse"
    dfc.ix[(dfc.f_norm_emp_title == "lvn"), 'f_norm_emp_title']="nurse"
    dfc.ix[(dfc.f_norm_emp_title == "pca"), 'f_norm_emp_title']="nurse"
    dfc.ix[(dfc.f_norm_emp_title == "cna"), 'f_norm_emp_title']="nurse"
    dfc.ix[(dfc.f_norm_emp_title == "crna"), 'f_norm_emp_title']="nurse"

    dfc.ix[(dfc.f_norm_emp_title == "hr"), 'f_norm_emp_title']="human resources"

    dfc.ix[(dfc.f_norm_emp_title == "mgr"), 'f_norm_emp_title']="manager"

    dfc.ix[(dfc.f_norm_emp_title == "VP"), 'f_norm_emp_title']="vice president"
    dfc.ix[(dfc.f_norm_emp_title == "AVP"), 'f_norm_emp_title']="vice president"
    dfc.ix[(dfc.f_norm_emp_title == "SVP"), 'f_norm_emp_title']="vice president"

    dfc.ix[(dfc.f_norm_emp_title == "GM"), 'f_norm_emp_title']="general manager"

    dfc.ix[(dfc.f_norm_emp_title == "cpa"), 'f_norm_emp_title']="certified public accountant"

    dfc.ix[(dfc.f_norm_emp_title == "executive assistant"), 'f_norm_emp_title']="admnistrative assistant"


    dfc.ix[(dfc.f_norm_emp_title == "it"), 'f_norm_emp_title']="information technology"

    dfc.ix[(dfc.f_norm_emp_title == "ceo"), 'f_norm_emp_title']="chief executive officer"
    dfc.ix[(dfc.f_norm_emp_title == "coo"), 'f_norm_emp_title']="chief operating officer"
    dfc.ix[(dfc.f_norm_emp_title == "cmo"), 'f_norm_emp_title']="chief marketing officer"
    dfc.ix[(dfc.f_norm_emp_title == "cto"), 'f_norm_emp_title']="chief technology officer"
    dfc.ix[(dfc.f_norm_emp_title == "cfo"), 'f_norm_emp_title']='chief financial officer'
    dfc.ix[(dfc.f_norm_emp_title == "cio"), 'f_norm_emp_title']='chief investment officer'

    return dfc
