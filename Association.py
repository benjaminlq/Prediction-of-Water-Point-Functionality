import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.graphics.mosaicplot import mosaic

class Chisq_Stats():
    def __init__(self,data):
        self.data = data
        self.columns = list(data.columns)
        p_value_mat = np.zeros((len(self.columns),len(self.columns)))
        crimer_v_mat = np.zeros((len(self.columns),len(self.columns)))
        df_mat = np.zeros((len(self.columns),len(self.columns)))
        
        for i in range(len(self.columns)):
            for j in range(len(self.columns)):
                crimer_v_mat[i][j] = self.crimer_v(self.columns[i],
                                                  self.columns[j])
                p_value_mat[i][j] = self.chisquare(self.columns[i],
                                                   self.columns[j])[1]
                df_mat[i][j] = self.chisquare(self.columns[i],
                                        self.columns[j])[2]
                
        self.crimer_v_mat = pd.DataFrame(crimer_v_mat,
                                         index = self.columns,
                                         columns = self.columns)
        
        self.p_value = pd.DataFrame(p_value_mat,
                                         index = self.columns,
                                         columns = self.columns)
        
    def chisquare(self,col1,col2):
        table = self.contingency(col1, col2)

        return stats.chi2_contingency(table,
                                      correction = False)
    
    def contingency(self,col1,col2):
    ## Return contigency table
        df = pd.crosstab(self.data[col1],
                         self.data[col2],
                         margins = False)
        return df.values
    
    def crimer_v(self,col1,col2):
        chi_stats = self.chisquare(col1,col2)
        n = np.sum(self.contingency(col1, col2))
        minDim = min(self.contingency(col1, col2).shape) - 1
        return np.sqrt((chi_stats[0]/n)/minDim)
    
    def mosaic_plot(self,col1,col2):
        mosaic(self.data,[col1,col2],title = col1 + " vs " + col2)        