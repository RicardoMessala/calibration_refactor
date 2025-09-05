# Metodos para calcular IQR, Mean etc - Apenas criar novas def
def parameters_filter(key, data, bins, bin_range):
    parameters_bin = list()
    
    for df in data:

        # Aplicando a condição nos dataframes
        df_filtrado = df[(abs(df["cluster_"+key]) <= bins[bin_range]) &
                        (abs(df["cluster_"+key]) > bins[bin_range - 1])].dropna()
        # Armazenando o resultado no dicionário com a chave sendo o nome do dataframe
        parameters_bin.append(df_filtrado.reset_index(drop=True))

    return parameters_bin

def UseRingsKeys(dict, useRings, array):
    if useRings == 0:
        dict['shower var'] = np.array(array)
    elif useRings == 1:
        dict['rings'] = np.array(array)
    else:
        dict['quarter rings'] = np.array(array)
    return dict

def AddKeyBins(dict, bins=None, bin_range=None, varMode=None):
    dict[varMode] = (bins[bin_range-1],bins[bin_range])  
    return dict

def ReturnArray(array, key, row):
    key_array = pd.Series(dtype = object)
    key_array = pd.concat([key_array, pd.Series(array[row][key]).reset_index(drop=True)], axis=0)
    return key_array

def IqrCalculus(yte, ypred=None, mode='raw'):
    if mode == 'raw':
        temp=stats.iqr(yte, axis=None, rng=(25, 75), scale=1.0, nan_policy='omit', interpolation='linear')
    else:
        temp=stats.iqr(np.divide(pd.Series.to_numpy(yte), pd.Series.to_numpy(ypred)), axis=None, rng=(25, 75), scale=1.0, nan_policy='omit', interpolation='linear')
    return temp

def MeanCalculus(yte, ypred=None, mode='raw'):
    if mode == 'raw':
        temp = np.mean(yte)
    else:
        temp = np.mean(np.divide(pd.Series.to_numpy(yte), pd.Series.to_numpy(ypred)))
    return temp

def MeanAbsError(yte, ypred):
    return mean_absolute_error(y_true=np.array(yte), y_pred=np.array(ypred))
    
def MeanSqrtError(yte, ypred):
    return  ((1/len(ypred))*np.sum((np.array(yte) - np.array(ypred))**2))**(1/2)

def ReturnBinRange(array, key):
    temp_return = list()
    for row in  range(len(array)):
        temp = array[row][key]
        temp_return.append(temp)     
    return temp_return
def ReturnCalculusIqr(array, mode='raw', var='eta'):
    temp_return = list()
    if mode == 'raw':
        for row in  range(len(array)):
            temp_array_yte=ReturnArray(array, 'yte', row=row)
            temp=IqrCalculus(yte=temp_array_yte, mode=mode)
            temp_return.append(temp)

    elif mode != 'raw':
        for row in  range(len(array)):
            temp_array_yte=ReturnArray(array, 'yte', row=row)
            temp_array_ypred=ReturnArray(array, 'ypred', row=row)
            temp=IqrCalculus(yte=temp_array_yte, ypred=temp_array_ypred, mode=mode)
            temp_return.append(temp)
    return temp_return

def ReturnCalculusMean(array, mode='raw', var='eta'):
    temp_return = list()
    if mode == 'raw':
        for row in  range(len(array)):
            temp_array_yte=ReturnArray(array, 'yte', row=row)
            temp=MeanCalculus(yte=temp_array_yte, mode=mode)
            temp_return.append(temp)

    elif mode != 'raw':
        for row in  range(len(array)):
            temp_array_yte=ReturnArray(array, 'yte', row=row)
            temp_array_ypred=ReturnArray(array, 'ypred', row=row)
            temp=MeanCalculus(yte=temp_array_yte, ypred=temp_array_ypred, mode=mode)
            temp_return.append(temp)

    return temp_return

def ReturnMeanAbsError(array, var='eta'):
    temp_return = list()
    for row in  range(len(array)):
        temp_array_yte=ReturnArray(array, 'yte', row=row)
        temp_array_ypred=ReturnArray(array, 'ypred', row=row)
        temp=MeanAbsError(yte=temp_array_yte, ypred=temp_array_ypred)
        temp_return.append(temp)

    return temp_return

def ReturnMeanSqrtError(array, var='eta'):
    temp_return = list()
    for row in  range(len(array)):
        temp_array_yte=ReturnArray(array, 'yte', row=row)
        temp_array_ypred=ReturnArray(array, 'ypred', row=row)
        temp=MeanSqrtError(yte=temp_array_yte, ypred=temp_array_ypred)
        temp_return.append(temp)
    
    return temp_return

#Não esta funcionando ainda Retorno de caroline
def ReturnBinRangeValues(array, range_bin, bins_et, key):
    temp_return = None
    temp_dict= dict()
    temp_value_error = list()
    for row in range(1, len(bins_et)):
        temp_dict = AddKeyBins(temp_dict, bins_et=bins_et, bin_et_range=row)
        temp_value_error.append(temp_dict['et'])
        if range_bin == temp_dict['et']:
            temp_return = ReturnArray(array, key, row,)
            break
    if temp_return is None:
        raise ValueError("Invalid bin range entry:", range_bin, 
                         "Possible values for bin range:", 
                         temp_value_error)
    
    return temp_return

# === variables to plot ETA ===
Et_eta_range= ReturnBinRange(data_tree_eta['shower var'], 'eta')
IQRraw_ETA = ReturnCalculusIqr(data_tree_eta['shower var'], mode='raw', var='eta')
IQRpredR0_ETA = ReturnCalculusIqr(data_tree_eta['shower var'], mode='pred', var='eta')
MDraw_ETA = ReturnCalculusMean(data_tree_eta['shower var'], mode='raw', var='eta')
MDpredR0_ETA = ReturnCalculusMean(data_tree_eta['shower var'], mode='pred', var='eta')

IQRrawR2_ETA =ReturnCalculusIqr(data_tree_eta['rings'], mode='raw', var='eta')
IQRpredR2_ETA = ReturnCalculusIqr(data_tree_eta['rings'], mode='pred', var='eta')
MDrawR2_ETA = ReturnCalculusMean(data_tree_eta['rings'], mode='raw', var='eta')
MDpredR2_ETA = ReturnCalculusMean(data_tree_eta['rings'], mode='pred', var='eta')

IQRrawR3_ETA=ReturnCalculusIqr(data_tree_eta['quarter rings'], mode='raw', var='eta')
IQRpredR3_ETA=ReturnCalculusIqr(data_tree_eta['quarter rings'], mode='pred', var='eta')
MDrawR3_ETA=ReturnCalculusMean(data_tree_eta['quarter rings'], mode='raw', var='eta')
MDpredR3_ETA=ReturnCalculusMean(data_tree_eta['quarter rings'], mode='pred', var='eta')

e_maeR0_ETA = ReturnMeanAbsError(data_tree_eta['shower var'], var='eta')
e_rmseR0_ETA=ReturnMeanSqrtError(data_tree_eta['shower var'], var='eta')
e_maeR2_ETA=ReturnMeanAbsError(data_tree_eta['rings'], var='eta')
e_rmseR2_ETA=ReturnMeanSqrtError(data_tree_eta['rings'], var='eta')
e_maeR3_ETA=ReturnMeanAbsError(data_tree_eta['quarter rings'], var='eta')
e_rmseR3_ETA=ReturnMeanSqrtError(data_tree_eta['quarter rings'], var='eta')

# === variables to plot ET ===
Eta_et_range= ReturnBinRange(data_tree_et['shower var'], 'et')
IQRraw_ET=ReturnCalculusIqr(data_tree_et['shower var'], mode='raw', var='et')
IQRpredR0_ET=ReturnCalculusIqr(data_tree_et['shower var'], mode='pred', var='et')
MDraw_ET=ReturnCalculusMean(data_tree_et['shower var'], mode='raw', var='et')
MDpredR0_ET=ReturnCalculusMean(data_tree_et['shower var'], mode='pred', var='et')

IQRrawR2_ET=ReturnCalculusIqr(data_tree_et['rings'], mode='raw', var='et')
IQRpredR2_ET=ReturnCalculusIqr(data_tree_et['rings'], mode='pred', var='et')
MDrawR2_ET=ReturnCalculusMean(data_tree_et['rings'], mode='raw', var='et')
MDpredR2_ET=ReturnCalculusMean(data_tree_et['rings'], mode='pred', var='et')

IQRrawR3_ET=ReturnCalculusIqr(data_tree_et['quarter rings'], mode='raw', var='et')
IQRpredR3_ET=ReturnCalculusIqr(data_tree_et['quarter rings'], mode='pred', var='et')
MDrawR3_ET=ReturnCalculusMean(data_tree_et['quarter rings'], mode='raw', var='et')
MDpredR3_ET=ReturnCalculusMean(data_tree_et['quarter rings'], mode='pred', var='et')

e_maeR0_ET=ReturnMeanAbsError(data_tree_et['shower var'], var='et')
e_rmseR0_ET=ReturnMeanSqrtError(data_tree_et['shower var'], var='et')
e_maeR2_ET=ReturnMeanAbsError(data_tree_et['rings'], var='et')
e_rmseR2_ET=ReturnMeanSqrtError(data_tree_et['rings'], var='et')
e_maeR3_ET=ReturnMeanAbsError(data_tree_et['quarter rings'], var='et')
e_rmseR3_ET=ReturnMeanSqrtError(data_tree_et['quarter rings'], var='et')

print(IQRraw_ETA)
print(IQRpredR0_ETA)
print(IQRpredR2_ETA)
print(IQRpredR3_ETA)

# Range in ETA  
etaXPos_ETA = []
etaXDelta_ETA = []
for r in Et_eta_range:       
    pto_med = (r[1]+r[0])/2
    diff = (r[1]-r[0])/2
    etaXPos_ETA.append(pto_med)
    etaXDelta_ETA.append(diff)

# Range in ET 
etaXPos_ET = []
etaXDelta_ET = []
for r in Eta_et_range:       
    pto_med = (r[1]+r[0])/2
    diff = (r[1]-r[0])/2
    etaXPos_ET.append(pto_med/1000)
    etaXDelta_ET.append(diff/1000)