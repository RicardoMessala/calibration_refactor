import pandas as pd
import numpy as np

# Metodos para atributos dos rings - Apenas criar novas def
# Function to compute the relevant variables

def calc_sv(e237, e277, rings):
	Reta = e237/e277
	E_E1E2 = rings.iloc[:,8:72].sum(axis=1)/rings.iloc[:,72:80].sum(axis=1)
	E_EM = rings.iloc[:,8:88].sum(axis=1)
	E_E0E1 = rings.iloc[:,0:8].sum(axis=1)/rings.iloc[:,8:72].sum(axis=1)
	RHad = rings.iloc[:,88:92].sum(axis=1)/E_EM #E_H1E

	return Reta, E_E1E2, E_EM, E_E0E1, RHad

def CalcRings(rings):
    RingsPS = rings.iloc[:,0:4]
    RingsEM1 = rings.iloc[:,8:41]
    RingsEM2 = rings.iloc[:,72:76]
    RingsEM3 = rings.iloc[:,80:84] 
    RingsHD = rings.iloc[:,88:94]
    Rings = pd.concat([RingsPS, RingsEM1, RingsEM2, RingsEM3, RingsHD], axis=1)

    return  Rings

def calc_asym_weights(qr,rings):
   qrPS=qr.iloc[:,0:29]#[0:13] #[0:29] 4 stdR
   qrEM1=qr.iloc[:,29:282]#[29:58] #[29:282] 8 stdR
   qrEM2=qr.iloc[:,282:311] #8 stdR
   qrEM3=qr.iloc[:,311:340]#[311:324] #[311:340] 4stdR
   RingsHD=rings.iloc[:,88:92] #4 stdR
   QRings = pd.concat([qrPS, qrEM1, qrEM2, qrEM3, RingsHD], axis=1)
   
   return QRings    

def calc_asym_weights_delta(qr, rings, cluster_eta, cluster_phi, delta_eta_calib, delta_phi_calib, hotCellEta, hotCellPhi, mc_et):

    qrEM1in = qr.iloc[:, 29:41]  # 8 stdR
    qrEM2in = qr.iloc[:, 42:46]  # 8 stdR
    qrEM3in = qr.iloc[:, 311:323]  # 4 stdR

    qrH1 = qr.iloc[:, 340:348]
    qrH2 = qr.iloc[:, 353:357]
    qrH3 = qr.iloc[:, 366:370]
	
    RingsPS = rings.iloc[:, 0:4]
    RingsEM1 = rings.iloc[:, 8:41]
    RingsEM2 = rings.iloc[:, 72:76]
    RingsEM3 = rings.iloc[:, 80:84] 
    RingsHD = rings.iloc[:, 88:94]

    phi_p_eta_p_EM1 = qrEM1in.iloc[:, 0:10:4].to_numpy()
    phi_p_eta_m_EM1 = qrEM1in.iloc[:, 1:11:4].to_numpy()
    phi_m_eta_p_EM1 = qrEM1in.iloc[:, 2:12:4].to_numpy()
    phi_m_eta_m_EM1 = qrEM1in.iloc[:, 3:13:4].to_numpy()

    phi_p_eta_p = qrEM2in.iloc[:, 0:2:4].to_numpy()
    phi_p_eta_m = qrEM2in.iloc[:, 1:3:4].to_numpy()
    phi_m_eta_p = qrEM2in.iloc[:, 2:4:4].to_numpy()
    phi_m_eta_m = qrEM2in.iloc[:, 3:5:4].to_numpy()

    phi_p_eta_p_EM3 = qrEM3in.iloc[:, 0:10:4].to_numpy()
    phi_p_eta_m_EM3 = qrEM3in.iloc[:, 1:11:4].to_numpy()
    phi_m_eta_p_EM3 = qrEM3in.iloc[:, 2:12:4].to_numpy()
    phi_m_eta_m_EM3 = qrEM3in.iloc[:, 3:13:4].to_numpy()

    phi_p_eta_p_H1 = qrH1.iloc[:, 0:6:4].to_numpy()
    phi_p_eta_m_H1 = qrH1.iloc[:, 1:7:4].to_numpy()
    phi_m_eta_p_H1 = qrH1.iloc[:, 2:8:4].to_numpy()
    phi_m_eta_m_H1 = qrH1.iloc[:, 3:9:4].to_numpy()

    phi_p_eta_p_H2 = qrH2.iloc[:, 0:2:4].to_numpy()
    phi_p_eta_m_H2 = qrH2.iloc[:, 1:3:4].to_numpy()
    phi_m_eta_p_H2 = qrH2.iloc[:, 2:4:4].to_numpy()
    phi_m_eta_m_H2 = qrH2.iloc[:, 3:5:4].to_numpy()

    phi_p_eta_p_H3 = qrH3.iloc[:, 0:2:4].to_numpy()
    phi_p_eta_m_H3 = qrH3.iloc[:, 1:3:4].to_numpy()
    phi_m_eta_p_H3 = qrH3.iloc[:, 2:4:4].to_numpy()
    phi_m_eta_m_H3 = qrH3.iloc[:, 3:5:4].to_numpy()

    delta_eta_phi_p_EM1 = pd.DataFrame(np.subtract(phi_p_eta_p_EM1,phi_p_eta_m_EM1)).reset_index(drop=True)
    delta_eta_phi_m_EM1 = pd.DataFrame(np.subtract(phi_m_eta_p_EM1,phi_m_eta_m_EM1)).reset_index(drop=True)
    delta_phi_eta_p_EM1 = pd.DataFrame(np.subtract(phi_p_eta_p_EM1,phi_m_eta_p_EM1)).reset_index(drop=True)
    delta_phi_eta_m_EM1 = pd.DataFrame(np.subtract(phi_p_eta_m_EM1,phi_m_eta_m_EM1)).reset_index(drop=True)

    delta_eta_phi_p = pd.DataFrame(np.subtract(phi_p_eta_p,phi_p_eta_m)).reset_index(drop=True)
    delta_eta_phi_m = pd.DataFrame(np.subtract(phi_m_eta_p,phi_m_eta_m)).reset_index(drop=True)
    delta_phi_eta_p = pd.DataFrame(np.subtract(phi_p_eta_p,phi_m_eta_p)).reset_index(drop=True)
    delta_phi_eta_m = pd.DataFrame(np.subtract(phi_p_eta_m,phi_m_eta_m)).reset_index(drop=True)
    
    delta_eta_phi_p_EM3 = pd.DataFrame(np.subtract(phi_p_eta_p_EM3,phi_p_eta_m_EM3)).reset_index(drop=True)
    delta_eta_phi_m_EM3 = pd.DataFrame(np.subtract(phi_m_eta_p_EM3,phi_m_eta_m_EM3)).reset_index(drop=True)
    delta_phi_eta_p_EM3 = pd.DataFrame(np.subtract(phi_p_eta_p_EM3,phi_m_eta_p_EM3)).reset_index(drop=True)
    delta_phi_eta_m_EM3 = pd.DataFrame(np.subtract(phi_p_eta_m_EM3,phi_m_eta_m_EM3)).reset_index(drop=True)
    
    delta_eta_phi_p_H1 = pd.DataFrame(np.subtract(phi_p_eta_p_H1, phi_p_eta_m_H1)).reset_index(drop=True)
    delta_eta_phi_m_H1 = pd.DataFrame(np.subtract(phi_m_eta_p_H1, phi_m_eta_m_H1)).reset_index(drop=True)
    delta_phi_eta_p_H1 = pd.DataFrame(np.subtract(phi_p_eta_p_H1, phi_m_eta_p_H1)).reset_index(drop=True)
    delta_phi_eta_m_H1 = pd.DataFrame(np.subtract(phi_p_eta_m_H1, phi_m_eta_m_H1)).reset_index(drop=True)

    delta_eta_phi_p_H2 = pd.DataFrame(np.subtract(phi_p_eta_p_H2, phi_p_eta_m_H2)).reset_index(drop=True)
    delta_eta_phi_m_H2 = pd.DataFrame(np.subtract(phi_m_eta_p_H2, phi_m_eta_m_H2)).reset_index(drop=True)
    delta_phi_eta_p_H2 = pd.DataFrame(np.subtract(phi_p_eta_p_H2, phi_m_eta_p_H2)).reset_index(drop=True)
    delta_phi_eta_m_H2 = pd.DataFrame(np.subtract(phi_p_eta_m_H2, phi_m_eta_m_H2)).reset_index(drop=True)

    delta_eta_phi_p_H3 = pd.DataFrame(np.subtract(phi_p_eta_p_H3, phi_p_eta_m_H3)).reset_index(drop=True)
    delta_eta_phi_m_H3 = pd.DataFrame(np.subtract(phi_m_eta_p_H3, phi_m_eta_m_H3)).reset_index(drop=True)
    delta_phi_eta_p_H3 = pd.DataFrame(np.subtract(phi_p_eta_p_H3, phi_m_eta_p_H3)).reset_index(drop=True)
    delta_phi_eta_m_H3 = pd.DataFrame(np.subtract(phi_p_eta_m_H3, phi_m_eta_m_H3)).reset_index(drop=True)

    QRings = pd.concat([
        delta_eta_phi_p_EM1, delta_eta_phi_m_EM1,
        delta_phi_eta_p_EM1, delta_phi_eta_m_EM1,
        delta_eta_phi_p, delta_eta_phi_m,
        delta_phi_eta_p, delta_phi_eta_m,
        delta_eta_phi_p_EM3, delta_eta_phi_m_EM3,
        delta_phi_eta_p_EM3, delta_phi_eta_m_EM3,
        delta_eta_phi_p_H1, delta_eta_phi_m_H1,
        delta_phi_eta_p_H1, delta_phi_eta_m_H1,
        delta_eta_phi_p_H2, delta_eta_phi_m_H2,
        delta_phi_eta_p_H2, delta_phi_eta_m_H2,
        delta_eta_phi_p_H3, delta_eta_phi_m_H3,
        delta_phi_eta_p_H3, delta_phi_eta_m_H3,
        RingsPS, RingsEM1, RingsEM2, RingsEM3, RingsHD,
    ], axis=1)

    QRings.columns = range(QRings.shape[1])

    QRings = pd.concat([QRings, 
                        delta_eta_calib, delta_phi_calib,
                        hotCellEta, hotCellPhi], axis=1)

    return QRings
