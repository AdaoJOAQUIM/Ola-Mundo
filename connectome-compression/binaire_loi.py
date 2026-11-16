# Fichier binaire minimal et autonome de la loi : indices des 20 entrees, indices des sorties (ecarts), parametres en 16 bits.
import numpy as np, simlib as S, lzma, os
P=np.load('loi_cerveau_params.npz'); out=P['out']
gaps=np.diff(np.r_[0,out]); print("plus grand ecart entre sorties :",gaps.max())
for dd in (3,4,6):
    A,C,b=(P[f'{k}{dd}'].astype(np.float16) for k in 'ACb')
    blob=(np.array([dd,len(out)],np.uint16).tobytes()+np.array(S.SUGAR,np.uint32).tobytes()
          +gaps.astype(np.uint32 if gaps.max()>65535 else np.uint16).tobytes()+A.tobytes()+C.tobytes()+b.tobytes())
    fn=f'loi_sucre_d{dd}.bin'; open(fn,'wb').write(blob); xz=len(lzma.compress(blob,preset=9|lzma.PRESET_EXTREME))
    print(f"d={dd}: fichier {len(blob):,} octets (xz : {xz:,}) -> 92 613 300 / taille = {92613300/len(blob):,.0f} : 1  (xz : {92613300/xz:,.0f} : 1)")
